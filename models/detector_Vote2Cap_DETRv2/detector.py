import math, os
from functools import partial

import numpy as np
import torch
from torch import nn, Tensor
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

from utils.misc import huber_loss
from utils.pc_util import scale_points, shift_scale_points
from datasets.scannet import BASE
from typing import Dict

from models.detector_Vote2Cap_DETRv2.config import model_config
from models.detector_Vote2Cap_DETRv2.criterion import build_criterion
from models.detector_Vote2Cap_DETRv2.helpers import GenericMLP

from models.detector_Vote2Cap_DETRv2.vote_query import VoteQuery

from models.detector_Vote2Cap_DETRv2.position_embedding import PositionEmbeddingCoordsSine

from models.detector_Vote2Cap_DETRv2.transformer import (
    MaskedTransformerEncoder, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer
)
from typing import Optional
from models.detector_Vote2Cap_DETRv2.pointnet2 import Pointnet2
from models.detector_Vote2Cap_DETRv2.basic import HybridEncoder

    
class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model_Vote2Cap_DETR(nn.Module):
    
    def __init__(
        self,
        tokenizer,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        criterion=None,
        is_pretrain=True
    ):
        super().__init__()
        self.is_pretrain = is_pretrain
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = num_queries
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        
        self.revote_layers = [0, 1, 2]
        self.revoting_module = nn.ModuleDict({
            f'layer-{layer_id}': nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.LayerNorm(decoder_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(decoder_dim, 3)
            ) for layer_id in self.revote_layers
        })
        
        self.vote_query_generator = VoteQuery(decoder_dim, num_queries)
        
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        if is_pretrain is False:
            self.interim_proj = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, decoder_dim),
                    nn.LayerNorm(decoder_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(decoder_dim, decoder_dim),
                    nn.Dropout(p=0.3)
                ) for _ in self.decoder.layers
            ])
        
        
        self.mlp_heads = self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.box_processor = BoxProcessor(dataset_config)
        self.criterion = criterion
        


    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        return nn.ModuleDict(mlp_heads)


    def _break_up_pc(self, pc):
        # pc may contain color/normals.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features


    def run_encoder(self, point_clouds, input_range):
        xyz, features = self._break_up_pc(point_clouds)
        
        ## pointcloud tokenization
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints
        l_xyzs, l_features, p_inds = self.tokenizer(xyz, features)

        enc_xyz = torch.cat(l_xyzs[-3:], dim=1)
        enc_inds = torch.cat(p_inds[-3:], dim=1)

        # xyz points are in batch x npoint x channel order
        l_features = self.encoder(l_xyzs[-3:], l_features[-3:], input_range)
        enc_features = torch.cat(l_features[-3:], dim=1)
        
        return enc_xyz, enc_features, enc_inds


    def run_decoder(self, tgt, memory, enc_xyz, query_xyz, query_outputs, input_range):
        
        # batch x channel x npenc
        enc_pos = self.pos_embedding(enc_xyz, input_range=input_range)
        enc_pos = enc_pos.permute(2, 0, 1)
        
        output = tgt.repeat(2, 1, 1)
        
        intermediate = []
        attns = []
        layer_query_xyz = [query_xyz]
        
        tgt_mask = torch.zeros(output.shape[0], output.shape[0]).to(output.device)
        tgt_mask[:self.num_queries, self.num_queries:] = 1
        tgt_mask[self.num_queries:, :self.num_queries] = 1
        
        tgt_mask = tgt_mask.bool()
        
        for dec_layer_id, layer in enumerate(self.decoder.layers):
            
            query_pos = self.pos_embedding(query_xyz, input_range=input_range)
            query_pos = self.query_projection(query_pos)
            query_pos = query_pos.permute(2, 0, 1)
            
            query_pos = query_pos.repeat(2, 1, 1)
            
            if self.is_pretrain is False:
                output = torch.cat(
                    [
                        output[:self.num_queries], 
                        output[self.num_queries:] + self.interim_proj[dec_layer_id](output[:self.num_queries])
                    ],
                    dim=0
                )
            else:
                pass
            
            output, attn = layer(
                output, 
                memory, 
                pos=enc_pos, 
                query_pos=query_pos, 
                tgt_mask=tgt_mask,
                return_attn_weights=True
            )
            
            # interaction
            intermediate.append(self.decoder.norm(output))
            attns.append(attn)
            
            # ==== revote to object center: 
            #   ntoken x batch x channel -> batch x ntoken x channel
            if dec_layer_id in self.revote_layers:
                step_shift = self.revoting_module[f'layer-{dec_layer_id}'](
                    intermediate[-1][:self.num_queries].permute(1, 0, 2)
                )
                query_xyz = query_xyz + 0.2 * torch.sigmoid(step_shift) - 0.1
            
            layer_query_xyz.append(query_xyz)

        attns = torch.stack(attns)
        intermediate = torch.stack(intermediate)
        layer_query_xyz = torch.stack(layer_query_xyz)
        
        return layer_query_xyz, intermediate, attns


    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: num_layers x batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz[l], point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, is_eval: bool=False):
        
        point_clouds = inputs["point_clouds"]
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        
        ## feature encoding
        # encoder features: batch x npoints x channel -> batch x channel x npoints
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds, input_range=point_cloud_dims)
        enc_features = enc_features.permute(0, 2, 1).contiguous()
        
        ## vote query generation
        query_outputs = self.vote_query_generator(enc_xyz, enc_features)
        query_outputs['seed_inds'] = enc_inds
        query_xyz = query_outputs['query_xyz']
        query_features = query_outputs["query_features"]
        
        
        ## decoding
        # batch x channel x npenc
        enc_features = self.encoder_to_decoder_projection(enc_features)

        # decoder expects: npoints x batch x channel
        enc_features = enc_features.permute(2, 0, 1)
        tgt = query_features.permute(2, 0, 1)
        
        layer_query_xyz, box_features, attn = self.run_decoder(
            tgt, enc_features, enc_xyz, query_xyz, query_outputs, input_range=point_cloud_dims
        )   # nlayers x nqueries x batch x channel
        
        
        # batch x nquery
        query_outputs['revote_seed_inds'] = torch.gather(enc_inds, 1, query_outputs['sample_inds'].long())
        query_outputs['revote_seed_xyz'] = \
            torch.gather(enc_xyz, dim=1, index=query_outputs['sample_inds'].long()[..., None].repeat(1, 1, 3))
        query_outputs['revote_vote_xyz'] = layer_query_xyz[1:]  # nlayers x batch x nqueries x 3
        

        box_predictions = self.get_box_predictions(
            layer_query_xyz, point_cloud_dims, box_features[:, :self.num_queries]
        )
        
        if self.criterion is not None and is_eval is False:
            (
                box_predictions['outputs']['assignments'], 
                box_predictions['outputs']['loss'], 
                _
            ) = self.criterion(query_outputs, box_predictions, inputs)
            
        
        query_feat = torch.cat(
            [
                box_features[[-1], :self.num_queries],  # 1 x nquery x batch x channel
                box_features[[-1], self.num_queries:]   # 1 x nquery x batch x channel
            ],
            dim=0
        ).permute(0, 2, 1, 3)
        
        box_predictions['outputs'].update({
            'prop_features': query_feat,    # ntoken x batch x nqueries x channel
            'enc_features': enc_features.permute(1, 0, 2),      # batch x npoints x channel
            'enc_xyz': enc_xyz,      # batch x npoints x 3
            'query_xyz': layer_query_xyz[-1],  # batch x nqueries x 3
        })
        
        return box_predictions['outputs']



def build_preencoder(cfg):
    mlp_dims = [cfg.in_channel, 64, 128, cfg.enc_dim]
    psa = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=cfg.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    encoder_layer = TransformerEncoderLayer(
        d_model=cfg.enc_dim,
        nhead=cfg.enc_nhead,
        dim_feedforward=cfg.enc_ffn_dim,
        dropout=cfg.enc_dropout,
        activation=cfg.enc_activation,
    )
    masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
    masked_encoder = MaskedTransformerEncoder(
        cfg=cfg,
        encoder_layer=encoder_layer,
        num_layers=3,
        masking_radius=masking_radius,
    )
    preencoder = Pointnet2(
        psa=psa,
        masked_encoder=masked_encoder,
    )
    return preencoder


def build_encoder(cfg):
    if cfg.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.enc_dim,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation=cfg.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=cfg.enc_nlayers
        )
    elif cfg.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.enc_dim,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation=cfg.enc_activation,
        )
        encoder = HybridEncoder(
            encoder_layer,
            in_channels=[cfg.enc_dim, cfg.enc_dim, 2 * cfg.enc_dim],
            hidden_dim=256,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            expansion=1.0,
            depth_mult=1.0,
            act=cfg.csp_activation,
            position_embedding="fourier",
            eval_spatial_size=None
        )
    else:
        raise ValueError(f"Unknown encoder type {cfg.enc_type}")
    return encoder


def build_decoder(cfg):
    decoder_layer = TransformerDecoderLayer(
        d_model=cfg.dec_dim,
        nhead=cfg.dec_nhead,
        dim_feedforward=cfg.dec_ffn_dim,
        dropout=cfg.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=cfg.dec_nlayers, return_intermediate=True
    )
    return decoder


def detector(args, dataset_config):
    cfg = model_config(args, dataset_config)
    
    tokenizer = build_preencoder(cfg)
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    
    criterion = build_criterion(cfg, dataset_config)
    is_pretrain = args.dataset == 'scannet'
    
    model = Model_Vote2Cap_DETR(
        tokenizer,
        encoder,
        decoder,
        cfg.dataset_config,
        encoder_dim=cfg.enc_dim,
        decoder_dim=cfg.dec_dim,
        mlp_dropout=cfg.mlp_dropout,
        num_queries=cfg.nqueries,
        criterion=criterion,
        is_pretrain=is_pretrain
    )
    return model
