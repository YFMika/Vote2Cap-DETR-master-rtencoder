import math, os
from functools import partial

import numpy as np
import torch
from torch import nn, Tensor
from collections import OrderedDict
import torch.nn.functional as F
from third_party.pointnet2.pointnet2_utils import (
    furthest_point_sample,
    QueryAndGroup, GroupAll,
    gather_operation
)

from models.detector_Vote2Cap_DETRv2.helpers import (
    ACTIVATION_DICT,
    get_clones
)

from models.detector_Vote2Cap_DETRv2.position_embedding import PositionEmbeddingCoordsSine


def get_neighbor_features_with_radius(
        xyz: torch.Tensor,
        npoint: int = None,
        radius: int = None,
        nsample: int = None,
        features: torch.Tensor = None,
        inds: torch.Tensor = None,
        use_xyz: bool = True,
        normalize_xyz: bool = False, # noramlize local XYZ with radius
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False
    ):
    """
        Obtain the neighborhood features for each point from the input point cloud and features,
        considering the threshold `radius` and the maximum number of points `nsample`.

        Inputs:
            xyz: Shape (B, N, 3), the coordinates of each point.
            npoint: The number of center points
            features: Shape (B, C, N), the features of each point.
            radius: The maximum radius for neighborhood queries.
            nsample: The number of neighborhood points for each point; if fewer, padding will be applied.
            inds:  Shape (B, npoint), tensor that stores index to the xyz points (values in 0-N-1),
            use_xyz: use local XYZ with radius,
            normalize_xyz: use noramlize local XYZ with radius.
            sample_uniformly,
            ret_unique_cnt

        Outputs:
            neighbor_features: Shape (B, N, nsample, C), the neighborhood features for each point.
    """
    
    if npoint is not None:
        grouper = QueryAndGroup(radius, nsample,
            use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
            sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
    else:
        grouper = GroupAll(use_xyz, ret_grouped_xyz=True)
        
    xyz_flipped = xyz.transpose(1, 2).contiguous()
    if inds is None:
        inds = furthest_point_sample(xyz, npoint)
    else:
        assert(inds.shape[1] == npoint)
    new_xyz = gather_operation(
        xyz_flipped, inds
    ).transpose(1, 2).contiguous() if npoint is not None else None

    if not ret_unique_cnt:
        grouped_features, grouped_xyz = grouper(
            xyz, new_xyz, features
        )  # (B, C, npoint, nsample)
    else:
        grouped_features, grouped_xyz, unique_cnt = grouper(
            xyz, new_xyz, features
        )  # (B, C, npoint, nsample), (B, 3, npoint, nsample), (B, npoint)

    if not ret_unique_cnt:
        return grouped_xyz, grouped_features, inds
    else:
        return grouped_xyz, grouped_features, inds, unique_cnt
    

def nearest_neighbor_upsample(src_points, tgt_points, src_feats):
    """
        Upsample features from the original point cloud to the target point cloud (nearest neighbor interpolation)
        
        Parameters:
            src_points: Original point cloud coordinates (B, N, 3)
            tgt_points: Target point cloud coordinates (B, X, 3)
            src_feats: Features of the original point cloud (B, N, C)
        
        Returns:
            tgt_feats: Upsampled features (B, X, C)
    """
    # (B, X, N)
    dist = torch.cdist(tgt_points, src_points)
    
    # The nearest neighbor index for each target point (B, X)
    nearest_indices = dist.argmin(dim=-1)
    
    # Copy features from the original features to the target point cloud based on the indices
    tgt_feats = torch.gather(
        src_feats, 
        dim=1, 
        index=nearest_indices.unsqueeze(-1).expand(-1, -1, src_feats.shape[2])
    )
    
    return tgt_feats


class Point_Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: int,
        gather_size: int,
        stride: int = 1,
        use_bias: bool = False,
    ):
        super().__init__()

        self.radius = radius
        self.gather_size = gather_size
        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels + 3,
            out_channels,
            kernel_size=(1, gather_size),
            stride=1,
            padding=0,
            bias=use_bias,
        )

    def forward(
        self,
        points,
        features,
        inds = None,
        return_inds: bool = False,
    ):
        """
            3D vision of conv.

            Inputs:
                points: Shape (B, N, 3), the coordinates of each point.
                features: Shape (B, N, C), the features of each point.

            Outputs:
                conv_features: Shape (B, npoint, Cout), the features of each point after conv.
        """
        _, N, _ = points.shape
        npoint = int((N + self.stride - 1) / self.stride)

        if inds is None:
            _, group_features, inds = get_neighbor_features_with_radius(points.contiguous(), npoint, 
                    radius=self.radius, nsample=self.gather_size, features=features.permute(0, 2, 1).contiguous(), use_xyz=True)
        else:
            _, group_features, _ = get_neighbor_features_with_radius(points.contiguous(), npoint, 
                    radius=self.radius, nsample=self.gather_size, features=features.permute(0, 2, 1).contiguous(), inds=inds, use_xyz=True)
        inds = inds.to(dtype=torch.long)

        # (B, npoint, Cout)
        conv_features = self.conv(group_features).squeeze(-1).transpose(1, 2)

        xyz_flipped = points.transpose(1, 2)
        selected_xyz = torch.gather(xyz_flipped, 2, inds.unsqueeze(1).expand(-1, 3, -1))  # (B, 3, npoint)
        selected_xyz = selected_xyz.transpose(1, 2)  # (B, npoint, 3)

        # (B, npoint, 3), (B, npoint, Cout)
        if return_inds:
            return selected_xyz, conv_features, inds
        return selected_xyz, conv_features
    

class Point_AVGPool(nn.Module):
    def __init__(
        self,
        radius: int,
        stride: int,
        gather_size: int = None,
        ceil_mode: bool = True,
    ):
        super().__init__()

        self.radius = radius
        self.gather_size = gather_size
        self.stride = stride
        self.ceil_mode = ceil_mode


    def forward(
        self,
        points,
        features,
    ):
        """
            3D vision of avgpool.

            Inputs:
                points: Shape (B, N, 3), the coordinates of each point.
                features: Shape (B, N, C), the features of each point.

            Outputs:
                pool_features: Shape (B, npoint, Cout), the features of each point after pool.
        """
        _, N, _ = points.shape
        if self.ceil_mode:
            npoint = int((N + self.stride - 1) / self.stride)
        else:
            npoint = int(N / self.stride)

        if self.gather_size is None:
            nsample = self.stride * self.stride
        else:
            nsample = self.gather_size

        _, group_features, inds = get_neighbor_features_with_radius(points.contiguous(), npoint, 
                radius=self.radius, nsample=nsample, features=features.permute(0, 2, 1).contiguous(), use_xyz=False)
        inds = inds.to(dtype=torch.long)

        # (B, C, npoint, 1)
        group_features = F.avg_pool2d(
            group_features, kernel_size=[1, group_features.size(3)]
        )

        # (B, npoint, Cout)
        group_features = group_features.squeeze(-1).transpose(1, 2)

        xyz_flipped = points.transpose(1, 2)
        selected_xyz = torch.gather(xyz_flipped, 2, inds.unsqueeze(1).expand(-1, 3, -1))  # (B, 3, npoint)
        selected_xyz = selected_xyz.transpose(1, 2)  # (B, npoint, 3)

        # (B, npoint, 3), (B, npoint, Cout)
        return selected_xyz, group_features


# class Point_MAXPool(nn.Module):
#     def __init__(
#         self,
#         radius: int,
#         stride: int,
#         gather_size: int = None,
#         ceil_mode: bool = True,
#     ):
#         super().__init__()

#         self.radius = radius
#         self.gather_size = gather_size
#         self.stride = stride
#         self.ceil_mode = ceil_mode


#     def forward(
#         self,
#         points,
#         features,
#     ):
#         """
#             3D vision of avgpool.

#             Inputs:
#                 points: Shape (B, N, 3), the coordinates of each point.
#                 features: Shape (B, N, C), the features of each point.

#             Outputs:
#                 pool_features: Shape (B, npoint, Cout), the features of each point after pool.
#         """
#         _, N, _ = points.shape
#         if self.ceil_mode:
#             npoint = int((N + self.stride - 1) / self.stride)
#         else:
#             npoint = int(N / self.stride)

#         if self.gather_size is None:
#             nsample = self.stride * self.stride
#         else:
#             nsample = self.gather_size

#         _, group_features, inds = get_neighbor_features_with_radius(points.contiguous(), npoint, 
#                 radius=self.radius, nsample=nsample, features=features.permute(0, 2, 1).contiguous())
#         inds = inds.to(dtype=torch.long)

#         # (B, C, npoint, 1)
#         group_features = F.max_pool2d(
#             group_features, kernel_size=[1, group_features.size(3)]
#         )

#         # (B, npoint, Cout)
#         group_features = group_features.squeeze(-1).transpose(1, 2)

#         xyz_flipped = points.transpose(1, 2)
#         selected_xyz = torch.gather(xyz_flipped, 2, inds.unsqueeze(1).expand(-1, 3, -1))  # (B, 3, npoint)
#         selected_xyz = selected_xyz.transpose(1, 2)  # (B, npoint, 3)

#         # (B, npoint, 3), (B, npoint, Cout)
#         return selected_xyz, group_features
    

class ConvBNLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            radius=0.2,
            gather_size=16,
            stride=1,
            bias=False,
            act=None
        ):
        super().__init__()

        self.conv = Point_Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            radius=radius,
            gather_size=gather_size,
            stride=stride,
            use_bias=bias
        )

        self.bn = nn.BatchNorm1d(out_channels)

        self.act = ACTIVATION_DICT[act]()

    def forward(self, xyz, features):
        """
            Inputs:
                points: Shape (B, N, 3), the coordinates of each point.
                features: Shape (B, N, C), the features of each point.

            Outputs:
                new_points: Shape (B, npoint, 3), the coordinates of each center point.
                y: Shape (B, npoint, Cout), the features after convbn.
            if stride == 1, then npoint == N
        """
        new_points, y = self.conv(xyz, features)
        y = y.transpose(1, 2)
        y = self.bn(y).transpose(1, 2)
        y = self.act(y)

        return new_points, y
    

class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shortcut, expansion=1, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = out_channels 
        self.expansion = expansion

        self.branch2a = ConvBNLayer(in_channels, width, 0.1, 1, stride1, act=act)
        self.branch2b = ConvBNLayer(width, width, 0.2, 16, stride2, act=act)
        self.branch2c = ConvBNLayer(width, out_channels * self.expansion, 0.1, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', Point_AVGPool(0.2, 16, ceil_mode=True)),
                    ('conv', ConvBNLayer(in_channels, out_channels * self.expansion, 0.1, 1, 1))
                ]))
            else:
                self.short = ConvBNLayer(in_channels, out_channels * self.expansion, 0.1, 1, stride)

        self.act = nn.Identity() if act is None else ACTIVATION_DICT[act]()

    def forward(self, input):
        xyz = input[0]
        feature = input[1]

        new_xyz, out = self.branch2a(xyz, feature)
        new_xyz, out = self.branch2b(new_xyz, out)
        new_xyz, out = self.branch2c(new_xyz, out)

        if self.shortcut:
            short = feature
        else:
            _, short = self.short(xyz, feature)

        out = out + short
        out = self.act(out)

        return (new_xyz, out)
    

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 shortcut=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNLayer(in_channels, hidden_channels, 0.1, 1, 1, bias=bias, act=act)
        self.conv2 = ConvBNLayer(in_channels, hidden_channels, 0.1, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            BottleNeck(hidden_channels, hidden_channels, stride=1, shortcut=shortcut, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.do_conv3 = True
            self.conv3 = ConvBNLayer(hidden_channels, out_channels, 0.1, 1, 1, bias=bias, act=act)
        else:
            self.do_conv3 = False
            self.conv3 = nn.Identity()

    def forward(self, xyz, features):
        _, f_1 = self.conv1(xyz, features)
        _, f_1 = self.bottlenecks((xyz, f_1))
        _, f_2 = self.conv2(xyz, features)
        if self.do_conv3:
            return self.conv3(xyz, f_1 + f_2)
        return f_1 + f_2


class Point_Conv_SingleOutput(Point_Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: int,
        gather_size: int,
        stride: int = 1,
        use_bias: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            radius,
            gather_size,
            stride,
            use_bias,
        )

    def forward(
        self,
        inputs,
    ):
        points = inputs[0]
        features = inputs[1]

        _, conv_features = super().forward(points, features)

        return conv_features.transpose(1, 2)


class HybridEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 in_channels=[128, 256, 512],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 position_embedding="fourier",
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    Point_Conv_SingleOutput(in_channel, hidden_dim, radius=0.1, gather_size=1, use_bias=False),
                    nn.BatchNorm1d(hidden_dim)
                )
            )

        # encoder transformer
        self.encoder = get_clones(encoder_layer, len(use_encoder_idx))

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvBNLayer(hidden_dim, hidden_dim, 0.1, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvBNLayer(hidden_dim, hidden_dim, 0.2, 16, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=hidden_dim, pos_type=position_embedding, normalize=True
        )

    def forward(self, xyzs, feats, input_range):
        assert len(xyzs) == len(feats) == len(self.in_channels)

        proj_feats = [self.input_proj[i]((xyz, feat)).transpose(1, 2) for i, (xyz, feat) in enumerate(zip(xyzs, feats))]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # (B, N, 3)
                xyz = xyzs[enc_ind]
                xyz = xyz.contiguous()
                pos_embed = self.pos_embedding(xyz, input_range=input_range).to(xyz.device)
                pos_embed = pos_embed.permute(2, 0, 1)

                src = proj_feats[enc_ind].permute(1, 0, 2)
                memory = self.encoder[i](src, pos=pos_embed)
                proj_feats[enc_ind] = memory.permute(1, 0, 2)

        for idx in range(len(self.in_channels)):
            proj_feats[idx] = proj_feats[idx].contiguous()
            xyzs[idx] = xyzs[idx].contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        inner_xyzs = [xyzs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            xyz_high = inner_xyzs[0]
            feat_low = proj_feats[idx - 1]
            xyz_low = xyzs[idx - 1]
            xyz_high, feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](xyz_high, feat_high)
            inner_outs[0] = feat_high
            inner_xyzs[0] = xyz_high
            upsample_feat = nearest_neighbor_upsample(xyz_high, xyz_low, feat_high)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](xyz_low, torch.concat([upsample_feat, feat_low], dim=2))
            inner_outs.insert(0, inner_out)
            inner_xyzs.insert(0, xyz_low)

        outs = [inner_outs[0]]
        xyzs = [inner_xyzs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            xyz_low = xyzs[-1]
            feat_high = inner_outs[idx + 1]
            xyz_high = inner_xyzs[idx + 1]
            _, downsample_feat = self.downsample_convs[idx](xyz_low, feat_low)
            out = self.pan_blocks[idx](xyz_high, torch.concat([downsample_feat, feat_high], dim=2))
            outs.append(out)
            xyzs.append(xyz_high)

        return outs