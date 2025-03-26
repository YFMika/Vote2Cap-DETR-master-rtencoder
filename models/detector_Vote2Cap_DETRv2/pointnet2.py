import torch
from torch import nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from typing import List

from models.detector_Vote2Cap_DETRv2.basic import Point_Conv
from models.detector_Vote2Cap_DETRv2.transformer import MaskedTransformerEncoder

    
# class Pointnet2(nn.Module):
#     def __init__(
#         self,
#         npoints: List[int],
#         mlps: List[List[int]],
#         radii: List[float],
#         nsamples: List[int],
#     ):
#         super().__init__()
        
#         assert len(radii) == len(nsamples) == len(mlps)

#         self.layers = nn.ModuleList()
#         for i in range(len(radii)):
#             npoint = npoints[i]
#             mlp = mlps[i]
#             radius = radii[i]
#             nsample = nsamples[i]
#             self.layers.append(PointnetSAModuleVotes(mlp=mlp, npoint=npoint, radius=radius, nsample=nsample, normalize_xyz=True))


#     def forward(self, xyz, features, return_all: bool = True):
#         if return_all:
#             l_xyzs = []
#             l_features = []

#         p_inds_list = []

#         l_xyz = xyz
#         l_feature = features
#         for layer in self.layers:
#             l_xyz, l_feature, p_inds = layer(l_xyz, l_feature)
#             p_inds = p_inds.to(dtype=torch.long)
#             if len(p_inds_list) > 0:
#                 p_inds_list.append(torch.gather(p_inds_list[-1], dim=1, index=p_inds))
#             else:
#                 p_inds_list.append(p_inds)
#             if return_all:
#                 l_xyzs.append(l_xyz)
#                 l_features.append(l_feature.transpose(1, 2))

#         if return_all:
#             return l_xyzs, l_features, p_inds_list
#         return l_xyz, l_feature.transpose(1, 2), p_inds_list[-1]


class Pointnet2(nn.Module):
    def __init__(
        self,
        psa: PointnetSAModuleVotes,
        masked_encoder: MaskedTransformerEncoder,
    ):
        super().__init__()
        
        self.psa = psa
        self.masked_encoder = masked_encoder


    def forward(self, xyz, features):
        l_xyzs = []
        p_inds_list = []

        l_xyz = xyz
        l_feature = features

        l_xyz, l_feature, p_inds = self.psa(l_xyz, l_feature)
        p_inds_list.append(p_inds)
        l_xyzs.append(l_xyz)

        l_feature = l_feature.permute(2, 0, 1)
        xyzs, outputs, xyz_indses = self.masked_encoder(
            l_feature, xyz=l_xyz
        )
        l_xyzs = l_xyzs + xyzs
        p_inds_list = p_inds_list + xyz_indses

        for i in range(1, len(p_inds_list)):
            p_inds_list[i] = torch.gather(p_inds_list[i - 1], 1, p_inds_list[i].long())

        for i in range(len(outputs)):
            outputs[i] = outputs[i].transpose(0, 1)

        return l_xyzs, outputs, p_inds_list