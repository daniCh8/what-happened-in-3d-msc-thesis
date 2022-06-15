import torch
import torch.nn as nn


class spatial_info_module(nn.Module):
    def _create_coord(self, scene_feat):
        batch_size, _, l = scene_feat.size()
        coord_map = scene_feat.new_zeros(1, l)
        for i in range(l):
            coord_map[0][i] = (i * 2. / l) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, scene_feat):
        coord_map = self._create_coord(scene_feat)
        scene_feat_aug = torch.cat([scene_feat, coord_map], dim=1)
        return scene_feat_aug
