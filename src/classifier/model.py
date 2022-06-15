import torch
import torch.nn as nn

from binding_module.change_attention import change_attention_module
from binding_module.spatial_info import spatial_info_module

from classifier.elastic_feature_extractor import get_feature_extractor
from classifier.classifier_block import ClassifierHead


class Classifier(nn.Module):
    def __init__(self, encoder, type, input_feats=8, embed_dim_detector=None, num_classes=None, siamese=1):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.type = type
        
        if 'attention' in self.type:
            self.feature_extractor = get_feature_extractor(
                net_type=self.encoder,
                input_feats=input_feats,
                post_process=True
            )
            self.spatial_info = spatial_info_module()
            hidden_size = self.feature_extractor.output_size + 1
            self.change_attention_module = change_attention_module(
                input_dim=hidden_size*2,
                emb_dim=embed_dim_detector
            )
            self.classifier_head = ClassifierHead(
                in_c=hidden_size*3,
                num_classes=num_classes
            )
        elif 'baseline' in self.type:
            self.feature_extractor = get_feature_extractor(
                net_type=self.encoder,
                input_feats=input_feats,
                post_process=False,
                siamese=siamese
            )
            self.classifier_head = ClassifierHead(
                in_c=self.feature_extractor.mid_output_size*siamese*2,
                num_classes=num_classes
            )
        elif 'ensemble' in self.type:
            self.feature_extractor_1 = get_feature_extractor(
                net_type='minkowski',
                input_feats=input_feats,
                post_process=False,
            )
            self.feature_extractor_2 = get_feature_extractor(
                net_type='kpconv',
                input_feats=input_feats,
                post_process=False,
                o_size=512
            )
            self.classifier_head = ClassifierHead(
                in_c=1024*2,
                num_classes=num_classes,
                ensemble=True
            )
        elif 'sub' in self.type:
            self.feature_extractor = get_feature_extractor(
                net_type=self.encoder,
                input_feats=input_feats,
                post_process=False
            )
            self.classifier_head = ClassifierHead(
                in_c=1024*2,
                num_classes=num_classes,
                sub=True
            )
        else:
            self.feature_extractor = get_feature_extractor(
                net_type=self.encoder,
                input_feats=input_feats,
                post_process=False
            )
            self.classifier_head = ClassifierHead(
                in_c=1024*2,
                num_classes=num_classes
            )

    def forward(self, scans_bef, scans_aft, pre_head=False, scans_bef_2=None, scans_aft_2=None):
        if 'attention' in self.type:
            scans_bef = self.feature_extractor(scans_bef)
            scans_aft = self.feature_extractor(scans_aft)

            scans_bef = self.spatial_info(scans_bef)
            scans_aft = self.spatial_info(scans_aft)

            _, _, _, chg_feats_bef, chg_feats_aft, chg_feats_diff = \
                self.change_attention_module(scans_bef, scans_aft)
            
            feats = torch.concat(
                [chg_feats_bef, chg_feats_aft, chg_feats_diff], dim=1
            )

            return self.classifier_head(feats, pre_head, feats_2)
        elif 'multi_baseline' in self.type:
            B,S,C,H,W = scans_bef.shape
            scans_bef = self.feature_extractor(scans_bef.reshape(-1,C,H,W)).reshape(B,S,-1)
            scans_aft = self.feature_extractor(scans_aft.reshape(-1,C,H,W)).reshape(B,S,-1)

            return self.classifier_head(
                torch.concat([scans_bef, scans_aft], dim=2).reshape(B,-1), pre_head
            )
        elif 'ensemble' in self.type:
            scans_bef = self.feature_extractor_1(scans_bef)
            scans_aft = self.feature_extractor_1(scans_aft)

            scans_bef_2 = self.feature_extractor_2(scans_bef_2)
            scans_aft_2 = self.feature_extractor_2(scans_aft_2)

            b_size = scans_bef.shape[0]
            feats = torch.concat(
                [scans_bef.reshape(b_size,-1), scans_aft.reshape(b_size,-1)], dim=1
            )

            b_size = scans_bef_2.shape[0]
            feats_2 = torch.concat(
                [scans_bef_2.reshape(b_size,-1), scans_aft_2.reshape(b_size,-1)], dim=1
            )

            return self.classifier_head(feats, pre_head, feats_2)
        else:
            scans_bef = self.feature_extractor(scans_bef)
            scans_aft = self.feature_extractor(scans_aft)
            b_size = scans_bef.shape[0]

            feats = torch.concat(
                [scans_bef.reshape(b_size,-1), scans_aft.reshape(b_size,-1)], dim=1
            )

            if 'sub' in self.type:
                sub_feats = scans_aft.reshape(b_size,-1) - scans_bef.reshape(b_size,-1)
                
                return self.classifier_head(feats, pre_head, x_sub=sub_feats)
            
        return self.classifier_head(feats, pre_head)
