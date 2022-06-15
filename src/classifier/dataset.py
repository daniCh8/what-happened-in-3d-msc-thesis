import random
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from MinkowskiEngine.utils import sparse_quantize
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from misc_utils import get_action_key_from_dt
from torchvision import transforms

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from DataManipulator import DataManipulator
from baseline_utils import get_same_fov_bb_image_pair_from_c_dt

class TemplateDataset(Dataset):
    def __init__(self, data_manip_path, captions_path, ssize, encoder, quantization_size, flip_prob, baseline_size, model, ref_split):
        self.data_manip = DataManipulator(data_manip_path)

        self.quantization_size = quantization_size
        self.encoder = encoder
        self.model = model
        self.ssize = ssize
        self.baseline_size = baseline_size

        if 'baseline' in self.model:
            self.mean_vec = [0.485, 0.456, 0.406]
            self.std_vec = [0.229, 0.224, 0.225]
            self.preprocess_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean_vec, std=self.std_vec)
            ])

        self.captions_path = captions_path
        self.captions = self.data_manip.create_captions_dict(
            c_path=self.captions_path, 
            filter_ssize=ssize,
            filter_baseline=baseline_size,
            ref_split=ref_split
        )
        self.flip_prob = flip_prob
        self.flip_action_dict = {
            'added':'removed',
            'removed':'added',
            'closed':'open',
            'open':'closed',
            'rotated':'rotated',
            'shifted':'shifted',
            'moved':'moved',
            'tidied up':'cluttered',
            'cluttered':'tidied up',
            'rearranged':'rearranged'
        }

        self.onehot_encoder = self.data_manip.get_captions_oh_encoder(
            c_path=captions_path
        )
        all_actions = self.onehot_encoder.transform(
            X=np.array([c['action'] for c in self.captions]).reshape((-1,1))
        )
        classes = np.argmax(all_actions, axis=1)

        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(classes),
            y=classes
        )
        self.num_classes = self.onehot_encoder.categories_[0].shape[0]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        c_dt = self.captions[index]
        action = c_dt['action']

        if 'baseline' in self.model:
            a, b = self.get_item_baseline(c_dt)
        else:
            a, b = self.get_item_pointcloud(c_dt)
        
        if action != 'decorated' and self.flip(self.flip_prob):
            x_ref = b
            x_chg = a
            action = self.flip_action_dict[action]
        else:
            x_ref = a
            x_chg = b
        
        if self.encoder == 'kpconv' or 'ensemble' in self.model:
            x_ref_kp = self.turn_to_geometry(x_ref)
            x_chg_kp = self.turn_to_geometry(x_chg)
            if 'ensemble' not in self.model:
                x_ref = x_ref_kp
                x_chg = x_chg_kp
        if self.encoder == 'minkowski' or 'ensemble' in self.model:
            x_ref_mink = sparse_quantize(
                coordinates=x_ref[:, :3],
                features=x_ref[:, 3:],
                quantization_size=self.quantization_size
            )
            x_chg_mink = sparse_quantize(
                coordinates=x_chg[:, :3],
                features=x_chg[:, 3:],
                quantization_size=self.quantization_size
            )
            if 'ensemble' not in self.model:
                x_ref = x_ref_mink
                x_chg = x_chg_mink
        if 'ensemble' in self.model:
            x_ref = (x_ref_mink, x_ref_kp)
            x_chg = (x_chg_mink, x_chg_kp)

        oh_action = self.onehot_encoder.transform(
            np.array(
                [action]
            ).reshape(-1, 1)
        ).ravel()
        return x_ref, x_chg, oh_action, c_dt
    
    def get_action_index(self, chg, action_key):
        action, idx = action_key.split('_')
        idx = int(idx)
        for i,c_dt in enumerate(self.captions):
            if c_dt['chg'] == chg and \
                c_dt['action'] == action and \
                    c_dt['id'] == idx:
                return i
        return -1

    def get_item_pointcloud(self, c_dt):
        if 'nm' not in self.model:
            x_ref = self.get_masked_x(c_dt, True)
            x_chg = self.get_masked_x(c_dt, False)
        else:
            x_ref = torch.tensor(
                np.load(
                    file=self.data_manip.get_single_npy_path(
                        c_dt=c_dt,
                        ref=True,
                        cap_path=self.captions_path,
                        ssize=self.ssize
                    )
                ).reshape((self.ssize,8))
            )
            x_chg = torch.tensor(
                np.load(
                    file=self.data_manip.get_single_npy_path(
                        c_dt=c_dt,
                        ref=False,
                        cap_path=self.captions_path,
                        ssize=self.ssize
                    )
                ).reshape((self.ssize,8))
            )
        
        return x_ref, x_chg

    def transform_image(self, img):
        img = img/255.
        return self.preprocess_img(img)
    
    def inv_transform_image(self, img):
        if type(img) != np.ndarray:
            img = img.cpu().detach().numpy()
        return (
            (
                (
                    img.transpose(1,2,0) * self.std_vec
                ) + self.mean_vec
            ) * 255. 
        ).astype(np.uint8)
        

    def get_item_baseline(self, c_dt):
        ref_bb_imgs, chg_bb_imgs = [], []
        for i in range(self.baseline_size):
            ref_bb_img, chg_bb_img = get_same_fov_bb_image_pair_from_c_dt(
                c_dt=c_dt,
                cap_path=self.captions_path,
                i=i
            )
            ref_bb_imgs.append(self.transform_image(ref_bb_img))
            chg_bb_imgs.append(self.transform_image(chg_bb_img))
        
        if self.baseline_size == 1:
            return ref_bb_imgs[0], chg_bb_imgs[0]

        return torch.stack(ref_bb_imgs), torch.stack(chg_bb_imgs)

    def flip(self, p):
        return (random.random() < p)

    def turn_to_geometry(self, data):
        pos = data[:, :3].type(torch.float32)
        feats = data[:, 3:].type(torch.float32)

        ones = torch.ones((data.shape[0], 1))
        return Data(pos = pos, x = torch.concat((ones, feats), dim=1))

    def get_masked_meta(self, c_dt, ref):
        if 'full' in self.model:
            x = self.data_manip.get_single_classifier_input(
                chg=c_dt['chg'],
                ssize=self.ssize,
                is_ref=ref
            )
        else:
            x = np.load(
                file=self.data_manip.get_single_npy_path(
                    c_dt=c_dt,
                    ref=ref,
                    cap_path=self.captions_path,
                    ssize=self.ssize
                )
            )
        
        m = self.data_manip.get_classifier_mask(
            c_path=self.captions_path,
            c_dt=c_dt,
            ssize=self.ssize,
            is_ref=ref
        )

        return x, m

    def get_masked_x(self, c_dt, ref):
        x, m = self.get_masked_meta(c_dt, ref)

        return torch.tensor(
            np.concatenate((x, m.reshape(-1,1)), axis=1)
        )


class CapCollate:
    def __init__(self, encoder, model):
        self.encoder = encoder
        self.model = model

    def __call__(self, batch):
        if self.encoder == 'kpconv' or 'ensemble' in self.model:
            if 'ensemble' not in self.model:
                scans_bef = Batch.from_data_list([item[0] for item in batch])
                scans_aft = Batch.from_data_list([item[1] for item in batch])
            else:
                scans_bef_kp = Batch.from_data_list([item[0][1] for item in batch])
                scans_aft_kp = Batch.from_data_list([item[1][1] for item in batch])
        if self.encoder == 'minkowski' or 'ensemble' in self.model:
            if 'ensemble' not in self.model:
                scans_bef = [item[0] for item in batch]
                scans_aft = [item[1] for item in batch]
            else:
                scans_bef_mink = [item[0][0] for item in batch]
                scans_aft_mink = [item[1][0] for item in batch]
        if self.encoder != 'kpconv' and self.encoder != 'minkowski':
            scans_bef = torch.stack([item[0] for item in batch])
            scans_aft = torch.stack([item[1] for item in batch])
        if 'ensemble' in self.model:
            scans_bef = (scans_bef_mink, scans_bef_kp)
            scans_aft = (scans_aft_mink, scans_aft_kp)
        
        oh_actions = torch.tensor(np.stack([item[2] for item in batch]))
        caps = [item[3] for item in batch]

        return scans_bef, scans_aft, oh_actions, caps


def get_loader(
    data_manip_path,
    captions_path,
    ssize,
    batch_size,
    encoder,
    model,
    quantization_size,
    flip_prob,
    baseline_size=1,
    num_workers=8,
    shuffle=True,
    pin_memory=False,
    val_split=.2
):
    data_manip = DataManipulator(data_manip_path)
    train_refs, val_refs = data_manip.get_caption_split(
        c_path=captions_path,
        filter_ssize=ssize,
        filter_baseline=baseline_size,
        split=val_split,
        split_path=captions_path
    )

    train_dt = TemplateDataset(
        data_manip_path=data_manip_path,
        captions_path=captions_path,
        ssize=ssize,
        encoder=encoder,
        quantization_size=quantization_size,
        model=model,
        flip_prob=flip_prob,
        baseline_size=baseline_size,
        ref_split=train_refs
    )

    val_dt = TemplateDataset(
        data_manip_path=data_manip_path,
        captions_path=captions_path,
        ssize=ssize,
        encoder=encoder,
        quantization_size=quantization_size,
        model=model,
        flip_prob=flip_prob,
        baseline_size=baseline_size,
        ref_split=val_refs
    )

    train_loader = DataLoader(
        dataset=train_dt,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapCollate(encoder=encoder, model=model)
    )

    val_loader = DataLoader(
        dataset=val_dt,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapCollate(encoder=encoder, model=model)
    )

    return train_loader, val_loader, train_dt, val_dt
        