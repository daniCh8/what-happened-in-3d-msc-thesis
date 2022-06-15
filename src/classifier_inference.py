import random
import torch
import argparse
import numpy as np
import pickle
import os

from misc_utils import get_available_device, set_mode

from classifier.dataset import get_loader, CapCollate
from classifier.model import Classifier
from classifier_metrics import load_sd_for_, eval_step

from MinkowskiEngine.utils import sparse_collate
from MinkowskiEngine import TensorField


def collate_minkowski_(batch, device):
    # collate must be here to not re-initialized CUDA in subprocesses
    coords = []
    feats = []
    for item in batch:
        c, f = item
        coords.append(c)
        feats.append(f)
    
    c, f = sparse_collate(coords=coords, feats=feats)
    c = c.to(device)
    f = f.to(device).float()

    scans = TensorField(coordinates=c, features=f, device=device)
    return scans


def process_input_helper(x, device, encoder):
    if encoder == 'minkowski':
        x = collate_minkowski_(x, device)
    else:
        x = x.to(device)
        if  encoder in ['pointnet2', 'resnet18', 'resnet34', 'resnet50', 'resnext']:
            x = x.float()
    
    return x


def process_input(x, device, encoder, model):
    if 'ensemble' not in model:
        return process_input_helper(
            x=x,
            device=device,
            encoder=encoder
        )
    
    return process_input_helper(x[0], device, 'minkowski'), process_input_helper(x[1], device, 'kpconv')


def inference(classifier_model, train_loader, t_dt, v_dt, loaded_args, device, chg, action_key):
    models = [classifier_model]
    set_mode('eval', models)

    with torch.no_grad():
        if chg is not None and action_key is not None:
            coll = CapCollate(loaded_args['encoder'], loaded_args['model'])
            idx = t_dt.get_action_index(chg, action_key)
            if idx != -1:
                a, b, c, d = t_dt.__getitem__(idx)
            else:
                idx = v_dt.get_action_index(chg, action_key)
                assert idx != -1, "This sample is not in the dataset."
                a, b, c, d = v_dt.__getitem__(idx)

            batch = coll.__call__([(a,b,c,d)])
        else:
            idx = np.random.randint(len(train_loader))
            for _ in range(idx):
                batch = next(iter(train_loader))
            
        _, outs, cats, _, _ = eval_step(
            classifier_model=classifier_model,
            batch=batch,
            loaded_args=loaded_args,
            device=device
        )
        outs = outs.detach().cpu()
        cats = cats.detach().cpu()

        for pred,ground,cap in zip(train_dt.onehot_encoder.inverse_transform(outs), \
            train_dt.onehot_encoder.inverse_transform(cats),batch[3]):
            predicted,groundtruth = pred[0],ground[0]
            print(f"{cap['dir']}/{cap['id']}/{cap['action']}:")
            print(f"\tPREDICTED: {pred}; GROUNDTRUTH: {ground}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on the classifier network.')
    parser.add_argument('-ck', '--ckpt', dest='ckpt', type=str, default='classifier-2022-06-08--10:13:36.239172', help='checkpoint id path')
    parser.add_argument('-c', '--cpu', dest='cpu', type=bool, default=False, help='run training on cpu')
    parser.add_argument('-chg', '--changed', dest='changed', type=str, default=None, help='inference case change directory')
    parser.add_argument('-k', '--key', dest='action_key', type=str, default=None, help='inference case action key')
    # if minkowski: CUDA_VISIBLE_DEVICES=one,device -g 0
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=-1, help='manually select gpu index')
    args = parser.parse_args()

    checkpoint_path = args.ckpt
    with open(os.path.join(checkpoint_path, 'args.pkl'), 'rb') as f:
        loaded_args = pickle.load(f)
    print(loaded_args)
    
    if 'manual_seed' not in loaded_args:
        loaded_args['manual_seed'] = 0
    torch.manual_seed(loaded_args['manual_seed'])
    random.seed(loaded_args['manual_seed'])
    np.random.seed(loaded_args['manual_seed'])
    
    if 'quantization_size' not in loaded_args:
        loaded_args['quantization_size'] = None
    
    if 'siamese' not in loaded_args:
        loaded_args['siamese'] = 1
    
    if 'flip_prob' not in loaded_args:
        loaded_args['flip_prob'] = 0

    train_loader, val_loader, train_dt, val_dt = get_loader(
        data_manip_path=loaded_args['path'],
        captions_path=loaded_args['cap_path'],
        ssize=loaded_args['ssize'],
        batch_size=loaded_args['b_size'],
        encoder=loaded_args['encoder'],
        model=loaded_args['model'],
        quantization_size=loaded_args['quantization_size'],
        flip_prob=loaded_args['flip_prob'],
        baseline_size=loaded_args['siamese']
    )

    device = get_available_device(args.cpu, args.gpu)
    print(f'Running on gpu:{device}')

    input_feats = 9
    if loaded_args['model'] in ['nm_attention', 'nm_features']:
        input_feats = 8

    classifier_model = Classifier(
        encoder=loaded_args['encoder'],
        type=loaded_args['model'],
        input_feats=input_feats,
        embed_dim_detector=loaded_args['embed_dim_detector'],
        num_classes=train_dt.num_classes,
        siamese=loaded_args['siamese']
    ).float().to(device)
    load_sd_for_(
        model=classifier_model,
        name='checkpoint_epoch_final_classifier_model',
        base_path=checkpoint_path,
        device=device
    )
    
    is_done = inference(
        classifier_model=classifier_model,
        train_loader=train_loader,
        t_dt=train_dt,
        v_dt=val_dt,
        loaded_args=loaded_args,
        device=device,
        chg=args.changed,
        action_key=args.action_key
    )
