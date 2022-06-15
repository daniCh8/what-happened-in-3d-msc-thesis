import random
import torch
import argparse
import numpy as np
import pickle
import os
from tqdm import tqdm

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import cm
from MulticoreTSNE import MulticoreTSNE as TSNE

from misc_utils import get_available_device, set_mode, write_to_file

from classifier.dataset import get_loader
from classifier.model import Classifier

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


def load_sd_for_(model, name, base_path, device):
    model_sd = torch.load(
        f=os.path.join(base_path, name),
        map_location=device
    )
    model.load_state_dict(model_sd)


def eval_step(classifier_model, batch, loaded_args, device):
    scans_bef, scans_aft, cats, _ = batch
    scans_bef = process_input(
        x=scans_bef,
        device=device,
        encoder=loaded_args['encoder'],
        model=loaded_args['model']
    )
    scans_aft = process_input(
        x=scans_aft,
        device=device,
        encoder=loaded_args['encoder'],
        model=loaded_args['model']
    )
    
    cats = cats.to(device)

    if 'ensemble' not in loaded_args['model']:
        hidden_x, outputs = classifier_model(scans_bef, scans_aft, pre_head=True)
    else:
        hidden_x, outputs = classifier_model(
            scans_bef=scans_bef[0],
            scans_aft=scans_aft[0],
            pre_head=True,
            scans_bef_2=scans_bef[1], 
            scans_aft_2=scans_aft[1]
        )

    acc = torch.sum(
        torch.argmax(outputs, dim=1) == \
            torch.argmax(cats, dim=1)
    ).item()
    bsize = outputs.shape[0]

    return hidden_x, outputs, cats, acc, bsize


def create_confusion_figure(true, pred, classes, file):
    _, ax = plt.subplots(figsize=(22, 15))

    ConfusionMatrixDisplay.from_predictions(
        y_true = true.ravel(),
        y_pred = pred.ravel(),
        labels = classes,
        display_labels = classes,
        ax = ax
    )

    plt.xlabel('Predicted Label', fontsize=28, labelpad=20)
    plt.ylabel('True Label', fontsize=28, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(file, bbox_inches='tight', dpi=600)


def create_tsne_plot(hidden_x, y, dt, file):
    cmap = cm.get_cmap('tab20')
    _, ax = plt.subplots(figsize=(8,8))

    emb_hx = TSNE(n_jobs=4).fit_transform(
        hidden_x.double().numpy()
    )
    y = torch.argmax(y, dim=1).numpy()

    for lab in range(dt.num_classes):
        indices = y == lab
        ax.scatter(
            x=emb_hx[indices,0],
            y=emb_hx[indices,1], 
            c=np.array(cmap(lab)).reshape(1,4), 
            label=lab,
            alpha=0.5
        )
    ax.legend(
        labels=dt.onehot_encoder.categories_[0],
        fontsize='medium',
        markerscale=2
    )
    plt.savefig(file, bbox_inches='tight', dpi=600)


def eval_epoch(loader, dt, classifier_model, training, loaded_args, checkpoint_path, log_f, device):
    prefix = 'validation'
    if training:
        prefix = 'training'
    print(f'evaluating {prefix} split...')

    racc = ratt = ri = 0
    x = torch.empty(size=(len(loader.dataset),dt.num_classes))
    y = torch.empty(size=(len(loader.dataset),dt.num_classes))

    hidden_x = torch.empty(size=(len(loader.dataset),classifier_model.classifier_head.pre_head_dim))
    for idx, batch in tqdm(enumerate(loader), total=len(loader), leave=False):
        hid_x, outs, cats, acc, bsize = eval_step(
            classifier_model=classifier_model,
            batch=batch,
            loaded_args=loaded_args,
            device=device
        )
        for hx, to, tc in zip(hid_x, outs, cats):
            hidden_x[ri] = hx
            x[ri] = to
            y[ri] = tc
            ri += 1
        racc += acc
        ratt += bsize
    
    racc = (racc/ratt)*100

    tsne_f = os.path.join(checkpoint_path, f'{prefix}_tsne.png')
    create_tsne_plot(
        hidden_x=hidden_x,
        y=y,
        dt=dt,
        file=tsne_f
    )
    write_to_file(
        f=log_f,
        message=f'saved {prefix} tsne plot at {tsne_f}\n'
    )

    conf_mat_f = os.path.join(checkpoint_path, f'{prefix}_confusion_matrix.png')
    tr, pr = dt.onehot_encoder.inverse_transform(y.cpu()).ravel(), dt.onehot_encoder.inverse_transform(x.cpu()).ravel()
    cl = dt.onehot_encoder.categories_[0]

    create_confusion_figure(
        true=tr,
        pred=pr,
        classes=cl,
        file=conf_mat_f
    )
    write_to_file(
        f=log_f,
        message=f'saved {prefix} confusion matrix at {conf_mat_f}\n'
    )

    write_to_file(
        f=log_f,
        message=classification_report(
            y_true=tr,
            y_pred=pr,
            target_names=cl,
            zero_division=1
        )
    )

    write_to_file(
        f=log_f,
        message=f'\n\n\n'
    )


def metrics_generator(checkpoint_path, classifier_model, train_loader, val_loader, dt, loaded_args, device):
    log_f = open(os.path.join(checkpoint_path, 'metrics_log.txt'), 'w')
    models = [classifier_model]
    set_mode('eval', models)

    with torch.no_grad():
        eval_epoch(
            loader=train_loader,
            dt=dt,
            classifier_model=classifier_model,
            training=True,
            loaded_args=loaded_args,
            checkpoint_path=checkpoint_path,
            log_f=log_f,
            device=device
        )

        eval_epoch(
            loader=val_loader,
            dt=dt,
            classifier_model=classifier_model,
            training=False,
            loaded_args=loaded_args,
            checkpoint_path=checkpoint_path,
            log_f=log_f,
            device=device
        )

    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on the classifier network.')
    parser.add_argument('-ck', '--ckpt', dest='ckpt', type=str, default='classifier-2022-05-17--12:59:21.956113', help='checkpoint id path')
    parser.add_argument('-c', '--cpu', dest='cpu', type=bool, default=False, help='run training on cpu')
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
    
    metrics_generator(
        checkpoint_path=checkpoint_path,
        classifier_model=classifier_model,
        train_loader=train_loader,
        val_loader=val_loader,
        dt=train_dt,
        loaded_args=loaded_args,
        device=device
    )
