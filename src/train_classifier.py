import time
import random
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
from datetime import datetime
from misc_utils import count_parameters, get_available_device, get_run_id, write_to_file, set_mode

from classifier.dataset import get_loader
from classifier_metrics import metrics_generator
from classifier.model import Classifier

from MinkowskiEngine.utils import sparse_collate
from MinkowskiEngine import TensorField

from torch.utils.tensorboard import SummaryWriter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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


def checkpoint_state_dict(model, path, model_name, file):
    save_model_path = f'{path}_{model_name}'
    torch.save(model.state_dict(), save_model_path)
    write_to_file(file, f'Saved {model_name} at {save_model_path}.\n')


def classifier_step(batch, classifier_model, \
    criterion_cat, args, device):
    scans_bef, scans_aft, cats, _ = batch

    scans_bef = process_input(
        x=scans_bef,
        device=device,
        encoder=args.encoder,
        model=args.model
    )
    scans_aft = process_input(
        x=scans_aft,
        device=device,
        encoder=args.encoder,
        model=args.model
    )

    cats = torch.argmax(cats, dim=1).to(device)

    if 'ensemble' not in args.model:
        outputs = classifier_model(scans_bef, scans_aft)
    else:
        outputs = classifier_model(
            scans_bef=scans_bef[0],
            scans_aft=scans_aft[0],
            scans_bef_2=scans_bef[1], 
            scans_aft_2=scans_aft[1]
        )
    loss = criterion_cat(outputs, cats)

    acc = torch.sum(
        torch.argmax(outputs, dim=1) == cats
    ).item()
    bsize = outputs.shape[0]

    return loss, acc, bsize


def epoch_metric_log(train, e_rloss, e_steps_m, e_racc, e_tacc, summary_step, f, writer, runtime):
    prefix = 'VAL'
    if train:
        prefix = 'TRAIN'

    e_rloss /= e_steps_m
    e_racc /= e_tacc

    writer.add_scalar(f'Loss/{prefix}_loss_every_epoch', e_rloss, summary_step)
    writer.add_scalar(f'Acc/{prefix}_acc_every_epoch', e_racc, summary_step)

    desc_str = f'EPOCH {summary_step} --> {prefix} loss: {e_rloss} - {prefix} acc: {e_racc} - elapsed: {runtime:.2f} seconds'
    write_to_file(f, f'{desc_str}\n')
    tqdm.write(desc_str)
    return e_rloss.item(), e_racc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the classifier network.')
    parser.add_argument('-d', '--data-path', dest='path', type=str, default=None, help='3Rscan data path')
    parser.add_argument('-cd', '--captions-path', dest='cap_path', type=str, default='../indoor-scene-activity-recognition/', help='manual captions data path')
    parser.add_argument('-lr', '--learning-rate', dest='l_rate', type=float, default=.001, help='Learning rate')
    parser.add_argument('-b', '--batch-size', dest='b_size', type=int, default=16, help='Batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=100, help='Epochs')
    parser.add_argument('-m', '--metrics', dest='metrics', type=int, default=5, help='Save metrics every ? steps')
    parser.add_argument('-lr_dec', '--learning-rate-decay', dest='lr_dec', type=float, default=.5, help='Learning rate decay')
    parser.add_argument('-s', '--sample-size', dest='ssize', type=int, default=20000, help='input pointcloud dimension')
    parser.add_argument('-ed', '--embed-dim-detector', dest='embed_dim_detector', type=int, default=128, help='embed dimension of change detector')
    parser.add_argument('-q', '--quantization-size', dest='quantization_size', type=float, default=0.1, help='quantization size for minkowski engine sparse collate')
    parser.add_argument('-en', '--encoder', dest='encoder', type=str, default='kpconv', choices=['kpconv', 'pointnet2', 'minkowski', 'resnet18', 'resnet34', 'resnet50', 'resnext'], help='encoder network choices')
    parser.add_argument('-ms', '--manual-seed', dest='manual_seed', type=int, default=0, help='torch manual seed')
    parser.add_argument('-c', '--cpu', dest='cpu', type=bool, default=False, help='run training on cpu')
    # if minkowski: CUDA_VISIBLE_DEVICES=one,device -g 0
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=-1, help='manually select gpu index')
    parser.add_argument('-mo', '--model', dest='model', type=str, default='m_sub', choices=['full_features', 'full_attention', 'm_features', 'm_attention', 'm_ensemble', 'm_sub', 'nm_features', 'nm_attention', 'baseline', 'multi_baseline'], help='choose model type')
    parser.add_argument('-si', '--siamese', dest='siamese', type=int, default=1, help='number of baseline samples')
    parser.add_argument('-fp', '--flip-prob', dest='flip_prob', type=float, default=.0, help='chance of flipping the data samples')
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    
    run_name = 'classifier'
    train_loader, val_loader, train_dt, _ = get_loader(
        data_manip_path=args.path,
        captions_path=args.cap_path,
        ssize=args.ssize,
        batch_size=args.b_size,
        encoder=args.encoder,
        model=args.model,
        quantization_size=args.quantization_size,
        flip_prob=args.flip_prob,
        baseline_size=args.siamese
    )

    train_id = get_run_id()
    checkpoint_path = f'/local/crv/danich/model_checkpoints/{run_name}/{run_name}-{train_id}'
    os.makedirs(checkpoint_path, exist_ok=True)

    with open(os.path.join(checkpoint_path, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    tqdm.write(f'save path name: {train_id}')

    f = open(os.path.join(checkpoint_path, 'log.txt'), 'w')
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") 
    write_to_file(f, f'run date and time: {dt_string}\n\n{str(args)}\n\n')

    device = get_available_device(args.cpu, args.gpu)
    tqdm.write(f'Running on gpu:{device}')

    input_feats = 9
    if args.model in ['nm_attention', 'nm_features']:
        input_feats = 8

    classifier_model = Classifier(
        encoder=args.encoder,
        type=args.model,
        input_feats=input_feats,
        embed_dim_detector=args.embed_dim_detector,
        num_classes=train_dt.num_classes,
        siamese=args.siamese
    ).float().to(device)

    params_num = count_parameters(classifier_model)
    params_str = f"Model created. Total parameters number: {params_num:,}\n"
    tqdm.write(params_str)
    write_to_file(f, params_str)

    criterion_cat = nn.CrossEntropyLoss(
        weight=torch.tensor(
            data=train_dt.class_weights,
            dtype=torch.float
        )
    ).to(device)

    optimizer = optim.Adam(
        params=list(classifier_model.parameters()),
        lr=args.l_rate    
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=args.lr_dec,
        patience=5,
        min_lr=1e-5
    )
    curr_lr = get_lr(optimizer)

    step = 0
    write_to_file(f, 'Starting Training:\n')
    writer = SummaryWriter(log_dir=checkpoint_path)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    rloss = 0.
    steps_m = racc = tacc = 0
    training_start = time.time()
    for epoch in range(args.epochs):
        e_rloss = 0.
        e_steps_m = e_racc = e_tacc = 0
        set_mode(mode='train', models=[classifier_model])
        start = time.time()
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            loss, acc, bsize = classifier_step(
                batch=batch,
                classifier_model=classifier_model,
                criterion_cat=criterion_cat,
                args=args,
                device=device
            )

            step += 1
            writer.add_scalar('Loss/TRAIN_loss_every_step', loss.item(), step)

            rloss += loss
            e_rloss += loss
            racc += acc
            e_racc += acc

            steps_m += 1
            e_steps_m += 1
            tacc += bsize
            e_tacc += bsize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.metrics == 0:
                rloss /= steps_m
                racc /= tacc

                writer.add_scalar(f'Loss/TRAIN_loss_every_{args.metrics}_step', rloss, step)
                writer.add_scalar(f'Acc/TRAIN_acc_every_{args.metrics}_step', racc, step)

                rloss = 0.
                steps_m = racc = tacc = 0
    
        end = time.time()
        e_rloss_box, e_racc_box = epoch_metric_log(
            train=True,
            e_rloss=e_rloss,
            e_steps_m=e_steps_m,
            e_racc=e_racc,
            e_tacc=e_tacc,
            summary_step=epoch+1,
            f=f,
            writer=writer,
            runtime=(end-start)
        )
        train_losses.append(e_rloss_box)
        train_accs.append(e_racc_box)

        e_rloss = 0.
        e_steps_m = e_racc = e_tacc = 0
        set_mode('eval', [classifier_model])
        start = time.time()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
                loss, acc, bsize = classifier_step(
                    batch=batch,
                    classifier_model=classifier_model,
                    criterion_cat=criterion_cat,
                    args=args,
                    device=device
                )

                e_rloss += loss
                e_racc += acc

                e_steps_m += 1
                e_tacc += bsize
        end = time.time()

        e_rloss_box, e_racc_box = epoch_metric_log(
            train=False,
            e_rloss=e_rloss,
            e_steps_m=e_steps_m,
            e_racc=e_racc,
            e_tacc=e_tacc,
            summary_step=epoch+1,
            f=f,
            writer=writer,
            runtime=(end-start)
        )
        val_losses.append(e_rloss_box)
        val_accs.append(e_racc_box)
    
        lr_scheduler.step(e_racc_box)
        if get_lr(optimizer) != curr_lr:
            curr_lr = get_lr(optimizer)
            lr_str = f'Learning rate changed to {curr_lr}\n'
            write_to_file(f, lr_str)
            tqdm.write(lr_str)

    training_end = time.time()
    base_save_path = f'{checkpoint_path}/checkpoint_epoch_final'
    checkpoint_state_dict(
        model=classifier_model,
        path = base_save_path,
        model_name='classifier_model',
        file=f
    )
    finished_string = f'Finished Training! Total elapsed time: {(training_end-training_start):.2f} seconds.\n\n'
    tqdm.write(finished_string)
    write_to_file(f, finished_string)

    write_to_file(f, f'Train Losses: {str(train_losses)}\n\n')
    write_to_file(f, f'Train Accuracies: {str(train_accs)}\n\n')
    write_to_file(f, f'Validation Losses: {str(val_losses)}\n\n')
    write_to_file(f, f'Validation Accuracies: {str(val_accs)}\n\n')

    generating_metrics = 'Generating model metrics...\n'
    tqdm.write(generating_metrics)
    write_to_file(f, generating_metrics)
    metrics_generator(
        checkpoint_path=checkpoint_path,
        classifier_model=classifier_model,
        train_loader=train_loader,
        val_loader=val_loader,
        dt=train_dt,
        loaded_args=vars(args),
        device=device
    )

    generated_metrics = 'Generated model metrics.\n'
    tqdm.write(generated_metrics)
    write_to_file(f, generated_metrics)
