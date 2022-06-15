import os
import argparse
import matplotlib.pyplot as plt
import pickle


def get_accs_labs(runs):
    train_accs = []
    val_accs = []
    labs = []

    for run in runs:
        run_name = 'classifier'

        checkpoint_path = f'/local/crv/danich/model_checkpoints/{run_name}/{run}/'
        with open(os.path.join(checkpoint_path, 'args.pkl'), 'rb') as f:
            loaded_args = pickle.load(f)
            l = f'{loaded_args["encoder"]}_{loaded_args["model"]}'
            if loaded_args["model"] == 'multi_baseline':
                l += f'_{loaded_args["siamese"]}'
            labs.append(l)
        
        acc_lines = get_log_acc_lines(checkpoint_path)
        train_acc, val_acc = create_acc_ls(acc_lines[0]), create_acc_ls(acc_lines[1])
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    return train_accs, val_accs, labs


def create_acc_multiplot(labs, accs, train, base_path):
    prefix = 'validation'
    if train:
        prefix = 'train'

    plt.figure(figsize=(14,14))
    xs = [list(range(1,len(x)+1)) for x in accs]

    # labs = [l[:100] for l in labs]
    # accs = [a[:100] for a in accs]
    # xs = [x[:100] for x in xs]

    for l,t,x in zip(labs,accs,xs):
        plt.plot(x,t,label=l)

    plt.legend(loc=2, prop={'size': 24})
    plt.grid(axis='both')
    # plt.title(f'{prefix.capitalize()} Accuracy', fontsize=32)
    plt.xlabel('Epoch', fontsize=28, labelpad=20)
    plt.ylabel('Accuracy', fontsize=28, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig(
        os.path.join(base_path, f'{prefix}_acc.png'),
        bbox_inches='tight',
        dpi=300
    )


def get_log_acc_lines(checkpoint_path):
    with open(os.path.join(checkpoint_path, 'log.txt')) as f:
        lines = f.readlines()
    return [l for l in lines if 'Accuracies' in l]


def create_acc_ls(line):
    to_rep = ['Validation Accuracies: ', '\n', '[', ']']
    if 'Train' in line:
        to_rep[0] = 'Train Accuracies: '
    for tr in to_rep:
        line = line.replace(tr, '')
    
    return [float(x) for x in line.split(', ')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare runs of training of the classifier network.')
    parser.add_argument('-c','--ckpt', dest='ckpt', nargs='+', help='train runs directories', required=True)
    parser.add_argument('-l','--labs', default=[], dest='labs', nargs='+', help='legend names')
    parser.add_argument('-s', '--save-path', dest='save_path', type=str, default='./', help='choose where to save the graphs')
    args = parser.parse_args()

    runs = args.ckpt
    train_accs, val_accs, labs = get_accs_labs(runs)
    if len(args.labs) == len(labs):
        labs = args.labs

    os.makedirs(args.save_path, exist_ok=True)

    create_acc_multiplot(labs, train_accs, True, args.save_path)
    create_acc_multiplot(labs, val_accs, False, args.save_path)

    with open(os.path.join(args.save_path, 'metrics.txt'), 'w') as f:
        for l,t,v in zip(labs,train_accs,val_accs):
            f.write(f'{l}\n')
            f.write(f'\tTRAIN_FINAL//MAX:\t\t{t[-1]:.5f}\t\t{max(t):.5f}\n')
            f.write(f'\tVAL_FINAL//MAX:\t\t\t{v[-1]:.5f}\t\t{max(v):.5f}\n\n')