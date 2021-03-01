import os
import glob
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--load_model_root", required=True,
                        help="Path to the saved model directory")
    args = parser.parse_args()
    
    load_model_root = args.load_model_root
    best_ckpt = None
    best_acc = 0
    for ckpt in glob.glob(os.path.join(load_model_root, 'models', '*.ckpt')):
        checkpoint = torch.load(ckpt)


        if best_ckpt is None or best_acc < checkpoint['acc']:
            best_ckpt = ckpt
            best_acc = checkpoint['acc']


    print(best_ckpt, best_acc)
    os.symlink(best_ckpt, os.path.join(load_model_root, 'models', 'best.ckpt'))
