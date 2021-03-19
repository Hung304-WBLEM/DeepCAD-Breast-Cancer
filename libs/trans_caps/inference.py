import os
import warnings
import torch.backends.cudnn as cudnn
import glob

from data_loader import get_train_valid_loader, cbis_ddsm_get_dataloaders
from utils.loss_utils import select_loss
import torch.optim as optim
warnings.filterwarnings("ignore")
from datetime import datetime
from torch.optim import Adam, SGD
import numpy as np
from configs import options
import torch
import torch.nn.functional as F
from utils.eval_utils import compute_accuracy
from utils.logger_utils import Logger
import torch.nn as nn
import time
from capsule_model import CapsuleNet
from features_classification.eval_utils import eval_all

# torch.autograd.set_detect_anomaly(True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


@torch.no_grad()
def evaluate():
    model.eval()
    test_loss = 0
    targets, predictions, outputs = [], [], []

    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        target_ohe = F.one_hot(target, options.num_classes)
        y_pred, output = model(data)
        loss = capsule_loss(output, target)

        targets += [target_ohe]
        predictions += [y_pred]
        outputs += [output]
        test_loss += loss

    test_loss /= (batch_id + 1)
    test_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))

    return torch.cat(outputs), torch.cat(targets)


if __name__ == '__main__':
    
    model = CapsuleNet(options)

    ##################################
    # Use cuda
    ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    models_dir = options.load_model_path
    best_acc = None
    best_state_dict = None
    for model_path in glob.glob(os.path.join(models_dir, '*.ckpt')):
        checkpoint = torch.load(model_path)
        if best_acc is None or checkpoint['acc'] > best_acc:
            best_acc = checkpoint['acc']
            best_state_dict = checkpoint['state_dict']

    model.load_state_dict(best_state_dict)

    ##################################
    # Loss and Optimizer
    ##################################
    capsule_loss = select_loss(options)

    _, _, test_loader, classes = \
        cbis_ddsm_get_dataloaders(options.data_name,
                                  options.batch_size)

    outputs, targets = evaluate()

    outputs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()
    targets = torch.max(targets, 1).indices.detach().cpu().numpy()

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    eval_all(targets, outputs, classes, save_dir)
