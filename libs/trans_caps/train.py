import os
import warnings
import torch.backends.cudnn as cudnn

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

# torch.autograd.set_detect_anomaly(True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0

    for epoch in range(options.epochs):
        start_time = time.time()
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        model.train()

        train_loss = 0
        targets, predictions = [], []
        # increments the margin for spread loss
        if options.loss_type == 'spread' and (
                epoch + 1) % options.n_eps_for_m == 0 and epoch != 0 and capsule_loss.margin != options.m_max:
            capsule_loss.margin += options.m_delta
            capsule_loss.margin = min(capsule_loss.margin, options.m_max)
            log_string(' *------- Margin increased to {0:.1f}'.format(capsule_loss.margin))

        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            global_step += 1
            target_ohe = F.one_hot(target, options.num_classes)

            y_pred, output = model(data, target_ohe)
            loss = capsule_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            targets += [target_ohe]
            predictions += [y_pred]
            train_loss += loss.item()

            # if (batch_id + 1) % options.disp_freq == 0:
            if (batch_id + 1) % train_freq == 0:
                train_loss /= options.disp_freq
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))
                log_string("epoch: {0}, step: {1}, time: {2:.2f}, train_loss: {3:.4f} train_accuracy: {4:.02%}"
                           .format(epoch + 1, batch_id + 1, time.time() - start_time, train_loss, train_acc))
                info = {'loss': train_loss,
                        'accuracy': train_acc}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                train_loss = 0
                targets, predictions = [], []
                start_time = time.time()

            if (batch_id + 1) % val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(best_loss=best_loss,
                                               best_acc=best_acc,
                                               global_step=global_step)
                model.train()
        lr_decay.step()


@torch.no_grad()
def evaluate(**kwargs):
    eval_start = time.time()
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    model.eval()
    test_loss = 0
    targets, predictions = [], []

    for batch_id, (data, target) in enumerate(valid_loader):
        data, target = data.cuda(), target.cuda()
        target_ohe = F.one_hot(target, options.num_classes)
        y_pred, output = model(data)
        loss = capsule_loss(output, target)

        targets += [target_ohe]
        predictions += [y_pred]
        test_loss += loss

    test_loss /= (batch_id + 1)
    test_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))

    # check for improvement
    loss_str, acc_str = '', ''
    if test_loss <= best_loss:
        loss_str, best_loss = '(improved)', test_loss
    if test_acc >= best_acc:
        acc_str, best_acc = '(improved)', test_acc

    eval_end = time.time()

    # display
    log_string('Validation time: {0:.4f}'.format(eval_end - eval_start))
    log_string("validation_loss: {0:.2f} {1}, validation_accuracy: {2:.02%} {3}"
               .format(test_loss, loss_str, test_acc, acc_str))

    # write to TensorBoard
    info = {'loss': test_loss,
            'accuracy': test_acc}
    for tag, value in info.items():
        test_logger.scalar_summary(tag, value, global_step)

    # save checkpoint model
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
    torch.save({
        'global_step': global_step,
        'loss': test_loss,
        'acc': test_acc,
        'save_dir': model_dir,
        'state_dict': state_dict},
        save_path)
    log_string('Model saved at: {}'.format(save_path))
    log_string('--' * 30)
    return best_loss, best_acc


if __name__ == '__main__':
    cudnn.deterministic = True
    cudnn.benchmark = False

    # ensure reproducibility
    torch.manual_seed(options.random_seed)
    kwargs = {}
    if torch.cuda.is_available():
        torch.cuda.manual_seed(options.random_seed)
        kwargs = {'num_workers': 4, 'pin_memory': False}

    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    save_dir += '_' + options.routing + '_' + options.backbone + '_' + options.data_name
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/capsule_model.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    model = CapsuleNet(options)
    log_string('Model Generated.')
    log_string("Number of parameters: {}".format(sum(param.numel() for param in model.parameters())))

    ##################################
    # Use cuda
    ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    ##################################
    # Loss and Optimizer
    ##################################
    capsule_loss = select_loss(options)

    if options.routing == 'TR':
        # optimizer = Adam(model.parameters(), lr=options.lr, betas=(options.beta1, 0.999),
        #                  weight_decay=options.weight_decay, amsgrad=True)
        optimizer = SGD(model.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    if options.data_name == "cifar10" or options.data_name == "cifar100":
        lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    elif options.data_name == "svhn":
        lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif options.data_name == "smallnorb":
        lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif options.data_name == 'four_classes_mass_calc_pathology':
        lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


    ##################################
    # Load dataset
    ##################################
    # train_loader, valid_loader = \
    #     get_train_valid_loader(options.data_dir,
    #                            options.data_name,
    #                            options.batch_size,
    #                            options.random_seed,
    #                            options.exp,
    #                            options.valid_size, **kwargs)
    train_loader, valid_loader, test_loader, _ = \
        cbis_ddsm_get_dataloaders(options.data_name,
                                  options.batch_size)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('MODEL NAME: {}'.format(save_dir))
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_loader.dataset), len(valid_loader.dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))
    train_freq = int(np.ceil(len(train_loader.dataset))/options.batch_size)
    val_freq = int(np.ceil(len(train_loader.dataset))/options.batch_size)
    train()
