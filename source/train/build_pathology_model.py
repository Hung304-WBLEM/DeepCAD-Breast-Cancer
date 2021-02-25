import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import glob
import time
import copy
import logging
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from test import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from dataprocessing.process_cbis_ddsm import get_info_lesion
from config.cfg_loader import proj_paths_json
from skimage import io, transform
from torch import nn
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from train import set_parameter_requires_grad
from datasets import Features_Pathology_Dataset
from PIL import Image
matplotlib.use('Agg')


class Pathology_Model(nn.Module):
    def __init__(self, model_name, freeze_type, input_vector_dim, use_pretrained=True):
        super(Pathology_Model, self).__init__()
        self.model_name = model_name
        if model_name == "resnet50":
            self.cnn = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(self.cnn, model_name, freeze_type)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            self.fc1 = nn.Linear(2048 + input_vector_dim, 512)
            self.fc2 = nn.Linear(512, 2)
        elif model_name == 'vgg16':
            self.cnn = models.vgg16_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(self.cnn, model_name, freeze_type)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-3])

            self.fc1 = nn.Linear(self.cnn.classifier[3].out_features + input_vector_dim, 512)
            self.fc2 = nn.Linear(512, 2)


    def forward(self, image, vector_data, training):
        x1 = self.cnn(image)
        x2 = vector_data

        if self.model_name == 'resnet50':
            x1 = x1.squeeze()
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=training)
        x = self.fc2(x)
        return x


def train_pathology_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        logging.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample['image']
                labels = sample['pathology']
                input_vectors = sample['feature_vector']
                input_vectors = input_vectors.type(torch.FloatTensor)

                inputs = inputs.to(device)
                labels = labels.to(device)
                input_vectors = input_vectors.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        if phase == 'val':
                            outputs = model(inputs, input_vectors, training=False)
                        elif phase == 'train':
                            outputs = model(inputs, input_vectors, training=True)
                            
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


@torch.no_grad()
def get_all_preds_pathology(model, loader, device):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([], dtype=torch.long)
    all_labels = all_labels.to(device)

    for sample in loader:
        inputs = sample['image']
        labels = sample['pathology']
        input_vectors = sample['feature_vector']
        input_vectors = input_vectors.type(torch.FloatTensor)

        inputs = inputs.to(device)
        labels = labels.to(device)
        input_vectors = input_vectors.to(device)

        all_labels = torch.cat((all_labels, labels), dim=0)

        preds = model(inputs, input_vectors, training=False)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds, all_labels
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="Name of the available datasets")
    parser.add_argument("-s", "--save_path",
                        help="Path to save the trained model")
    parser.add_argument(
        "-m", "--model_name", help="Select the backbone for training. Available backbones include: 'resnet', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Batch size for training")
    parser.add_argument(
        "-e", "--epochs", type=int, help="the number of epochs for training")
    parser.add_argument("-f", "--freeze_type", help="For Resnet50, freeze_type could be: 'none', 'all', 'last_fc', 'top1_conv_block', 'top2_conv_block', 'top3_conv_block'. For VGG16, freeze_type could be: 'none', 'all', 'last_fc', 'fc2', 'fc1', 'top1_conv_block', 'top2_conv_block'")

    args = parser.parse_args()

    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])
    if args.dataset in ['mass_pathology']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][args.dataset])
    elif args.dataset in ['calc_pathology']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][args.dataset])

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    # Models to choose from [resnet, resnet50, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.model_name

    # Number of classes in the dataset
    classes = ['MALIGNANT', 'BENIGN']
    num_classes = len(classes)

    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batch_size

    # Number of epochs to train
    num_epochs = args.epochs
    
    if args.dataset == 'mass_pathology':
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        model = Pathology_Model(model_name, args.freeze_type, input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats)
    elif args.dataset == 'calc_pathology':
        breast_density_cats = 4
        calc_type_cats = 14
        calc_dist_cats = 5
        model = Pathology_Model(model_name, args.freeze_type, input_vector_dim=breast_density_cats+calc_type_cats+calc_dist_cats)


    # print(model)

    input_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    if args.dataset == 'mass_pathology':
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif args.dataset == 'calc_pathology':
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
        


    dataloaders_dict = {x: DataLoader(pathology_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Select params to update
    params_to_update = model.parameters()
    print("Params to learn:")
    if args.freeze_type != 'none':
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_pathology_model(model,
                              dataloaders_dict,
                              criterion,
                              optimizer_ft,
                              num_epochs=num_epochs,
                              is_inception=False)

    # save best checkpoint 
    torch.save(model.state_dict(), os.path.join(save_path, 'ckpt.pth'))

    # plot train-val lost and tran-val accuracy
    fig = plt.figure()
    plt.plot(range(args.epochs), train_loss_hist, label='train loss')
    plt.plot(range(args.epochs), val_loss_hist, label='val loss')
    plt.xlabel('#epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

    fig = plt.figure()
    plt.plot(range(args.epochs), train_acc_hist, label='train accuracy')
    plt.plot(range(args.epochs), val_acc_hist, label='val accuracy')
    plt.xlabel('#epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc_plot.png'))
    plt.close()

    ################### Test Model #############################
    if args.dataset == 'mass_pathology':
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif args.dataset == 'calc_pathology':
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}

    test_dataloaders_dict = {'test': torch.utils.data.DataLoader(test_image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)}

    model.eval()
    
    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels = get_all_preds_pathology(model, prediction_loader, device)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(labels.cpu(), classes=[*range(num_classes)])
        print(binarized_labels.shape, softmaxs.shape)

    ######### PLOT ##########
    matplotlib.use('Agg')

    # plot confusion matrix
    cm = confusion_matrix(labels.cpu(), preds.argmax(dim=1).cpu())

    plt.figure(figsize=(5, 5))
    plot_confusion_matrix(cm, classes)
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(5, 5))
    norm_cm = plot_confusion_matrix(cm, classes, normalize=True)
    plt.savefig(os.path.join(save_path, 'norm_confusion_matrix.png'))
    plt.close()

    # Plot PR curve
    prs, recs, aps = [], [], []
    if args.dataset in ['mass_pathology', 'calc_pathology']:
        pr, rec, ap = plot_precision_recall_curve(binarized_labels, softmaxs[:, 0].cpu(), class_name='Pathology')
        prs.append(pr)
        recs.append(rec)
        aps.append(ap)
    else:
        for i in range(num_classes):
            pr, rec, ap = plot_precision_recall_curve(binarized_labels[:, i], softmaxs[:, i].cpu(), class_name=classes[i])
            prs.append(pr)
            recs.append(rec)
            aps.append(ap)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'pr_curve.png'))
    plt.close()

    # Plot ROC curve
    fprs, tprs, aucs = [], [], []
    if args.dataset in ['mass_pathology', 'calc_pathology']:
        fpr, tpr, auc = plot_precision_recall_curve(binarized_labels, softmaxs[:, 0].cpu(), class_name='Pathology')
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc)
    else:
        for i in range(num_classes):
            fpr, tpr, auc = plot_roc_curve(binarized_labels[:, i], softmaxs[:, i].cpu(), class_name=classes[i])
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="best")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()

    # Save Data for plotting in the future
    plot_data = {'confusion_matrix': cm, 'norm_confusion_matrix': norm_cm,
                 'precision_list': prs, 'recall_list': recs, 'ap_list': aps,
                 'fpr_list': fprs, 'tpr_list': tprs, 'auc_list': aucs,
                 'classes': classes}

    with open(os.path.join(save_path, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(plot_data, f)
