from config.cfg_loader import proj_paths_json
from torchvision import datasets, models, transforms
from test import get_all_preds, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from utils.fileio import json
from datasets import Mass_Shape_Dataset, Mass_Margins_Dataset, Calc_Type_Dataset, Calc_Dist_Dataset, Pathology_Dataset, Breast_Density_Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import matplotlib
import logging
import pickle
matplotlib.use('Agg')


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                        outputs = model(inputs)
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


def set_parameter_requires_grad(model, model_name, freeze_type):
    '''
    Parameters:
    model_name - can be 'vgg16' or 'resnet50'
    freeze_type - can be 'none', 'all'
    '''
    if model_name == 'resnet50':
        if freeze_type != 'none':
            last_frozen_idx = {'all': 160, 'last_fc': 158, 'top1_conv_block': 149, 'top2_conv_block': 140, 'top3_conv_block': 128}
            for idx, (name, param) in enumerate(model.named_parameters()):
                # print(idx, name)
                if idx <= last_frozen_idx[freeze_type]:
                    param.requires_grad = False
    elif model_name == 'vgg16':
        if freeze_type != 'none':
            last_frozen_idx = {'all': 57, 'last_fc': 55, 'fc2': 53, 'fc1': 51, 'top1_conv_block': 45, 'top2_conv_block': 39}
            for idx, (name, param) in enumerate(model.named_parameters()):
                # print(idx, name)
                if idx <= last_frozen_idx[freeze_type]:
                    param.requires_grad = False


def initialize_model(model_name, num_classes, freeze_type, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, model_name, freeze_type)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, freeze_type)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, model_name, freeze_type)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, model_name, freeze_type)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'vgg16':
        
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, freeze_type)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # set_parameter_requires_grad(model_ft, model_name, freeze_type)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, model_name, freeze_type)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, model_name, freeze_type)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


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
    # parser.add_argument("-f", "--frozen", default=False, action='store_true',
    #                     help="True if you want to frozen all the layers except the last one")
    parser.add_argument("-f", "--freeze_type", help="For Resnet50, freeze_type could be: 'none', 'all', 'last_fc', 'top1_conv_block', 'top2_conv_block', 'top3_conv_block'. For VGG16, freeze_type could be: 'none', 'all', 'last_fc', 'fc2', 'fc1', 'top1_conv_block', 'top2_conv_block'")

    args = parser.parse_args()

    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    # CLASSES ID
    if args.dataset in ['mass_pathology', 'calc_pathology']:
        classes = np.array(['BENIGN', 'MALIGNANT'])
    elif args.dataset == 'mass_shape_comb_feats_omit':
        classes = np.array(['ROUND', 'OVAL', 'IRREGULAR', 'LOBULATED', 'ARCHITECTURAL_DISTORTION', 'ASYMMETRIC_BREAST_TISSUE', 'LYMPH_NODE', 'FOCAL_ASYMMETRIC_DENSITY'])
    elif args.dataset == 'mass_margins_comb_feats_omit':
        classes = np.array(['ILL_DEFINED', 'CIRCUMSCRIBED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'])
    elif args.dataset == 'calc_type_comb_feats_omit':
        classes = np.array(["AMORPHOUS", "PUNCTATE","VASCULAR","LARGE_RODLIKE","DYSTROPHIC","SKIN","MILK_OF_CALCIUM","EGGSHELL","PLEOMORPHIC","COARSE","FINE_LINEAR_BRANCHING","LUCENT_CENTER","ROUND_AND_REGULAR","LUCENT_CENTERED"])
    elif args.dataset == 'calc_dist_comb_feats_omit':
        classes = np.array(["CLUSTERED", "LINEAR","REGIONAL","DIFFUSELY_SCATTERED","SEGMENTAL"])
    elif args.dataset in ['mass_breast_density_lesion', 'mass_breast_density_image', 'calc_breast_density_lesion', 'calc_breast_density_image']:
        classes = np.array([1, 2, 3, 4])
        
        
    if args.dataset in ['mass_pathology', 'mass_shape_comb_feats_omit', 'mass_margins_comb_feats_omit', 'mass_breast_density_lesion', 'mass_breast_density_image']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][args.dataset])
    elif args.dataset in ['calc_pathology', 'calc_type_comb_feats_omit', 'calc_dist_comb_feats_omit', 'calc_breast_density_lesion', 'calc_breast_density_image']:
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
    num_classes = len(classes.tolist())

    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batch_size

    # Number of epochs to train
    num_epochs = args.epochs

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    # feature_extract = args.frozen

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_name, num_classes, args.freeze_type, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
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

    # Create training and validation datasets
    if args.dataset in ['mass_pathology', 'calc_pathology']:
        image_datasets = {x: Pathology_Dataset(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    elif args.dataset == 'mass_shape_comb_feats_omit':
        image_datasets = {x: Mass_Shape_Dataset(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    elif args.dataset == 'mass_margins_comb_feats_omit':
        image_datasets = {x: Mass_Margins_Dataset(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    elif args.dataset == 'calc_type_comb_feats_omit':
        image_datasets = {x: Calc_Type_Dataset(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    elif args.dataset == 'calc_dist_comb_feats_omit':
        image_datasets = {x: Calc_Dist_Dataset(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    elif args.dataset in ['mass_breast_density_lesion', 'mass_breast_density_image', 'calc_breast_density_lesion', 'calc_breast_density_image']:
        image_datasets = {x: Breast_Density_Dataset(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
        

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.freeze_type != 'none':
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                                                         num_epochs=num_epochs, is_inception=(model_name == "inception"))
    torch.save(model_ft.state_dict(), os.path.join(save_path, 'ckpt.pth'))

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
    test_dataloaders_dict = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)}

    model_ft.eval()
    
    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels = get_all_preds(model_ft, prediction_loader, device)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(labels.cpu(), classes=[*range(num_classes)])
        print(binarized_labels.shape, softmaxs.shape)

    ######### PLOT ##########
    matplotlib.use('Agg')

    # plot confusion matrix
    cm = confusion_matrix(labels.cpu(), preds.argmax(dim=1).cpu())

    # rm not existed classes for plotting confusion matrix
    test_classes = os.listdir(os.path.join(data_dir, 'test'))
    cm_test_classes = []
    for _class in classes:
        _class = str(_class)
        if _class in test_classes:
            cm_test_classes.append(_class)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, cm_test_classes)
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(10, 10))
    norm_cm = plot_confusion_matrix(cm, cm_test_classes, normalize=True)
    plt.savefig(os.path.join(save_path, 'norm_confusion_matrix.png'))
    plt.close()

    # Plot PR curve
    prs, recs, aps = [], [], []
    if args.dataset in ['mass_pathology', 'calc_pathology']:
        print(binarized_labels.shape, softmaxs.shape)
        pr, rec, ap = plot_precision_recall_curve(binarized_labels, softmaxs[:, 1].cpu(), class_name='Pathology')
        prs.append(pr)
        recs.append(rec)
        aps.append(ap)
    else:
        for i in range(num_classes):
            if np.sum(binarized_labels[:, i]) > 0:
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
        fpr, tpr, auc = plot_roc_curve(binarized_labels, softmaxs[:, 1].cpu(), class_name='Pathology')
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc)
    else:
        for i in range(num_classes):
            if np.sum(binarized_labels[:, i]) > 0:
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
