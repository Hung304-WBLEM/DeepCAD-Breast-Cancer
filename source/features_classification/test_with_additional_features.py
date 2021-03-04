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
import custom_transforms
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix
from eval_utils import eval_all
from sklearn.preprocessing import label_binarize
from test import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from dataprocessing.process_cbis_ddsm import get_info_lesion
from config.cfg_loader import proj_paths_json
from skimage import io, transform
from torch import nn
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from train import set_parameter_requires_grad
from datasets import Features_Pathology_Dataset, Four_Classes_Features_Pathology_Dataset
from PIL import Image
from train_utils import compute_classes_weights, compute_classes_weights_mass_calc_pathology, compute_classes_weights_mass_calc_pathology_4class, compute_classes_weights_mass_calc_pathology_5class, set_seed, plot_train_val_loss


class Pathology_Model(nn.Module):
    def __init__(self, model_name, freeze_type, input_vector_dim, num_classes, use_pretrained=True):
        super(Pathology_Model, self).__init__()
        self.model_name = model_name
        if model_name == "resnet50":
            self.cnn = models.resnet50(pretrained=use_pretrained)
            # set_parameter_requires_grad(self.cnn, model_name, freeze_type)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            self.fc1 = nn.Linear(2048 + input_vector_dim, 512)
            self.fc2 = nn.Linear(512, num_classes)
        elif model_name == 'vgg16':
            self.cnn = models.vgg16_bn(pretrained=use_pretrained)
            # set_parameter_requires_grad(self.cnn, model_name, freeze_type)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-3])

            self.fc1 = nn.Linear(self.cnn.classifier[3].out_features + input_vector_dim, 512)
            self.fc2 = nn.Linear(512, num_classes)


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


def train_stage_pathology(model_ft, model_name, criterion, optimizer_type, freeze_type, learning_rate, weight_decay, num_epochs, dataloaders_dict):
    set_parameter_requires_grad(model_ft, model_name, freeze_type=freeze_type)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    # Observe that all parameters are being optimized
    if optimizer_type == 'sgd':
        optimizer_ft = optim.SGD(params_to_update,
                                 lr=learning_rate,
                                 weight_decay=weight_decay,
                                 momentum=0.9)
    elif optimizer_type == 'adam':
        optimizer_ft = optim.Adam(params_to_update,
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    # Train and evaluate
    model_ft, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_pathology_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                    num_epochs=num_epochs, is_inception=(model_name == "inception"))
    return model_ft, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

@torch.no_grad()
def get_all_preds_pathology(model, loader, device):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([], dtype=torch.long)
    all_labels = all_labels.to(device)
    all_paths = []

    for sample in loader:
        inputs = sample['image']
        img_path = sample['img_path']
        labels = sample['pathology']
        input_vectors = sample['feature_vector']
        input_vectors = input_vectors.type(torch.FloatTensor)

        inputs = inputs.to(device)
        labels = labels.to(device)
        input_vectors = input_vectors.to(device)

        all_labels = torch.cat((all_labels, labels), dim=0)

        preds = model(inputs, input_vectors, training=False)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_paths += img_path

    return all_preds, all_labels, all_paths
        


if __name__ == '__main__':
    ##############################################
    ############## Parse Arguments ###############
    ##############################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="Name of the available datasets")
    parser.add_argument("-s", "--save_path",
                        help="Path to save the trained model")
    parser.add_argument("-m", "--model_name",
                        help="Select the backbone for training. Available backbones include: 'resnet', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int,
                        help="the number of epochs for training")
    parser.add_argument("-wc", "--weighted_classes",
                        default=False, action='store_true',
                        help="enable if you want to train with classes weighting")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="Learning rate")
    parser.add_argument("-wd", "--weights_decay", type=float, default=0,
                        help="Weights decay")
    parser.add_argument("-opt", "--optimizer", type=str,
                        help="Choose optimizer: sgd, adam")
    parser.add_argument("-f", "--freeze_type",
                        help="For Resnet50, freeze_type could be: 'none', 'all', 'last_fc', 'top1_conv_block', 'top2_conv_block', 'top3_conv_block'. For VGG16, freeze_type could be: 'none', 'all', 'last_fc', 'fc2', 'fc1', 'top1_conv_block', 'top2_conv_block'")

    args = parser.parse_args()

    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    # Import dataset
    if args.dataset in ['mass_pathology', 'mass_pathology_clean', 'calc_pathology', 'calc_pathology_clean']:
        from datasets import Features_Pathology_Dataset as data
    elif args.dataset in ['four_classes_mass_calc_pathology']:
        from datasets import Four_Classes_Features_Pathology_Dataset as data

    # Get classes
    classes = data.classes

    if args.dataset in ['mass_pathology', 'mass_pathology_clean']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][args.dataset])
    elif args.dataset in ['calc_pathology', 'calc_pathology_clean']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][args.dataset])
    elif args.dataset in ['four_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])

    # Fix random seed
    set_seed()

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
    
    # Initialize model
    if args.dataset in ['mass_pathology', 'mass_pathology_clean']:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        model = Pathology_Model(model_name, args.freeze_type, input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats, num_classes=num_classes)
    elif args.dataset in ['calc_pathology', 'calc_pathology_clean']:
        breast_density_cats = 4
        calc_type_cats = 14
        calc_dist_cats = 5
        model = Pathology_Model(model_name, args.freeze_type, input_vector_dim=breast_density_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes)
    elif args.dataset in ['four_classes_mass_calc_pathology']:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        calc_type_cats = 14
        calc_dist_cats = 5
        model = Pathology_Model(model_name, args.freeze_type,
                                input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes)
        

    # print the model we just instantiated
    print(model)

    # Data augmentation and normalization for training
    # Just normalization for validation
    # input_size = 512 # remove after experiment
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(25, scale=(0.8, 1.2)),
            custom_transforms.IntensityShift((-20, 20)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    if args.dataset in ['mass_pathology', 'mass_pathology_clean']:
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif args.dataset in ['calc_pathology', 'calc_pathology_clean']:
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif args.dataset in ['four_classes_mass_calc_pathology']:
        pathology_datasets = \
            {x: Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, x),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, x),
                uncertainty=0.5,
                transform=data_transforms[x]
            ) for x in ['train', 'val']}
        


    dataloaders_dict = {x: DataLoader(pathology_datasets[x], batch_size=batch_size, worker_init_fn=np.random.seed(42), shuffle=True, num_workers=0) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup the loss fn
    if args.weighted_classes:
        print('Optimization with classes weighting')
        if args.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
            classes_weights = compute_classes_weights_mass_calc_pathology(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        elif args.dataset in ['four_classes_mass_calc_pathology']:
            classes_weights = compute_classes_weights_mass_calc_pathology_4class(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        else:
            classes_weights = compute_classes_weights(
                data_root=os.path.join(data_dir, 'train'), classes_names=classes)


        classes_weights = torch.from_numpy(
            classes_weights).type(torch.FloatTensor)
        criterion = nn.CrossEntropyLoss(weight=classes_weights.to(device))
    else:
        print('Optimization without classes weighting')
        criterion = nn.CrossEntropyLoss()

    # all_train_losses = []
    # all_val_losses = []
    # all_train_accs = []
    # all_val_accs = []

    # # 1st stage
    # model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = \
    #     train_stage_pathology(model, model_name, criterion,
    #                 optimizer_type=args.optimizer,
    #                 freeze_type='first_freeze',
    #                 learning_rate=0.001, weight_decay=0.01,
    #                 num_epochs=6, dataloaders_dict=dataloaders_dict)
    # all_train_losses.extend(train_loss_hist)
    # all_val_losses.extend(val_loss_hist)
    # all_train_accs.extend(train_acc_hist)
    # all_val_accs.extend(val_acc_hist)

    # # 2nd stage
    # model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = \
    #     train_stage_pathology(model, model_name, criterion,
    #                 optimizer_type=args.optimizer,
    #                 freeze_type='second_freeze',
    #                 learning_rate=0.0001, weight_decay=0.01,
    #                 num_epochs=20, dataloaders_dict=dataloaders_dict)
    # all_train_losses.extend(train_loss_hist)
    # all_val_losses.extend(val_loss_hist)
    # all_train_accs.extend(train_acc_hist)
    # all_val_accs.extend(val_acc_hist)

    # # 3rd stage
    # model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = \
    #     train_stage_pathology(model, model_name, criterion,
    #                 optimizer_type=args.optimizer,
    #                 freeze_type='third_freeze',
    #                 learning_rate=0.00001, weight_decay=0.01,
    #                 num_epochs=74, dataloaders_dict=dataloaders_dict)
    # all_train_losses.extend(train_loss_hist)
    # all_val_losses.extend(val_loss_hist)
    # all_train_accs.extend(train_acc_hist)
    # all_val_accs.extend(val_acc_hist)

    # # save best checkpoint 
    # torch.save(model.state_dict(), os.path.join(save_path, 'ckpt.pth'))

    # plot_train_val_loss(args.epochs, all_train_losses, all_val_losses,
    #                     all_train_accs, all_val_accs, save_path)


    ################### Test Model #############################
    if args.dataset in ['mass_pathology', 'mass_pathology_clean']:
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif args.dataset in ['calc_pathology', 'calc_pathology_clean']:
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif args.dataset in ['four_classes_mass_calc_pathology']:
        test_image_datasets = \
            {'test': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'test'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'test'),
                uncertainty=1,
                transform=data_transforms['test']
            )}

    test_dataloaders_dict = {'test': torch.utils.data.DataLoader(test_image_datasets['test'], batch_size=batch_size, shuffle=False, worker_init_fn=np.random.seed(42), num_workers=0)}

    model.load_state_dict(torch.load(os.path.join(save_path, 'ckpt.pth')))
    model.eval()
    
    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels, paths = get_all_preds_pathology(model, prediction_loader, device)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(labels.cpu(), classes=[*range(num_classes)])

    # eval_all(labels.cpu().detach().numpy(),
    #          softmaxs.cpu().detach().numpy(), classes, save_path)

    # predicted_labels = torch.max(softmaxs, 1).indices.cpu().numpy().tolist()
    # labels = labels.cpu().numpy().tolist()
    # paths = (paths)

    # with open(os.path.join(save_path, 'test_result.txt'), 'w') as f:
    #     for path, label, predict in zip(paths, labels, predicted_labels):
    #         f.write(' '.join((path, str(label), str(predict))) + '\n')

