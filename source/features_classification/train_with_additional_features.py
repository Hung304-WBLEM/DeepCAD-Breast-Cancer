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
import math
import pickle
import custom_transforms
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix, accuracy_score
from eval_utils import eval_all, evalplot_precision_recall_curve, evalplot_roc_curve, evalplot_confusion_matrix
from sklearn.preprocessing import label_binarize
from dataprocessing.process_cbis_ddsm import get_info_lesion
from config.cfg_loader import proj_paths_json
from config_fusion import options
from skimage import io, transform
from torch import nn
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from datasets import Features_Pathology_Dataset, Four_Classes_Features_Pathology_Dataset
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from train_utils import compute_classes_weights, compute_classes_weights_mass_calc_pathology, compute_classes_weights_mass_calc_pathology_4class, compute_classes_weights_mass_calc_pathology_5class
from train_utils import compute_classes_weights_within_batch
from train_utils import set_seed, plot_train_val_loss
from train_utils import images_to_probs_pathology, plot_classes_preds_pathology, add_pr_curve_tensorboard

GLOBAL_EPOCH = 0
test_image_datasets = None

def set_parameter_requires_grad(model, model_name, last_frozen_layer):
    '''
    Parameters:
    model_name - can be 'vgg16' or 'resnet50'
    freeze_type - can be 'none', 'all'
    '''
    if model_name in ['resnet50', 'dilated_resnet50']:
        for idx, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = True

        # last_frozen_idx = {'none': -1, 'all': 160, 'last_fc': 158, 'top1_conv_block': 149,
        #                     'top2_conv_block': 140, 'top3_conv_block': 128,
        #                     'first_freeze': 158, 'second_freeze': 87,
        #                     'third_freeze': -1}
        for idx, (name, param) in enumerate(model.named_parameters()):
            print(idx, name)
            # if idx <= last_frozen_idx[freeze_type]:
            if idx <= last_frozen_layer:
                param.requires_grad = False
    elif model_name == 'vgg16':
        if freeze_type != 'none':
            last_frozen_idx = {'all': 57, 'last_fc': 55, 'fc2': 53,
                               'fc1': 51, 'top1_conv_block': 45, 'top2_conv_block': 39}
            for idx, (name, param) in enumerate(model.named_parameters()):
                # print(idx, name)
                if idx <= last_frozen_idx[freeze_type]:
                    param.requires_grad = False


class Pathology_Model(nn.Module):
    def __init__(self, model_name, input_vector_dim, num_classes, use_pretrained=True):
        super(Pathology_Model, self).__init__()
        self.model_name = model_name
        if model_name == "resnet50":
            self.cnn = models.resnet50(pretrained=use_pretrained)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            self.fc1 = nn.Linear(2048 + input_vector_dim, 512)
            self.fc2 = nn.Linear(512, num_classes)
        elif model_name == 'vgg16':
            self.cnn = models.vgg16_bn(pretrained=use_pretrained)
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


class Attentive_Pathology_Model(nn.Module):
    def __init__(self, model_name, input_vector_dim, num_classes, attention_type='cross-attention', use_pretrained=True):
        '''
        Parameters:
        attention_type - select type of attention. Available options includes: 
                         co-attention, cross-attention
        '''

        super(Attentive_Pathology_Model, self).__init__()
        self.model_name = model_name
        self.attention_type = attention_type

        if model_name == 'resnet50':
            self.cnn = models.resnet50(pretrained=use_pretrained)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            self.img_emb_proj = nn.Linear(2048, 100)
            self.vec_emb_proj = nn.Linear(input_vector_dim, 100)

            if self.attention_type == 'co-attention':
                self.img_emb_att = nn.Linear(2048 + input_vector_dim, 100)
                self.vec_emb_att = nn.Linear(2048 + input_vector_dim, 100)
            elif self.attention_type == 'cross-attention':
                self.img_emb_att = nn.Linear(input_vector_dim, 100)
                self.vec_emb_att = nn.Linear(2048, 100)

            self.fc1 = nn.Linear(200, 200)
            self.fc1_att = nn.Linear(200, 200)

            self.fc2 = nn.Linear(200, num_classes)
            self.fc2_att = nn.Linear(200, num_classes)


    def forward(self, image, vector_data, training):
        img_emb = self.cnn(image)
        if self.model_name == 'resnet50':
            img_emb = img_emb.squeeze()
        vec_emb = vector_data

        proj_img_emb = F.relu(self.img_emb_proj(img_emb))
        proj_vec_emb = F.relu(self.vec_emb_proj(vec_emb))

        if self.attention_type == 'co-attention':
            alpha_img = torch.sigmoid(self.img_emb_att(torch.cat((img_emb, vec_emb), dim=1)))
            alpha_vec = torch.sigmoid(self.vec_emb_att(torch.cat((img_emb, vec_emb), dim=1)))
        elif self.attention_type == 'cross-attention':
            alpha_img = torch.sigmoid(self.img_emb_att(vec_emb))
            alpha_vec = torch.sigmoid(self.vec_emb_att(img_emb))
            

        aug_img_emb = torch.mul(proj_img_emb, alpha_img)
        aug_vec_emb = torch.mul(proj_vec_emb, alpha_vec)

        concat_emb = torch.cat((aug_img_emb, aug_vec_emb), dim=1)

        x = F.relu(self.fc1(concat_emb))
        alpha_x = torch.sigmoid(self.fc1_att(concat_emb))
        aug_x = torch.mul(x, alpha_x)

        x = self.fc2(aug_x)
        alpha_x = torch.sigmoid(self.fc2_att(aug_x))
        aug_x = torch.mul(x, alpha_x)

        x = aug_x

        return x


def train_pathology_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25, weight_sample=True, is_inception=False):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        global GLOBAL_EPOCH
        GLOBAL_EPOCH += 1

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('Global epoch:', GLOBAL_EPOCH)
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
                labels = sample['label']
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

                    if weight_sample and phase == 'train':
                        sample_weight = compute_classes_weights_within_batch(labels)
                        sample_weight = torch.from_numpy(np.array(sample_weight)).to(device)
                        loss = (loss * sample_weight / sample_weight.sum()).sum()

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

            # For tensorboard
            writer.add_scalar(f'{phase} loss', epoch_loss, GLOBAL_EPOCH)
            writer.add_scalar(f'{phase} acc', epoch_acc, GLOBAL_EPOCH)
            writer.add_figure(f'{phase} predictions vs. actuals',
                              plot_classes_preds_pathology(model, inputs, input_vectors, labels, classes, num_images=min(inputs.shape[0], 16)),
                              global_step=GLOBAL_EPOCH)
            evaluate_pathology(model, classes, device, writer, epoch=GLOBAL_EPOCH)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
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
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def train_stage_pathology(model, model_name, criterion, optimizer_type, last_frozen_layer, learning_rate, weight_decay, num_epochs, dataloaders_dict, weighted_samples, writer):
    set_parameter_requires_grad(model, model_name, last_frozen_layer)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
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
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_pathology_model(model, dataloaders_dict, criterion, optimizer_ft, writer,
                              num_epochs=num_epochs, weight_sample=weighted_samples, is_inception=(model_name == "inception"))
    return model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

@torch.no_grad()
def get_all_preds_pathology(model, loader, device, classes=None, plot_test_images=False):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([], dtype=torch.long)
    all_labels = all_labels.to(device)

    for idx, data_info in enumerate(loader):
        images = data_info['image']
        labels = data_info['label']
        input_vectors = data_info['feature_vector']
        input_vectors = input_vectors.type(torch.FloatTensor)

        images = images.to(device)
        labels = labels.to(device)
        input_vectors = input_vectors.to(device)

        if plot_test_images:
            writer.add_figure(f'test predictions vs. actuals',
                                plot_classes_preds_pathology(model, images, input_vectors, labels, classes, num_images=images.shape[0]),
                                global_step=idx)

        all_labels = torch.cat((all_labels, labels), dim=0)

        preds = model(images, input_vectors, training=False)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds, all_labels


@torch.no_grad()
def evaluate_pathology(model, classes, device, writer, epoch):
    model.eval()
    global pathology_datasets

    test_dataloaders_dict = {
        'test': torch.utils.data.DataLoader(
            test_image_datasets['test'], batch_size=32,
            shuffle=False,
            worker_init_fn=np.random.seed(42), num_workers=options.num_workers)}

    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels = get_all_preds_pathology(model, prediction_loader, device)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(
            labels.cpu(), classes=[*range(num_classes)])

    y_true = labels.cpu().detach().numpy()
    y_proba_pred = softmaxs.cpu().detach().numpy()
    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])
    y_pred = y_proba_pred.argmax(axis=1)

    # accuracy
    acc = accuracy_score(y_true, y_pred)
    writer.add_scalar(f'test acc', acc, epoch)

    # AUCs
    _, _, pr_aucs = evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes)
    _, _, roc_aucs = evalplot_roc_curve(binarized_y_true, y_proba_pred, classes)

    idx = 0
    for class_id, class_name in enumerate(classes):
        if np.sum(binarized_y_true[:, class_id]) > 0:
            writer.add_scalar(f'test pr auc - {class_name}', pr_aucs[idx], epoch)
            writer.add_scalar(f'test roc auc - {class_name}', roc_aucs[idx], epoch)
            idx += 1

    model.train()


@torch.no_grad()
def final_evaluate_pathology(model, classes, device, writer):
    model.eval()
    global test_image_datasets

    test_dataloaders_dict = {
        'test': torch.utils.data.DataLoader(
            test_image_datasets['test'], batch_size=16,
            shuffle=False,
            worker_init_fn=np.random.seed(42), num_workers=options.num_workers)}

    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels = get_all_preds_pathology(model, prediction_loader, device, classes, plot_test_images=True)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(
            labels.cpu(), classes=[*range(num_classes)])

    y_true = labels.cpu().detach().numpy()
    y_proba_pred = softmaxs.cpu().detach().numpy()
    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])
    y_pred = y_proba_pred.argmax(axis=1)

    writer.add_figure(f'test confusion matrix',
                      evalplot_confusion_matrix(y_true,
                                                y_pred, classes, fig_only=True),
                        global_step=None)
    writer.add_figure(f'test roc curve',
                        evalplot_roc_curve(binarized_y_true,
                                           y_proba_pred, classes, fig_only=True),
                        global_step=None)
    writer.add_figure(f'test pr curve',
                        evalplot_precision_recall_curve(binarized_y_true,
                                           y_proba_pred, classes, fig_only=True),
                        global_step=None)

if __name__ == '__main__':
    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    # Import dataset
    if options.dataset in ['mass_pathology', 'mass_pathology_clean', 'calc_pathology', 'calc_pathology_clean']:
        from datasets import Features_Pathology_Dataset as data
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        from datasets import Four_Classes_Features_Pathology_Dataset as data

    # Get classes
    classes = data.classes

    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][options.dataset])
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][options.dataset])
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])

    # Fix random seed
    set_seed()

    save_path = options.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    # TensorBoard Summary Writer
    # writer = SummaryWriter('runs/' + datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard_logs'))

    # Models to choose from [resnet, resnet50, alexnet, vgg, squeezenet, densenet, inception]
    model_name = options.model_name

    # Number of classes in the dataset
    num_classes = len(classes.tolist())

    # Batch size for training (change depending on how much memory you have)
    batch_size = options.batch_size

    # Number of epochs to train
    num_epochs = options.epochs
    
    # Initialize model
    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        model = Attentive_Pathology_Model(model_name,
                                input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats, num_classes=num_classes)
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        breast_density_cats = 4
        calc_type_cats = 14
        calc_dist_cats = 5
        model = Attentive_Pathology_Model(model_name,
                                input_vector_dim=breast_density_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes)
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        calc_type_cats = 14
        calc_dist_cats = 5
        model = Attentive_Pathology_Model(model_name, 
                                input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes)
        

    # print the model we just instantiated
    print(model)

    # Data augmentation and normalization for training
    input_size = options.input_size
    if options.augmentation_type == 'torch':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(25, scale=(0.8, 1.2)),
                custom_transforms.IntensityShift((-20, 20)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif options.augmentation_type == 'albumentations':
        data_transforms = {
            'train': albumentations.Compose([
                albumentations.Transpose(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.RandomContrast(limit=0.2, p=0.75),
                albumentations.OneOf([
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                albumentations.OneOf([
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                    albumentations.ElasticTransform(alpha=3),
                ], p=0.7),

                albumentations.CLAHE(clip_limit=4.0, p=0.7),
                albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                albumentations.Resize(input_size, input_size),
                albumentations.Cutout(max_h_size=int(input_size * 0.375), max_w_size=int(input_size * 0.375), num_holes=1, p=0.7),
                albumentations.Normalize()
            ]),
            'val': albumentations.Compose([
                albumentations.Resize(input_size, input_size),
                albumentations.Normalize()
            ]),
            'test': albumentations.Compose([
                albumentations.Resize(input_size, input_size),
                albumentations.Normalize()
            ])
        }

    print("Initializing Datasets and Dataloaders...")

    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        pathology_datasets = \
            {'train': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'train'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'train'),
                uncertainty=options.train_uncertainty,
                transform=data_transforms['train']
            ),
            'val': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'val'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'val'),
                uncertainty=options.test_uncertainty,
                transform=data_transforms['val']
            )}

    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        test_image_datasets = \
            {'test': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'test'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'test'),
                uncertainty=options.test_uncertainty,
                transform=data_transforms['test']
            )}
        

    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(
            pathology_datasets['train'], batch_size=batch_size,
            worker_init_fn=np.random.seed(42),
            shuffle=True, num_workers=options.num_workers),
        'val': torch.utils.data.DataLoader(
            pathology_datasets['val'], batch_size=batch_size,
            worker_init_fn=np.random.seed(42),
            shuffle=False, num_workers=options.num_workers)
    }

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup the loss fn
    if options.weighted_classes:
        print('Optimization with classes weighting')
        if options.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
            classes_weights = compute_classes_weights_mass_calc_pathology(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        elif options.dataset in ['four_classes_mass_calc_pathology']:
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

    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    # 1st stage
    epochs_1st_stage = int(math.ceil(options.epochs * 0.06))
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_stage_pathology(model, model_name, criterion,
                    optimizer_type=options.optimizer,
                    # freeze_type='first_freeze',
                    last_frozen_layer=options.first_stage_last_frozen_layer,
                    # learning_rate=0.001, weight_decay=0.01,
                    learning_rate=options.first_stage_learning_rate,
                    weight_decay=options.first_stage_weight_decay,
                    num_epochs=epochs_1st_stage,
                    dataloaders_dict=dataloaders_dict,
                    weighted_samples=options.weighted_samples,
                    writer=writer)
    all_train_losses.extend(train_loss_hist)
    all_val_losses.extend(val_loss_hist)
    all_train_accs.extend(train_acc_hist)
    all_val_accs.extend(val_acc_hist)

    # 2nd stage
    epochs_2nd_stage = int(math.ceil(options.epochs * 0.2))
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_stage_pathology(model, model_name, criterion,
                    optimizer_type=options.optimizer,
                    # freeze_type='second_freeze',
                    last_frozen_layer=options.second_stage_last_frozen_layer,
                    # learning_rate=0.0001, weight_decay=0.01,
                    learning_rate=options.second_stage_learning_rate,
                    weight_decay=options.second_stage_weight_decay,
                    num_epochs=epochs_2nd_stage, dataloaders_dict=dataloaders_dict,
                    weighted_samples=options.weighted_samples,
                    writer=writer)
    all_train_losses.extend(train_loss_hist)
    all_val_losses.extend(val_loss_hist)
    all_train_accs.extend(train_acc_hist)
    all_val_accs.extend(val_acc_hist)

    # 3rd stage
    if options.train_with_fourth_stage:
        epochs_3rd_stage = int(math.ceil(options.epochs * 0.4))
    else:
        epochs_3rd_stage = options.epochs - (epochs_1st_stage + epochs_2nd_stage)
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_stage_pathology(model, model_name, criterion,
                    optimizer_type=options.optimizer,
                    # freeze_type='third_freeze',
                    last_frozen_layer=options.third_stage_last_frozen_layer,
                    # learning_rate=0.00001, weight_decay=0.01,
                    learning_rate=options.third_stage_learning_rate,
                    weight_decay=options.third_stage_weight_decay,
                    num_epochs=epochs_3rd_stage, dataloaders_dict=dataloaders_dict,
                    weighted_samples=options.weighted_samples,
                    writer=writer)
    all_train_losses.extend(train_loss_hist)
    all_val_losses.extend(val_loss_hist)
    all_train_accs.extend(train_acc_hist)
    all_val_accs.extend(val_acc_hist)

    # 4th stage
    if options.train_with_fourth_stage:
        epochs_4th_stage = options.epochs - (epochs_1st_stage + epochs_2nd_stage + epochs_3rd_stage)
        model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
            train_stage_pathology(model, model_name, criterion,
                        optimizer_type=options.optimizer,
                        # freeze_type='third_freeze',
                        last_frozen_layer=options.fourth_stage_last_frozen_layer,
                        # learning_rate=0.00001, weight_decay=0.01,
                        learning_rate=options.fourth_stage_learning_rate,
                        weight_decay=options.fourth_stage_weight_decay,
                        num_epochs=epochs_4th_stage, dataloaders_dict=dataloaders_dict,
                        weighted_samples=options.weighted_samples,
                        writer=writer)
        all_train_losses.extend(train_loss_hist)
        all_val_losses.extend(val_loss_hist)
        all_train_accs.extend(train_acc_hist)
        all_val_accs.extend(val_acc_hist)
    # save best checkpoint 
    torch.save(model.state_dict(), os.path.join(save_path, 'ckpt.pth'))

    plot_train_val_loss(options.epochs, all_train_losses, all_val_losses,
                        all_train_accs, all_val_accs, save_path)


    ################### Test Model #############################

    test_dataloaders_dict = {'test': torch.utils.data.DataLoader(test_image_datasets['test'], batch_size=batch_size, shuffle=False, worker_init_fn=np.random.seed(42), num_workers=options.num_workers)}

    model.eval()
    
    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels = get_all_preds_pathology(model, prediction_loader, device)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(labels.cpu(), classes=[*range(num_classes)])

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(writer, classes, i, labels.cpu().detach().numpy(), softmaxs.cpu().detach().numpy())

    # my roc curve
    final_evaluate_pathology(model, classes, device, writer)
    
    eval_all(labels.cpu().detach().numpy(),
             softmaxs.cpu().detach().numpy(), classes, save_path)
