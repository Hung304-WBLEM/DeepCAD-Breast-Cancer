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

from dataprocessing.process_cbis_ddsm import get_info_lesion
from skimage import io, transform
from torch import nn
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Pathology_Model(nn.Module):
    def __init__(self, input_vector_dim):
        super(Pathology_Model, self).__init__()
        self.cnn = models.vgg16(pretrained=True)
        self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-3])

        self.fc1 = nn.Linear(self.cnn.classifier[3].out_features + input_vector_dim, 512)
        self.fc2 = nn.Linear(512, 2)


    def forward(self, image, vector_data):
        x1 = self.cnn(image)
        x2 = vector_data

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def convert_mass_feats_1hot(breast_density, mass_shape, mass_margins):
    BREAST_DENSITY_TYPES = np.array([1, 2, 3, 4])
    BREAST_MASS_SHAPES = np.array(['ROUND', 'OVAL', 'IRREGULAR', 'LOBULATED', 'ARCHITECTURAL_DISTORTION', 'ASYMMETRIC_BREAST_TISSUE', 'LYMPH_NODE', 'FOCAL_ASYMMETRIC_DENSITY'])
    BREAST_MASS_MARGINS = np.array(['ILL_DEFINED', 'CIRCUMSCRIBED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'])

    one_hot_breast_density = (BREAST_DENSITY_TYPES == breast_density).astype('int')
    one_hot_mass_shape = (BREAST_MASS_SHAPES == mass_shape).astype('int')
    one_hot_mass_margins = (BREAST_MASS_MARGINS == mass_margins).astype('int')

    ret = np.concatenate((one_hot_breast_density, one_hot_mass_shape, one_hot_mass_margins))
    assert np.sum(ret) >= 3

    return ret

def convert_calc_feats_1hot(breast_density, calc_type, calc_distribution):
    BREAST_DENSITY_TYPES = np.array([1, 2, 3, 4])
    BREAST_CALC_TYPES = np.array(["AMORPHOUS", "PUNCTATE","VASCULAR","LARGE_RODLIKE","DYSTROPHIC","SKIN","MILK_OF_CALCIUM","EGGSHELL","PLEOMORPHIC","COARSE","FINE_LINEAR_BRANCHING","LUCENT_CENTER","ROUND_AND_REGULAR","LUCENT_CENTERED"])
    BREAST_CALC_DISTS = np.array(["CLUSTERED", "LINEAR","REGIONAL","DIFFUSELY_SCATTERED","SEGMENTAL"])

    one_hot_breast_density = (BREAST_DENSITY_TYPES == breast_density).astype('int')
    one_hot_calc_type = (BREAST_CALC_TYPES == calc_type).astype('int')
    one_hot_calc_distribution = (BREAST_CALC_DISTS == calc_distribution).astype('int')

    ret = np.concatenate((one_hot_breast_density, one_hot_calc_type, one_hot_calc_distribution))
    assert np.sum(ret) >= 3

    return ret
    

class Pathology_Dataset(Dataset):
    def __init__(self, lesion_type, annotation_file, root_dir, transform=None):
        self.annotations = pd.read_csv(os.path.join(root_dir, annotation_file))
        self.root_dir = root_dir
        self.transform = transform
        self.lesion_type = lesion_type

        malignant_images_list = glob.glob(os.path.join(root_dir, 'MALIGNANT', '*.png'))
        benign_images_list = glob.glob(os.path.join(root_dir, 'BENIGN', '*.png'))
        self.images_list = malignant_images_list + benign_images_list
        self.labels = [0] * len(malignant_images_list) + [1] * len(benign_images_list)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        # image = io.imread(img_path)
        image = Image.open(img_path)
        label = self.labels[idx]

        roi_idx = 0
        while True:
            roi_idx += 1
            rslt_df = get_info_lesion(self.annotations, f'{img_name}')

            if len(rslt_df) > 0:
                break

        if self.lesion_type == 'mass':
            breast_density = rslt_df['breast_density'].to_numpy()[0]
            mass_shape = rslt_df['mass shape'].to_numpy()[0]
            mass_margins = rslt_df['mass margins'].to_numpy()[0]
            input_vector = convert_mass_feats_1hot(breast_density, mass_shape, mass_margins)
        elif self.lesion_type == 'calc':
            breast_density = rslt_df['breast density'].to_numpy()[0]
            calc_type = rslt_df['calc type'].to_numpy()[0]
            calc_distribution = rslt_df['calc distribution'].to_numpy()[0]
            input_vector = convert_calc_feats_1hot(breast_density, calc_type, calc_distribution)
            
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'pathology': label, 'input_vector': input_vector}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for idx, (name, param) in enumerate(model.named_parameters()):
            print(idx, name)
            param.requires_grad = False


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
                input_vectors = sample['input_vector']
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
                        outputs = model(inputs, input_vectors)
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

if __name__ == '__main__':
    breast_density_cats = 4
    mass_shape_cats= 8
    mass_margins_cats = 5
    model = Pathology_Model(input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats)

    set_parameter_requires_grad(model, False)

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
    }

    pathology_datasets = \
        {x: Pathology_Dataset(lesion_type='mass',
            annotation_file=f'mass_case_description_train_set.csv',
            root_dir=f'/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/cls/mass_pathology/{x}', transform=data_transforms[x]) for x in ['train', 'val']}


    # pathology_dataloader = DataLoader(pathology_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloaders_dict = {x: DataLoader(pathology_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Select params to update
    feature_extract = False
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
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

    num_epochs = 30
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_pathology_model(model,
                              dataloaders_dict,
                              criterion,
                              optimizer_ft,
                              num_epochs=num_epochs,
                              is_inception=False)


