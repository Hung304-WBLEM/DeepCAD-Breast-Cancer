import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models, transforms, utils
from train_utils import compute_classes_weights_within_batch

# CFO == Clinical Features Only
class CFO_Pathology_Model(nn.Module):
    def __init__(self, num_classes):
        super(Clinical_Feats_Only_Pathology_Model, self).__init__()
        
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, vector_data, training):
        x = F.relu(self.fc1(vector_data))
        x = F.dropout(x, p=0.5, training=training)
        x = self.fc2(x)

        return x


def train_cfo_pathology_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25, weight_sample=True):
    
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
                labels = sample['label']
                input_vectors = sample['feature_vector']
                input_vectors = input_vectors.type(torch.FloatTensor)

                labels = labels.to(device)
                input_vectors = input_vectors.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
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


def train_stage_cfo_pathology(model, criterion, optimizer_type, learning_rate, weight_decay, num_epochs, dataloaders_dict, weighted_samples, writer):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    for name, param in model.named_parameters():
        param.requires_grad = True

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
        train_cfo_pathology_model(model, dataloaders_dict, criterion, optimizer_ft, writer,
                              num_epochs=num_epochs, weight_sample=weighted_samples, is_inception=(model_name == "inception"))
    return model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist


@torch.no_grad()
def get_all_preds_cfo_pathology(model, loader, device, classes=None, plot_test_images=False, use_predicted_feats=False):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)

    all_labels = torch.tensor([], dtype=torch.long)
    all_labels = all_labels.to(device)

    all_paths = []

    for idx, data_info in enumerate(loader):
        images = data_info['image']
        labels = data_info['label']
        image_paths = data_info['img_path']

        if use_predicted_feats:
            input_vectors = get_predicted_clinical_feats(lesion_paths=image_paths)
        else:
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

        all_paths += image_paths

    return all_preds, all_labels, all_paths
