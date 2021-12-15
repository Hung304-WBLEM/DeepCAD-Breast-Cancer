import torch.optim as optim
import time
import copy
import logging
import torch
import numpy as np

from features_classification.models.model_initializer import initialize_model, set_parameter_requires_grad
from features_classification.eval.eval_funcs import evaluate
from features_classification.eval.eval_utils import plot_classes_preds
from features_classification.train.train_utils import compute_classes_weights_within_batch

GLOBAL_EPOCH = 0

def train_model(options, model, dataloaders_dict, criterion, optimizer, writer, device, classes, dataset, num_epochs=25, weight_sample=True, is_inception=False):
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
            for it, data_info in enumerate(dataloaders_dict[phase]):
                # enumerate is used here to reset data loader

                inputs = data_info['image']
                labels = data_info['label']


                inputs = inputs.to(device)
                labels = labels.to(device)

                if options.criterion == 'bce':
                    binarized_multilabels = data_info['binarized_multilabel']
                    binarized_multilabels = binarized_multilabels.to(device)

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

                        if options.criterion == 'ce':
                            loss = criterion(outputs, labels)
                        elif options.criterion == 'bce':
                            loss = criterion(outputs, binarized_multilabels)

                    if weight_sample and phase == 'train':
                        sample_weight = compute_classes_weights_within_batch(labels)
                        sample_weight = torch.from_numpy(np.array(sample_weight)).to(device)
                        loss = (loss * sample_weight / sample_weight.sum()).sum()

                    # if options.criterion == 'ce':
                    #     _, preds = torch.max(outputs, 1)
                    # elif options.criterion == 'bce':
                    #     preds = (torch.sigmoid(outputs) > 0.5).int()

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)


                # ignore accuracy metric when training for multi-label classification
                # if options.criterion == 'ce':
                #     running_corrects += torch.sum(preds == labels.data)
                # elif options.criterion == 'bce':
                #     running_corrects += torch.sum(torch.sum(preds == binarized_multilabels.data, dim=1) == labels.data)

                running_corrects += torch.sum(preds == labels.data)

                if it == 0:
                    writer.add_figure(f'{phase} predictions vs. actuals',
                                    plot_classes_preds(model, inputs, labels,
                                                       num_images=inputs.shape[0],
                                                       multilabel_mode=(options.criterion=='bce'),
                                                       dataset=dataset
                                                       ),
                                    global_step=GLOBAL_EPOCH)
                #     


            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # For tensorboard
            writer.add_scalar(f'{phase} loss', epoch_loss, GLOBAL_EPOCH)
            writer.add_scalar(f'{phase} acc', epoch_acc, GLOBAL_EPOCH)

            if options.criterion == 'ce':
                _multilabel_mode = False
            elif options.criterion == 'bce':
                _multilabel_mode = True

            evaluate(model, classes, dataloaders_dict['test'],
                     device, writer, epoch=GLOBAL_EPOCH,
                     multilabel_mode=_multilabel_mode,
                     dataset=dataset
                     )

            # deep copy the model
            if options.best_ckpt_metric == 'acc' and \
               phase == 'val' and epoch_acc > best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if options.best_ckpt_metric == 'loss' and \
               phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu())
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu())

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

    print(type(train_loss_history), type(train_acc_history),
          type(val_loss_history), type(val_acc_history))
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def train_stage(options, model_ft, model_name, criterion, optimizer_type, last_frozen_layer, learning_rate, weight_decay, dataset, num_epochs, dataloaders_dict, weighted_samples, writer, device, classes):
    set_parameter_requires_grad(model_ft, model_name, last_frozen_layer)
    
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
        train_model(options, model_ft, dataloaders_dict,
                    criterion, optimizer_ft, writer, device, classes,
                    dataset=dataset,
                    num_epochs=num_epochs, weight_sample=weighted_samples,
                    is_inception=(model_name == "inception"))
    return model_ft, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist
