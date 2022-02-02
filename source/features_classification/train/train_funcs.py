import torch.optim as optim
import time
import copy
import logging
import torch
import numpy as np
import transformers

from features_classification.models.model_initializer import initialize_model, set_parameter_requires_grad
from features_classification.eval.eval_funcs import evaluate
from features_classification.eval.eval_utils import plot_classes_preds
from features_classification.train.train_utils import compute_classes_weights_within_batch


GLOBAL_EPOCH = 0


def train_model(options, model, dataloaders_dict, criterion, optimizer, writer, device, classes, dataset, num_epochs=25, weight_sample=True, is_inception=False, lr_scheduler=None):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    best_ckpt_metric = options.best_ckpt_metric
    best_model_wts = copy.deepcopy(model.state_dict())
    best_eval = 0.0
    best_acc = 0.0

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

                if options.use_clinical_feats:
                    input_vectors = data_info['feature_vector'].type(torch.FloatTensor)
                    input_vectors = input_vectors.to(device)

                if options.criterion == 'bce':
                    binarized_multilabels = data_info['binarized_multilabel']
                    binarized_multilabels = binarized_multilabels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        if not options.use_clinical_feats:
                            outputs = model(inputs)
                        else:
                            # if phase == 'val':
                            #     outputs = model(inputs, input_vectors, training=False)
                            # elif phase == 'train':
                            #     outputs = model(inputs, input_vectors, training=True)
                            outputs = model(inputs, input_vectors)

                        if options.criterion == 'ce':
                            loss = criterion(outputs, labels)
                        elif options.criterion == 'bce':
                            loss = criterion(outputs, binarized_multilabels)

                    if weight_sample and phase == 'train':
                        sample_weight = compute_classes_weights_within_batch(labels)
                        sample_weight = torch.from_numpy(np.array(sample_weight)).to(device)
                        loss = (loss * sample_weight / sample_weight.sum()).sum()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)


                if it == 0:
                    if not options.use_clinical_feats:
                        writer.add_figure(f'{phase} predictions vs. actuals',
                                        plot_classes_preds(model, inputs, labels,
                                                            num_images=inputs.shape[0],
                                                            multilabel_mode=\
                                                            (options.criterion=='bce'),
                                                            dataset=dataset
                                                            ),
                                        global_step=GLOBAL_EPOCH)
                    else:
                        writer.add_figure(f'{phase} predictions vs. actuals',
                                        plot_classes_preds(model, inputs, labels,
                                                           num_images=inputs.shape[0],
                                                           multilabel_mode=\
                                                           (options.criterion=='bce'),
                                                           dataset=dataset,
                                                           input_vectors=input_vectors
                                                           ),
                                        global_step=GLOBAL_EPOCH)


            # Calculate Epoch Loss
            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            writer.add_scalar(f'{phase} loss', epoch_loss, GLOBAL_EPOCH)

            # Calculate other evaluation metrics (Acc, AP, AUC)
            if options.criterion == 'ce':
                _multilabel_mode = False
            elif options.criterion == 'bce':
                _multilabel_mode = True

            # Evaluate on train/val set at each epoch
            epoch_acc, epoch_macro_ap, epoch_micro_ap, \
                epoch_macro_auc, epoch_micro_auc = \
                    evaluate(model, classes, dataloaders_dict[phase],
                             device, writer, epoch=GLOBAL_EPOCH,
                             multilabel_mode=_multilabel_mode,
                             dataset=dataset, eval_split=phase,
                             use_clinical_feats=options.use_clinical_feats
                             )

            print('{:>5} Loss: {:.4f} Acc: {:.4f} \
            Macro AP: {:.4f} Micro AP: {:.4f} \
            Macro AUC: {:.4f} Micro AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,
                epoch_macro_ap, epoch_micro_ap,
                epoch_macro_auc, epoch_micro_auc))

            logging.info('{} Loss: {:.4f} Acc: {:.4f} \
            Macro AP: {:.4f} Micro AP: {:.4f} \
            Macro AUC: {:.4f} Micro AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,
                epoch_macro_ap, epoch_micro_ap,
                epoch_macro_auc, epoch_micro_auc))

            # Evaluate on test set at each epoch
            evaluate(model, classes, dataloaders_dict['test'],
                     device, writer, epoch=GLOBAL_EPOCH,
                     multilabel_mode=_multilabel_mode,
                     dataset=dataset, eval_split='test',
                     use_clinical_feats=options.use_clinical_feats
                     )

            epoch_info = {
                'acc': epoch_acc,
                'macro_ap': epoch_macro_ap,
                'micro_ap': epoch_micro_ap,
                'macro_auc': epoch_macro_auc,
                'macro_auc_only': epoch_macro_auc,
                'micro_auc': epoch_micro_auc
            }

            # deep copy the model
            if phase == 'val' and epoch_info[best_ckpt_metric] >= best_eval:
                if best_ckpt_metric == 'macro_auc':
                    if epoch_info['acc'] > best_acc:
                        print('[+] Update best ckpt: ' + \
                              'Old Macro AUC {}; Old Acc {}; '.format(best_eval,
                                                                      round(best_acc, 2)) + \
                              'New Macro AUC {}; New Acc {}'.format(epoch_info['macro_auc'],
                                                                    round(epoch_info['acc'], 2)))
                        best_loss = epoch_loss
                        best_eval = epoch_info[best_ckpt_metric]
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_acc = epoch_info['acc']
                else:
                    best_loss = epoch_loss
                    best_eval = epoch_info[best_ckpt_metric]
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
    print('Best val {}: {:4f}'.format(best_ckpt_metric, best_eval))

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Loss: {:4f}'.format(best_loss))
    logging.info('Best val {}: {:4f}'.format(best_ckpt_metric, best_eval))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def train_stage(options, model_ft, model_name, criterion, optimizer_type, last_frozen_layer, learning_rate, weight_decay, dataset, num_epochs, dataloaders_dict, weighted_samples, writer, device, classes):
    # set_parameter_requires_grad(model_ft, model_name, last_frozen_layer)
    
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

    if options.use_lr_scheduler:
        total_samples = len(dataloaders_dict['train'].dataset)
        bs = options.batch_size 

        num_warmup_steps = (total_samples // bs) * 2
        num_total_steps = (total_samples // bs) * num_epochs
        lr_scheduler = \
            transformers.get_cosine_schedule_with_warmup(optimizer_ft, 
                                                         num_warmup_steps=num_warmup_steps, 
                                                         num_training_steps=num_total_steps)

    # Train and evaluate
    model_ft, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        train_model(options, model_ft, dataloaders_dict,
                    criterion, optimizer_ft, writer, device, classes,
                    dataset=dataset,
                    num_epochs=num_epochs, weight_sample=weighted_samples,
                    is_inception=(model_name == "inception"),
                    lr_scheduler=lr_scheduler)
    return model_ft, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist
