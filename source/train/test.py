from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib

from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score, roc_curve, roc_auc_score
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return cm

def plot_precision_recall_curve(y, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred)
    ap = average_precision_score(y, y_pred)
    ap = round(ap, 2)

    precisions = precisions[:-1].tolist()
    precisions.reverse()
    recalls = recalls[:-1].tolist()
    recalls.reverse()
    thresholds = thresholds[:-1].tolist()
    thresholds.reverse()

    for idx in range(0, len(precisions) - 1):
        precisions[idx] = max(precisions[idx+1:])

    plt.plot(recalls[:-1], precisions[:-1], "b-", label=f"pr curve (AP={ap})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    return precisions, recalls, ap


def plot_roc_curve(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    auc = round(auc, 2)

    plt.plot(fpr, tpr, "b-", label=f"roc curve (AUC={auc})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend()

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    return fpr, tpr, auc
    

def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
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


@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([], dtype=torch.long)
    all_labels = all_labels.to(device)
    for batch in loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        all_labels = torch.cat((all_labels, labels), dim=0)

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds), dim=0
        )
    return all_preds, all_labels


if __name__ == '__main__':
    data_dir = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/cls/mass_shape_rare_feats_omitted'
    model_name = 'resnet50'
    num_classes = 4
    # save_path = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/resnet50_fixed_mass_shape_rare_feats_omitted.pth'
    save_path = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/mass_shape_rare_r50_frozen-test_b32_e2/ckpt.pth'

    model_ft, input_size = initialize_model(
        model_name, num_classes, use_pretrained=True)

    model_ft.load_state_dict(torch.load(save_path))
    model_ft.eval()

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
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    with torch.no_grad():
        prediction_loader = dataloaders_dict['val']
        preds, labels = get_all_preds(model_ft, prediction_loader, device)
        print(preds.shape)
        print(labels.shape)

    stacked = torch.stack(
        (
            labels, preds.argmax(dim=1)
        ), dim=1
    )

    cmt = torch.zeros(10, 10, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1

    cm = confusion_matrix(labels.cpu(), preds.argmax(dim=1).cpu())

    matplotlib.use('Agg')
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, ['IRREGULAR', 'LOBULATED', 'OVAL', 'ROUND'])

    plt.savefig('confusion_matrix_fixed.png')
