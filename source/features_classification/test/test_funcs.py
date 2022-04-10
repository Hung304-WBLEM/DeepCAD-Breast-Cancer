import torch
import numpy as np
import torch.nn.functional as F

from features_classification.eval.eval_utils import plot_classes_preds

@torch.no_grad()
def get_all_preds(model, loader, device, writer, multilabel_mode, dataset,
                  plot_test_images=False, use_clinical_feats=False,
                  use_clinical_feats_only=False,
                  parallel_output=False):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)

    all_labels = torch.tensor([], dtype=torch.long)
    all_labels = all_labels.to(device)

    all_paths = []

    for idx, data_info in enumerate(loader):
        images = data_info['image']
        labels = data_info['label']
        image_paths = data_info['img_path']

        images = images.to(device)
        labels = labels.to(device)

        input_vectors = None
        if use_clinical_feats or use_clinical_feats_only:
            input_vectors = data_info['feature_vector'].type(torch.FloatTensor)
            input_vectors = input_vectors.to(device)

        if plot_test_images:
            writer.add_figure(f'test predictions vs. actuals',
                            plot_classes_preds(model, images, labels,
                                               num_images=images.shape[0],
                                               multilabel_mode=multilabel_mode,
                                               dataset=dataset,
                                               input_vectors=input_vectors,
                                               input_vectors_only=use_clinical_feats_only,
                                               parallel_output=parallel_output),
                            global_step=idx)


        all_labels = torch.cat((all_labels, labels), dim=0)

        if not (use_clinical_feats or use_clinical_feats_only):
            preds = model(images)
        elif use_clinical_feats_only:
            preds = model(input_vectors)
        elif use_clinical_feats:
            preds = model(images, input_vectors)

        if parallel_output:
            preds = preds[1]

        all_preds = torch.cat(
            (all_preds, preds), dim=0
        )

        all_paths += image_paths
            
    return all_preds, all_labels, all_paths


