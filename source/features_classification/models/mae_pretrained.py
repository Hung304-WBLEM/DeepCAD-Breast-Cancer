import torch
import torch.nn as nn

from mae.util.pos_embed import interpolate_pos_embed
from mae import models_vit
from timm.models.layers import trunc_normal_


def load_mae_pretrained_model(ckpt_path, arch, img_size,
                              num_classes, drop_path=0.1, global_pool=True):
    model = models_vit.__dict__[arch](
        num_classes=num_classes,
        drop_path_rate=drop_path,
        global_pool=global_pool,
        img_size=img_size
    )

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % ckpt_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    return model


def load_mae_pretrained_model_linprob(ckpt_path, arch, img_size,
                                      num_classes, drop_path=0.1, global_pool=True):
    model = models_vit.__dict__[arch](
        num_classes=num_classes,
        global_pool=global_pool,
        img_size=img_size
    )

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % ckpt_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    return model


class ViT_MAE(nn.Module):
    def __init__(self, ckpt_path, arch, img_size, num_classes,
                 drop_path=0.1, global_pool=True, linprobe=False):
        super(ViT_MAE, self).__init__()
        
        if not linprobe:
            self.model = load_mae_pretrained_model(ckpt_path, arch, img_size,
                                                num_classes, drop_path, global_pool)
        else:
            self.model = load_mae_pretrained_model_linprob(ckpt_path, arch, img_size,
                                                           num_classes, drop_path, global_pool)


    def forward(self, img):
        return self.model(img)


