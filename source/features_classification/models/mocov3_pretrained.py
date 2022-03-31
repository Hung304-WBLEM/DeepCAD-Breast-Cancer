import torch
import torch.nn as nn
import importlib
mocov3_vits = importlib.import_module('moco-v3.vits')


def load_mocov3_pretrained_model(ckpt_path, arch, img_size, num_classes):
    model = mocov3_vits.__dict__[arch](num_classes=num_classes, img_size=img_size)
    linear_keyword = 'head'

    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    return model


class ViT_Mocov3(nn.Module):
    def __init__(self, ckpt_path, arch, img_size, num_classes):
        super(ViT_Mocov3, self).__init__()

        self.model = load_mocov3_pretrained_model(ckpt_path, arch, img_size, num_classes)

    def forward(self, img):
        return self.model(img)
        

