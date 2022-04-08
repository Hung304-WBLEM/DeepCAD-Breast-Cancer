import torch
import torch.nn as nn

from timm.models.swin_transformer import swin_base_patch4_window7_224
from SimMIM.models.swin_transformer import SwinTransformer
from SimMIM.utils import remap_pretrained_keys_swin, remap_pretrained_keys_vit


def load_simmim_pretrained_model(model_type, ckpt_path, img_size, num_classes):
    # Create model arch
    model = swin_base_patch4_window7_224(img_size=img_size, num_classes=num_classes)

    # Load Pretrained Ckpt
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model = checkpoint['model']

    checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}

    if model_type == 'swin':
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)
    elif model_type == 'vit':
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    del checkpoint
    torch.cuda.empty_cache()

    return model

class Swin_SimMIM(nn.Module):
    def __init__(self, ckpt_path, arch, img_size, num_classes):
        super(Swin_SimMIM, self).__init__()

        self.model = load_simmim_pretrained_model('swin', ckpt_path, img_size, num_classes)

    def forward(self, img):
        return self.model(img)
