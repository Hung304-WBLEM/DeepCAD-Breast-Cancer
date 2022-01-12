import argparse
import torch.nn as nn
import torch

from dino.utils import load_pretrained_weights, init_distributed_mode, bool_flag
from dino import vision_transformer as vits


def load_dino_pretrained_model(ckpt_path, arch='vit_small', patch_size=16):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    if arch in ['vit_small', 'vit_tiny']:
        n_last_blocks = 4
        avgpool_patchtokens = False
    elif arch == 'vit_base':
        n_last_blocks = 1
        avgpool_patchtokens = True

    embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))

    load_pretrained_weights(model, ckpt_path, 'teacher', arch, 16)

    return model, embed_dim, n_last_blocks, avgpool_patchtokens


class ViT_DINO(nn.Module):
    def __init__(self, ckpt_path, arch, patch_size, num_labels):
        super(ViT_DINO, self).__init__()

        self.model, embed_dim, self.n_last_blocks, self.avgpool_patchtokens = load_dino_pretrained_model(ckpt_path, arch, patch_size)

        self.linear = nn.Linear(embed_dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, img):
        intermediate_output = self.model.get_intermediate_layers(img, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

        if self.avgpool_patchtokens:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)

        output = output.view(output.size(0), -1)

        return self.linear(output)


if __name__ == '__main__':
    ckpt_path = '/home/hqvo2/Projects/Breast_Cancer/libs/dino/save_dir/checkpoint.pth'
    
    # model = load_dino_pretrained_model(ckpt_path)
    model = ViT_DINO(ckpt_path, 'vit_small', 16, 10)

    print(model)
