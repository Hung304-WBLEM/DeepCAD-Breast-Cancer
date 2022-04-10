import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision import models


class Clinical_Model(nn.Module):
    def __init__(self, model_name, input_vector_dim, num_classes):
        super(Clinical_Model, self).__init__()
        self.model_name = model_name

        self.vec_emb_proj = nn.Linear(input_vector_dim, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.dropout_layer = nn.Dropout(p=0.5)


    def forward(self, vector_data, get_feats=False):
        x = vector_data.float()

        x = F.relu(self.vec_emb_proj(x))
        feats = self.fc1(x)
        x = F.relu(feats)
        x = self.dropout_layer(x)
        logits = self.fc2(x)

        if get_feats:
            return feats, logits 
        return logits

