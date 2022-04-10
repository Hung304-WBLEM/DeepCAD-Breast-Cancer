import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision import models
from features_classification.models.clinical_models.clinical_models import Clinical_Model


class Clinical_Concat_Model(nn.Module):
    def __init__(self, model_name, input_vector_dim, num_classes, use_pretrained=True):
        super(Clinical_Concat_Model, self).__init__()
        self.model_name = model_name
        if model_name == "fusion_resnet50":
            self.cnn = models.resnet50(pretrained=use_pretrained)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            # self.fc1 = nn.Linear(2048 + input_vector_dim, 512)
            # self.fc2 = nn.Linear(512, num_classes)

            #############################################################
            # new code from here (will optimize later)
            self.img_emb_proj = nn.Linear(2048, 100)
            self.vec_emb_proj = nn.Linear(input_vector_dim, 100)

            self.fc1 = nn.Linear(200, 200)
            self.fc2 = nn.Linear(200, num_classes)
            #############################################################

            self.dropout_layer = nn.Dropout(p=0.5)

        elif model_name == 'fusion_vgg16':
            self.cnn = models.vgg16_bn(pretrained=use_pretrained)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-3])

            self.fc1 = nn.Linear(self.cnn.classifier[3].out_features + input_vector_dim, 512)
            self.fc2 = nn.Linear(512, num_classes)


    def forward(self, image, vector_data):
        x1 = self.cnn(image)
        x2 = vector_data.float()

        if self.model_name == 'fusion_resnet50':
            x1 = x1.squeeze()

        ######################################################
        # new code from here (will optimize later)
        x1 = F.relu(self.img_emb_proj(x1))
        x2 = F.relu(self.vec_emb_proj(x2))
        ######################################################

        if len(x1.shape) == 1:
            x1 = torch.unsqueeze(x1, 0)

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))

        # if isinstance(training, torch.Tensor):
        #     x = F.dropout(x, p=0.5, training=training.item())
        # else:
        #     x = F.dropout(x, p=0.5, training=training)
        x = self.dropout_layer(x)
        x = self.fc2(x)

        return x


class Clinical_Attentive_Model(nn.Module):
    def __init__(self, model_name, input_vector_dim, num_classes, attention_type='crossatt', use_pretrained=True):
        '''
        Parameters:
        attention_type - select type of attention. Available options includes: 
                         co-attention, cross-attention
        '''

        super(Clinical_Attentive_Model, self).__init__()
        self.model_name = model_name
        self.attention_type = attention_type

        if model_name == 'resnet50':
            self.cnn = models.resnet50(pretrained=use_pretrained)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            self.img_emb_proj = nn.Linear(2048, 100)
            self.vec_emb_proj = nn.Linear(input_vector_dim, 100)

            if self.attention_type == 'coatt':
                self.img_emb_att = nn.Linear(2048 + input_vector_dim, 100)
                self.vec_emb_att = nn.Linear(2048 + input_vector_dim, 100)
            elif self.attention_type == 'crossatt':
                self.img_emb_att = nn.Linear(input_vector_dim, 100)
                self.vec_emb_att = nn.Linear(2048, 100)

            self.fc1 = nn.Linear(200, 200)
            self.fc1_att = nn.Linear(200, 200)

            self.fc2 = nn.Linear(200, num_classes)
            self.fc2_att = nn.Linear(200, num_classes)


    def forward(self, image, vector_data, training):
        img_emb = self.cnn(image)
        if self.model_name == 'resnet50':
            img_emb = img_emb.squeeze()
        vec_emb = vector_data

        proj_img_emb = F.relu(self.img_emb_proj(img_emb))
        proj_vec_emb = F.relu(self.vec_emb_proj(vec_emb))

        if self.attention_type == 'coatt':
            alpha_img = torch.sigmoid(self.img_emb_att(torch.cat((img_emb, vec_emb), dim=1)))
            alpha_vec = torch.sigmoid(self.vec_emb_att(torch.cat((img_emb, vec_emb), dim=1)))
        elif self.attention_type == 'crossatt':
            alpha_img = torch.sigmoid(self.img_emb_att(vec_emb))
            alpha_vec = torch.sigmoid(self.vec_emb_att(img_emb))
            

        aug_img_emb = torch.mul(proj_img_emb, alpha_img)
        aug_vec_emb = torch.mul(proj_vec_emb, alpha_vec)

        concat_emb = torch.cat((aug_img_emb, aug_vec_emb), dim=1)

        x = F.relu(self.fc1(concat_emb))
        alpha_x = torch.sigmoid(self.fc1_att(concat_emb))
        aug_x = torch.mul(x, alpha_x)

        aug_x = F.dropout(aug_x, p=0.5, training=training)

        x = self.fc2(aug_x)
        alpha_x = torch.sigmoid(self.fc2_att(aug_x))
        aug_x = torch.mul(x, alpha_x)

        x = aug_x

        return x


class Clinical_Parallel_Model(nn.Module):
    def __init__(self, model_name, input_vector_dim, num_classes, use_pretrained=True):
        super(Clinical_Parallel_Model, self).__init__()

        self.model_name = model_name
        if model_name == "fusion_parallel_resnet50":
            self.img_model = models.resnet50(pretrained=use_pretrained)
            self.img_model = nn.Sequential(*list(self.img_model.children())[:-1])
            self.img_fc1 = nn.Linear(2048, 100)
            self.img_fc2 = nn.Linear(100, num_classes)
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.5)

            self.vec_model = Clinical_Model(model_name,
                                            input_vector_dim, num_classes)


    def forward(self, image, vector_data):
        x = self.img_model(image)
        if self.model_name == 'fusion_parallel_resnet50':
            x = x.squeeze()

        x = self.dropout1(x)
        img_emb = self.img_fc1(F.relu(x))
        x = self.dropout2(F.relu(img_emb))
        img_logits = self.img_fc2(x)

        vec_emb, vec_logits = self.vec_model(vector_data, get_feats=True)

        return img_emb, img_logits, vec_emb, vec_logits 

