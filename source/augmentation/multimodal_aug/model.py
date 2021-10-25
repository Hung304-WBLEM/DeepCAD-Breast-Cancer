import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class GeneratorEncoder(nn.Module):
    def __init__(self, input_vector_dim, use_pretrained=True,
                 last_frozen_layer=158,
                 img_emb_dim=100, vec_emb_dim=100,
                 final_emb_dim=200):
        super(GeneratorEncoder, self).__init__()

        self.cnn = models.resnet50(pretrained=use_pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        for idx, (name, param) in enumerate(self.cnn.named_parameters()):
            param.requires_grad = False

        for idx, (name, param) in enumerate(self.cnn.named_parameters()):
            if idx <= last_frozen_layer:
                continue
            param.requires_grad = True

        self.img_emb_proj = nn.Linear(2048, img_emb_dim)
        self.vec_emb_proj = nn.Linear(input_vector_dim, vec_emb_dim)

        self.fc1 = nn.Linear(img_emb_dim+vec_emb_dim, final_emb_dim)

    def forward(self, image, clinic_vec, noise_vec):
        x1 = self.cnn(image)
        x2 = clinic_vec
        x3 = noise_vec

        x1 = x1.squeeze() # for resnet50

        x1 = F.relu(self.img_emb_proj(x1))
        x2 = F.relu(self.vec_emb_proj(x2))

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))

        x = torch.cat((x, x3), dim=1)

        return x

    
class GeneratorImageDecoder(nn.Module):
    def __init__(self, latent_size, ngf=64, nc=3):
        super(GeneratorImageDecoder, self).__init__()
        self.latent_size = latent_size
        self.ngf = ngf

        self.img_size = (32, 32)
        self.feature_sizes = (self.img_size[0] // 16, self.img_size[1] // 16)
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_size, 8 * ngf * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * ngf),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * ngf),
            nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ngf),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )


        # self.decode = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(self.latent_size, self.ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(self.ngf * 8),
        #     # nn.ReLU(True),
        #     nn.LeakyReLU(0.2, True),

        #     # state size. (self.ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ngf * 4),
        #     # nn.ReLU(True),
        #     nn.LeakyReLU(0.2, True),

        #     # state size. (self.ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ngf * 2),
        #     # nn.ReLU(True),
        #     nn.LeakyReLU(0.2, True),

        #     # state size. (self.ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ngf),
        #     # nn.ReLU(True),
        #     nn.LeakyReLU(0.2, True),

        #     # state size. (self.ngf) x 32 x 32
        #     nn.ConvTranspose2d(self.ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64
        # )

    def forward(self, latent_vec):
        # latent_vec = latent_vec.unsqueeze(-1)
        # latent_vec = latent_vec.unsqueeze(-1)
        # return self.decode(latent_vec)

        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(latent_vec)
        # Reshape
        x = x.view(-1, 8 * self.ngf, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.decode(x)


class GeneratorVectorDecoder(nn.Module):
    def __init__(self, latent_size, output_dim):
        super(GeneratorVectorDecoder, self).__init__()
        self.latent_size = latent_size
        self.output_dim = output_dim

        self.decode = nn.Sequential(
            nn.Linear(self.latent_size, 4096),
            nn.BatchNorm1d(4096),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, output_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, latent_vec):
        return self.decode(latent_vec)


class Generator(nn.Module):
    def __init__(self, modals_num, input_vector_dim, ngf=64, latent_size=300, use_pretrained=True):
        '''
        Args: 
        modals_num(int) - numbers of modalities
        input_vector_dim(int) - dimension of the input clinical vector
        '''
        super(Generator, self).__init__()
        self.modals_num = modals_num
        self.input_vector_dim = input_vector_dim
        self.ngf = ngf
        self.latent_size = latent_size

        self.gen_enc = GeneratorEncoder(self.input_vector_dim, use_pretrained)
        self.gen_conv_dec = GeneratorImageDecoder(self.latent_size, self.ngf)
        self.gen_fc_dec = GeneratorVectorDecoder(self.latent_size,
                                             output_dim=self.input_vector_dim)
        
        # self.gen_dec_list = nn.ModuleList()
        # for _ in range(self.modals_num):
        #     self.gen_dec_list.append(GeneratorDecoder())


    def forward(self, image, clinic_vec, noise_vec):
        latent_vec = self.gen_enc(image, clinic_vec, noise_vec)

        aug_image = self.gen_conv_dec(latent_vec)
        aug_clinic_vec = self.gen_fc_dec(latent_vec)

        return aug_image, aug_clinic_vec


class Discriminator(nn.Module):
    def __init__(self, input_vector_dim, ndf=64, nc=3, keep_sigmoid=True, fuse_type='concat'):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.keep_sigmoid = keep_sigmoid
        self.fuse_type = fuse_type

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (self.ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (self.ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (self.ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (self.ndf*8) x 4 x 4
            # nn.Conv2d(self.ndf * 8, self.ndf * 8, 2, 1, 0, bias=False), # for 64x64
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 2, 1, 0, bias=False), # for 32x32
            # nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.img_emb_proj = nn.Linear(512, 100)
        self.vec_emb_proj = nn.Linear(input_vector_dim, 100)

        if self.fuse_type == 'coatt':
            self.img_emb_att = nn.Linear(512 + input_vector_dim, 100)
            self.vec_emb_att = nn.Linear(512 + input_vector_dim, 100)
        elif self.fuse_type == 'crossatt':
            self.img_emb_att = nn.Linear(input_vector_dim, 100)
            self.vec_emb_att = nn.Linear(512, 100)

        self.fc1 = nn.Linear(200, 200)
        self.fc1_att = nn.Linear(200, 200)

        num_classes = 1
        self.fc2 = nn.Linear(200, num_classes)
        self.fc2_att = nn.Linear(200, num_classes)


    def forward(self, image, clinic_vec, training):
        img_emb = self.main(image)
        vec_emb = clinic_vec


        img_emb = img_emb.squeeze() # for resnet50


        proj_img_emb = F.relu(self.img_emb_proj(img_emb))
        proj_vec_emb = F.relu(self.vec_emb_proj(vec_emb))

        if self.fuse_type == 'coatt':
            alpha_img = torch.sigmoid(self.img_emb_att(torch.cat((img_emb, vec_emb), dim=1)))
            alpha_vec = torch.sigmoid(self.vec_emb_att(torch.cat((img_emb, vec_emb), dim=1)))
        elif self.fuse_type == 'crossatt':
            alpha_img = torch.sigmoid(self.img_emb_att(vec_emb))
            alpha_vec = torch.sigmoid(self.vec_emb_att(img_emb))


        if self.fuse_type in ['coatt', 'crossatt']:
            aug_img_emb = torch.mul(proj_img_emb, alpha_img)
            aug_vec_emb = torch.mul(proj_vec_emb, alpha_vec)
            concat_emb = torch.cat((aug_img_emb, aug_vec_emb), dim=1)
        elif self.fuse_type == 'concat':
            concat_emb = torch.cat((proj_img_emb, proj_vec_emb), dim=1)

        if self.fuse_type in ['coatt', 'crossatt']:
            x = F.relu(self.fc1(concat_emb))
            alpha_x = torch.sigmoid(self.fc1_att(concat_emb))
            aug_x = torch.mul(x, alpha_x)
            aug_x = F.dropout(aug_x, p=0.5, training=training)

            x = self.fc2(aug_x)
            alpha_x = torch.sigmoid(self.fc2_att(aug_x))
            x = torch.mul(x, alpha_x)
        elif self.fuse_type == 'concat':
            x = F.relu(self.fc1(concat_emb))
            x = F.dropout(x, p=0.5, training=training)
            x = self.fc2(x)

        if self.keep_sigmoid:
            x = F.sigmoid(x)

        return x

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
