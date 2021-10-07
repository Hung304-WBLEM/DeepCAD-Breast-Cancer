import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class GeneratorEncoder(nn.Module):
    def __init__(self, input_vector_dim, use_pretrained=True):
        super(GeneratorEncoder, self).__init__()

        self.cnn = models.resnet50(pretrained=use_pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        self.img_emb_proj = nn.Linear(2048, 100)
        self.vec_emb_proj = nn.Linear(input_vector_dim, 100)

        self.fc1 = nn.Linear(200, 200)

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
    def __init__(self, ngpu, latent_size, ngf=64, nc=3):
        super(GeneratorImageDecoder, self).__init__()
        self.ngpu = ngpu
        self.latent_size = latent_size
        self.ngf = ngf

        self.decode = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_size, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, latent_vec):
        latent_vec = latent_vec.unsqueeze(-1)
        latent_vec = latent_vec.unsqueeze(-1)
        return self.decode(latent_vec)


class GeneratorVectorDecoder(nn.Module):
    def __init__(self, ngpu, latent_size, output_dim):
        super(GeneratorVectorDecoder, self).__init__()
        self.ngpu = ngpu
        self.latent_size = latent_size
        self.output_dim = output_dim

        self.decode = nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.ReLU(True),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )


    def forward(self, latent_vec):
        return self.decode(latent_vec)
        

        
class Generator(nn.Module):
    def __init__(self, modals_num, input_vector_dim, latent_size, ngpu, use_pretrained=True):
        '''
        Args: 
        modals_num(int) - numbers of modalities
        input_vector_dim(int) - dimension of the input clinical vector
        '''
        super(Generator, self).__init__()
        self.modals_num = modals_num
        self.input_vector_dim = input_vector_dim
        self.latent_size = latent_size
        self.ngpu = ngpu

        self.gen_enc = GeneratorEncoder(self.input_vector_dim, use_pretrained)
        self.gen_conv_dec = GeneratorImageDecoder(self.ngpu, self.latent_size)
        self.gen_fc_dec = GeneratorVectorDecoder(self.ngpu, self.latent_size,
                                             output_dim=self.input_vector_dim)
        
        # self.gen_dec_list = nn.ModuleList()
        # for _ in range(self.modals_num):
        #     self.gen_dec_list.append(GeneratorDecoder(ngpu))


    def forward(self, image, clinic_vec, noise_vec):
        latent_vec = self.gen_enc(image, clinic_vec, noise_vec)

        aug_image = self.gen_conv_dec(latent_vec)
        aug_clinic_vec = self.gen_fc_dec(latent_vec)

        return aug_image, aug_clinic_vec


class Discriminator(nn.Module):
    def __init__(self, ngpu, input_vector_dim, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*8) x 4 x 4
            # nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            nn.AvgPool2d(4)
        )
        self.img_emb_proj = nn.Linear(512, 100)
        self.vec_emb_proj = nn.Linear(input_vector_dim, 100)

        self.fc1 = nn.Linear(200, 200)
        num_classes = 1
        self.fc2 = nn.Linear(200, num_classes)


    def forward(self, image, clinic_vec, training):
        x1 = self.main(image)
        x2 = clinic_vec


        x1 = x1.squeeze() # for resnet50

        x1 = F.relu(self.img_emb_proj(x1))
        x2 = F.relu(self.vec_emb_proj(x2))

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=training)
        x = F.sigmoid(self.fc2(x))

        return x
