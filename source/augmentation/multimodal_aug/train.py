import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
import os

from torchvision import datasets, transforms
from torch import nn, optim
from model import Generator, Discriminator


def show_score_bars(all_classes_prob, classes):
    labels = ['classes']
    x = np.arange(len(labels))
    width = 0.05

    num_classes = len(classes)

    for idx, prob in enumerate(all_classes_prob.tolist()):
        class_score = [prob]
        rect = plt.bar(idx*width + width/2, class_score, width, label=classes[idx])

        plt.yticks([0, 0.25, 0.5, 0.75, 1])

    plt.legend(prop={'size': 5})


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    image_size = 64
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])


    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ngpu = 2
    netG = Generator(modals_num=2, input_vector_dim=10,
                     latent_size=300, ngpu=ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))


    netG.apply(weights_init)


    netD = Discriminator(ngpu, input_vector_dim=10, nc=3, ndf=64).to(device)
    
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    netD.apply(weights_init)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    ####################### FIXED ##################################
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    nz = 100
    fixed_noise = torch.randn(64, nz, device=device)

    dataiter = iter(valloader)
    fixed_images, fixed_labels = dataiter.next()
    fixed_labels = fixed_labels.unsqueeze(-1)
    fixed_one_hot_labels = torch.FloatTensor(fixed_labels.shape[0], 10)
    fixed_one_hot_labels.zero_()
    fixed_one_hot_labels.scatter_(1, fixed_labels, 1)
    ####################### FIXED ##################################

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    lr = 0.0002
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    vec_list = []
    G_losses = []
    D_losses = []
    iters = 0

    num_epochs = 100

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(trainloader, 0):
            images, img_labels = data
            img_labels = img_labels.unsqueeze(-1)
            one_hot_labels = torch.FloatTensor(img_labels.shape[0], 10)
            one_hot_labels.zero_()
            one_hot_labels.scatter_(1, img_labels, 1)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            if one_hot_labels.shape[1] == 9:
                print(one_hot_labels)
            output = netD(real_cpu, one_hot_labels, training=True).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward(retain_graph=True)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            noise = torch.randn(b_size, nz, device=device)
            # Generate fake image batch with G
            fake_img, fake_feat_vec = netG(real_cpu, one_hot_labels, noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake_img, fake_feat_vec, training=True).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_img, fake_feat_vec, training=True).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward(retain_graph=True)
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 5000 == 0) or ((epoch == num_epochs-1) and (i == len(trainloader)-1)):
                with torch.no_grad():
                    val_fake_imgs, val_fake_vecs = netG(fixed_images, fixed_one_hot_labels, fixed_noise)
                    val_fake_imgs = val_fake_imgs.detach().cpu()
                    val_fake_vecs = val_fake_vecs.detach().cpu()

                # for img, vec in zip(val_fake_imgs, val_fake_vecs):
                #     img = transforms.ToPILImage()(img).convert("RGB")
                #     img.save('test.jpg')
                #     print(vec.shape)
                    
                #     break

                # img_list.append(vutils.make_grid(val_fake_imgs, padding=2, normalize=True))
                result_img = vutils.make_grid(val_fake_imgs, padding=2, normalize=True)
                plt.figure(figsize=(22, 22))
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(result_img,(1,2,0)))
                os.makedirs('result_dir', exist_ok=True)
                plt.savefig(os.path.join('result_dir', f'img_iter{iters}.png'))
                # img_list.append((val_fake_imgs, val_fake_vecs))
                plt.close()

                plt.figure(figsize=(22, 22))
                for k in range(64):
                    plt.subplot(8, 8, k+1)
                    show_score_bars(val_fake_vecs[k], classes=['0', '1', '2', '3', '4',
                                                            '5', '6', '7', '8', '9'])
                plt.savefig(os.path.join('result_dir', f'vec_iter{iters}.png'))
                plt.close
                    

            iters += 1


