import numpy as np
import torch
import random
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
import os
import logging
import copy

from config_multimodal_aug_org import options
from torchvision import datasets, transforms
from torch import nn, optim
from model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from matplotlib import gridspec

torch.autograd.set_detect_anomaly(True)

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def show_score_bars(ax, all_classes_prob, classes, ignore_label=True):
    labels = ['classes']
    x = np.arange(len(labels))
    width = 0.05

    num_classes = len(classes)

    for idx, prob in enumerate(all_classes_prob.tolist()):
        class_score = [prob]
        if ignore_label:
            rect = ax.bar(idx*width + width/2, class_score, width)
        else:
            rect = ax.bar(idx*width + width/2, class_score, width, label=classes[idx])

        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([])

    # ax.legend(prop={'size': 5})


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty( critic, real_image, real_vector, fake_image, fake_vector, device="cpu"):
    batch_size, channel, height, width= real_image.shape
    batch_size, dim = real_vector.shape
    #alpha is selected randomly between 0 and 1
    alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width).to(device)
    beta = torch.rand(batch_size,1).repeat(1, dim).to(device)
    
    # interpolated image=randomly weighted average between a real and fake image
    #interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolated_image=(alpha*real_image) + (1-alpha) * fake_image
    interpolated_vector = (beta*real_vector) + (1-beta) * fake_vector
    
    # calculate the critic score on the interpolated image
    # interpolated_score= critic(interpolatted_image)
    interpolated_score = critic(interpolated_image,
                                interpolated_vector, training=True)
    
    # take the gradient of the score wrt to the interpolated image
    gradient= torch.autograd.grad(inputs=[interpolated_image, interpolated_vector],
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score)                          
                                 )[0]
    gradient= gradient.view(gradient.shape[0],-1)
    gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)
    # gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty


if __name__ == '__main__':
    set_seed() 


    # For Logging
    logging.basicConfig(filename=os.path.join(options.save_path, 'train.log'), level=logging.INFO,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    writer = SummaryWriter(os.path.join(options.save_path, 'tensorboard_logs'))

    image_size = options.image_size
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])


    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=options.batch_size,
                                              shuffle=True,
                                              worker_init_fn=np.random.seed(42),
                                              num_workers=options.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=options.batch_size,
                                            shuffle=False,
                                            worker_init_fn=np.random.seed(42),
                                            num_workers=options.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ngpu = options.ngpu
    netG = Generator(modals_num=2, input_vector_dim=10
                     ).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))


    netG.apply(weights_init)


    if options.loss_func == 'minmax':
        netD = Discriminator(input_vector_dim=10, fuse_type=options.disc_fuse_type).to(device)
    elif options.loss_func == 'wasserstein':
        netD = Discriminator(input_vector_dim=10, keep_sigmoid=False, fuse_type=options.disc_fuse_type).to(device)
        
    
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    netD.apply(weights_init)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    ############################### Fixed Data for visualization ##################################
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    nz = options.latent_size
    fixed_noise = torch.randn(options.batch_size, nz, device=device)

    dataiter = iter(valloader)
    fixed_images, fixed_labels = dataiter.next()
    fixed_images = fixed_images.to(device)
    fixed_labels = fixed_labels.unsqueeze(-1)
    fixed_one_hot_labels = torch.FloatTensor(fixed_labels.shape[0], 10)
    fixed_one_hot_labels.zero_()
    fixed_one_hot_labels.scatter_(1, fixed_labels, 1)
    fixed_one_hot_labels[fixed_one_hot_labels == 0] = -1
    fixed_one_hot_labels = fixed_one_hot_labels.to(device)

    ############################### Fixed data for visualization ##################################

    # Establish convention for real and fake labels during training
    if not options.random_lbl_smooth:
        real_label = 1.0
        fake_label = 0.0

    # Setup Adam optimizers for both G and D
    d_lr = options.disc_lr
    g_lr = options.gen_lr
    beta1 = options.beta1
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(beta1, 0.999))
    print(optimizerD, optimizerG)


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    vec_list = []
    G_losses = []
    D_losses = []
    iters = 0

    num_epochs = options.epochs

    print("Starting Training Loop...")
    logging.info("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(trainloader, 0):
            images, img_labels = data
            images.to(device)
            img_labels = img_labels.unsqueeze(-1)
            one_hot_labels = torch.FloatTensor(img_labels.shape[0], 10)
            one_hot_labels.zero_()
            one_hot_labels.scatter_(1, img_labels, 1)
            one_hot_labels[one_hot_labels == 0] = -1
            one_hot_labels = one_hot_labels.to(device)

            for it in range(options.d_num_iters + 1):
                netD.zero_grad()

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                if not options.random_lbl_smooth:
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                else:
                    label = torch.FloatTensor(b_size, 1).uniform_(options.real_label_min,
                                                                options.real_label_max).to(device)

                # Forward pass real batch through D
                output = netD(real_cpu, one_hot_labels, training=True).view(-1)

                # Calculate loss on all-real batch
                if options.loss_func == 'minmax':
                    errD_real = criterion(output, label)
                elif options.loss_func == 'wasserstein':
                    errD_real = -torch.mean(output)

                if options.loss_func == 'minmax':
                    # Calculate gradients for D in backward pass
                    errD_real.backward(retain_graph=True)

                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                # noise = torch.randn(b_size, nz, 1, 1, device=device)
                noise = torch.randn(b_size, nz, device=device)
                # Generate fake image batch with G
                fake_img, fake_feat_vec = netG(real_cpu, one_hot_labels, noise)

                # label.fill_(fake_label)

                if not options.random_lbl_smooth:
                    label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
                else:
                    label = torch.FloatTensor(b_size, 1).uniform_(options.fake_label_min,
                                                                options.fake_label_max).to(device)

                # Classify all fake batch with D
                output = netD(fake_img, fake_feat_vec, training=True).view(-1)

                # Calculate D's loss on the all-fake batch
                if options.loss_func == 'minmax':
                    errD_fake = criterion(output, label)
                elif options.loss_func == 'wasserstein':
                    errD_fake = torch.mean(output)

                if options.loss_func == 'minmax':
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward(retain_graph=True)

                D_G_z1 = output.mean().item()


                if options.loss_func == 'wasserstein':
                    gp = gradient_penalty(netD, real_cpu, one_hot_labels,
                                        fake_img, fake_feat_vec, device)
                    err_R = 10*gp


                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake + err_R
                # errD = errD_real + errD_fake

                if it == 0:
                    errD.backward()
                else:
                    errD.backward(create_graph=True)

                # Update D
                optimizerD.step()

                # Clamp data for Wasserstein Loss
                # for p in netD.parameters():
                #     p.data.clamp_(-0.01, 0.01)

                if it == 1:
                    backup = copy.deepcopy(netD)


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            # label.fill_(real_label)  # fake labels are real for generator cost
            if not options.random_lbl_smooth:
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            else:
                label = torch.FloatTensor(b_size, 1).uniform_(options.real_label_min,
                                                              options.real_label_max).to(device)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_img, fake_feat_vec, training=True).view(-1)

            # Calculate G's loss based on this output
            if options.loss_func == 'minmax':
                errG = criterion(output, label)
            elif options.loss_func == 'wasserstein':
                errG = -torch.mean(output)

            # Calculate gradients for G
            errG.backward(retain_graph=True)
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            netD.load(backup)    
            del backup

            writer.add_scalar('Loss_D', errD.item(), iters)
            writer.add_scalar('Loss_G', errG.item(), iters)
            writer.add_scalar('D(x)', D_x, iters)
            writer.add_scalar('D(G(z))', D_G_z1, iters)
            writer.add_scalar('D(G(z))', D_G_z2, iters)


            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(trainloader)-1)):
                with torch.no_grad():
                    val_fake_imgs, val_fake_vecs = netG(fixed_images, fixed_one_hot_labels, fixed_noise)
                    val_fake_imgs = val_fake_imgs.detach().cpu()
                    val_fake_vecs = val_fake_vecs.detach().cpu()


                # Plot images
                width_ratios = [el for _ in range(6) for el in [2, 1]]
                fig, a = plt.subplots(6, 12, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': width_ratios,
                                                 'height_ratios': [1 for _ in range(6)]})
                fig.tight_layout()

                for k in range(36):
                    # plt.subplot(6, 12, 2*k+1)
                    img = transforms.ToPILImage()(val_fake_imgs[k]).convert("RGB")
                    # plt.axis('off')
                    r = (2*k)//12
                    c = (2*k)%12
                    a[r, c].axis('off')
                    # plt.imshow(img)
                    a[r, c].imshow(img)

                    show_score_bars(a[(2*k+1)//12, (2*k+1)%12],
                                    val_fake_vecs[k],
                                    classes=['0', '1', '2', '3', '4',
                                             '5', '6', '7', '8', '9'],
                                    ignore_label=(k!=0))
                        

                fig.legend(loc="upper center", ncol=10)


                writer.add_figure(f'Gan Visualization Results',
                                  fig,
                                  global_step=iters)
                os.makedirs(os.path.join(options.save_path, 'visualize'), exist_ok=True)
                plt.savefig(os.path.join(options.save_path, 'visualize', f'vis_iter{iters}.png'))
                plt.close()

                # checkpoint
                os.makedirs(os.path.join(options.save_path, 'ckpts'), exist_ok=True)
                torch.save(netD, os.path.join(options.save_path, 'ckpts', f'netD_ckpt_iter{iters}.pth'))
                torch.save(netG, os.path.join(options.save_path, 'ckpts', f'netG_ckpt_iter{iters}.pth'))
                    

            iters += 1


