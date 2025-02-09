"""Train Flow++ on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Independent, Normal

import util
from vae_n_d_n_l import VAE
from models import FlowPlusPlus
from tqdm import tqdm


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    print(device)
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since model expects inputs in (0, 1)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model vae
    vae_net = VAE('cifar')
    vae_net.init_model()
    vae_net.load_state_dict(torch.load(args.vae_model_path))
    vae_net.eval()

    # Model flow++
    print('Building model..')
    net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                       in_shape=(3, 32, 32),
                       mid_channels=args.num_channels,
                       num_blocks=args.num_blocks,
                       num_dequant_blocks=args.num_dequant_blocks,
                       num_components=args.num_components,
                       use_attn=args.use_attn,
                       drop_prob=args.drop_prob)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at save/best.pth.tar...')
        assert os.path.isdir('save'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('save/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global global_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)
    warm_up = args.warm_up * args.batch_size
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / warm_up))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, vae_net,  trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm)
        test(epoch, net, vae_net, testloader, device, loss_fn, args.num_samples, args.save_dir)


@torch.enable_grad()
def train(epoch, net, vae_net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()

            # vae model both n
            mu_d, logvar_d, mu, logvar = vae_net(x)

            z, sldj = net(x, reverse=False)

            # loss = loss_fn(z, sldj)
            loss = loss_fn(z, sldj, mu_d, logvar_d)

            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)


@torch.no_grad()
def sample(net, vae_net, batch_size, device):
    # assume latent features space ~ N(0, 1)
    z = torch.randn(batch_size, vae_net.n_latent_features).to(device)
    z = vae_net.fc4(z)
    z = vae_net.decoder(z)
    z = z.view(-1, vae_net.n_neurons_last_decoder_layer)

    """ both normal """
    mu_d, logvar_d = vae_net.decoder_bottleneck(z)
    dist = Independent(Normal(loc=mu_d, scale=torch.exp(logvar_d)), 1)

    # sample from model
    z = dist.sample()
    z = z.view(-1, 3, 32, 32)

    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)
    return x


@torch.no_grad()
def test(epoch, net, vae_net, testloader, device, loss_fn, num_samples, save_dir):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)

            # vae model
            mu_d, logvar_d, mu, logvar = vae_net(x)

            z, sldj = net(x, reverse=False)
            # loss = loss_fn(z, sldj)
            loss = loss_fn(z, sldj, mu_d, logvar_d)

            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

        x_, _ = net(x, reverse=True)
        reconstruction_images = torch.sigmoid(x_)

    # Save checkpoint
    print('best_loss ', best_loss)
    print('loss_meter.avg  ', loss_meter.avg)
    if loss_meter.avg < best_loss:
        best_loss = loss_meter.avg

    print('Saving...')
    state = {
        'net': net.state_dict(),
        'test_loss': loss_meter.avg,
        'epoch': epoch,
    }
    os.makedirs('ckpts', exist_ok=True)
    torch.save(state, 'ckpts/flow++_' + str(epoch) + 'vae(n_d_n_l_64).pth.tar')

    # Save reconstruction images
    images = reconstruction_images
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/reconstruction_epoch_{}.png'.format(epoch))

    # Save samples and data
    images = sample(net, vae_net, num_samples, device)
    os.makedirs(save_dir, exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat,
                                 os.path.join(save_dir, 'epoch_{}.png'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow++ on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Peak learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--num_blocks', default=5, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_dequant_blocks', default=2, type=int, help='Number of blocks in dequantization')
    parser.add_argument('--num_channels', default=96, type=int, help='Number of channels in Flow++')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='samples', help='Directory for saving samples')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    parser.add_argument('--vae_model_path', default="vae_ckpts/vae_n_decoder_n_latent_cifar_model_0501.pt",
                       type=str, help='')
    parser.add_argument('--vae_optim_path', default="vae_n_decoder_n_latent_cifar_optim_0501.pt",
                        type=str, help='')

    best_loss = 0
    global_step = 0

    print(torch.__version__)
    main(parser.parse_args())
