import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
import h5py  
import numpy as np
import os 
from scipy.misc import imresize
import cv2
import random
import soundfile as sf
import argparse
from data import AudioDataset
from model import alignment

parser = argparse.ArgumentParser(description='PyTorch Audio-Visual')

parser.add_argument('model_dir', help='output directory to save models & results')

parser.add_argument('-g', '--gpu', type=int, default=4,\
                    help='gpu device id')

parser.add_argument('-t', '--is_train', type=int, default=1,\
                    help='use 1 to train model')

parser.add_argument('-e', '--epochs', type=int, default=1,\
                    help='number of training epochs')

parser.add_argument('-b', '--batchsize', type=int, default=5,\
                    help='number of samples per training batch')

parser.add_argument('-m', '--nthreads', type=int, default=4,\
                    help='pytorch data loader threads')

parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,\
                    help='learning rate')

parser.add_argument('-hs', '--n_hidden', type=int, default=128,\
                    help='Size of hidden state of LSTM')

args = parser.parse_args()


def train(args):
    transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor()])

    train_dataset = AudioDataset(train=True,transform=transform)
    test_dataset = AudioDataset(train=False,transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model_align = alignment()
    model_align.cuda()
    model_align.train(True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_align = optim.Adam(model_align.parameters(), lr = 1e-4)
    for epoch in range(500):
        accs = []
        losses = []
        model_align.train()
        for batch_idx, (images, sounds, labels) in enumerate(train_loader):
            images_v = Variable(images.type(torch.FloatTensor)).cuda()
            sounds_v = Variable(sounds.type(torch.FloatTensor)).cuda()
            labels_v = Variable(labels).cuda()
            
            optimizer_align.zero_grad()
            aligned_res, _ = model_align(sounds_v, images_v)
            loss = loss_fn(aligned_res, labels_v)
            loss.backward()
            optimizer_align.step()
            losses.append(loss.item())
            accs.append(np.mean((torch.argmax(aligned_res,1) == labels_v).detach().cpu().numpy()))
        print("Epoch :", epoch, np.mean(losses), np.mean(accs))
        if (epoch + 1)%25 == 0:
            accs = []
            losses = []
            model_align.eval()
            for batch_idx, (images, sounds, labels) in enumerate(test_loader):
                images_v = Variable(images.type(torch.FloatTensor)).cuda()
                sounds_v = Variable(sounds.type(torch.FloatTensor)).cuda()
                labels_v = Variable(labels).cuda()
                aligned_res, _ = model_align(sounds_v, images_v)
                loss = loss_fn(aligned_res, labels_v)
                losses.append(loss.item())
                accs.append(np.mean((torch.argmax(aligned_res,1) == labels_v).detach().cpu().numpy()))
            print("Validation :", epoch, np.mean(losses), np.mean(accs))
    torch.save(model_align, 'fixed_500.pth')


def test(args):
    test_dataset = AudioDataset(train=False,transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

def activation(feature_map, weights, label):
    output = np.zeros((224,224))
    for i in range(512):
        output += imresize(feature_map[i], (224,224))*weights[label,i]
    return output

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if(args.is_train == 1): 
		train(args)

    for name, param in model_align.state_dict().items():
        if name =='fc.weight_v':
            weight = param

    images, sounds, labels = train_dataset[1]
    images_v = Variable(torch.tensor(images)).type(torch.FloatTensor).cuda().unsqueeze(0)
    sounds_v = Variable(torch.tensor(sounds)).type(torch.FloatTensor).cuda().unsqueeze(0)
    labels_v = Variable(torch.tensor(labels)).cuda().unsqueeze(0)
    aligned_res, feature_maps = model_align(sounds_v, images_v)

    output = activation(feature_maps[0,:,0].detach().cpu().numpy(), weight.detach().cpu().numpy(),0)

    # plt.imshow(output, cmap='gray')
    # plt.show()
    # plt.imshow(images[0,0], cmap='gray')
	# test(args)

if __name__ == '__main__':
    main()

   
