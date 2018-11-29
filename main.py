import os
import os.path as osp
import argparse
import numpy as np 
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from data import AudioDataset
from model import alignment

parser = argparse.ArgumentParser(description='PyTorch Audio-Visual')

parser.add_argument('model_dir', help='output directory to save models & results')

parser.add_argument('-g', '--gpu', type=int, default=0,\
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
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor()])

    train_dataset = AudioDataset(train=True,transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)

    model_align = alignment()
    model_align.cuda()
    model_align.train(True)

    loss_fn = nn.CrossEntropyLoss()

    optimizer_align = optim.Adam(model_align.parameters(), lr = args.learning_rate)

    for epoch in range(args.epochs):
        for batch_idx, (images, sounds, labels) in enumerate(train_loader):
            images_v = Variable(images.type(torch.FloatTensor)).cuda()
            sounds_v = Variable(sounds.type(torch.FloatTensor)).cuda()

            optimizer_align.zero_grad()

            aligned_res = model_align(args.batchsize, sounds_v, images_v)

            loss = loss_fn(aligned_res, labels)
            loss.backward()
            
            optimizer_align.step()

            print(images.shape)
            print(sounds.shape)
            print(labels.shape)
        break


def test(args):
    test_dataset = AudioDataset(train=False,transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=args.batchsize, 
                                           shuffle=False, num_workers=4)

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	if(args.is_train == 1): 
		train(args)

	# test(args)
