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

sys.path.append('data/process/')

parser = argparse.ArgumentParser(description='PyTorch Audio-Visual')

parser.add_argument('model', help='output directory to save models & results')

parser.add_argument('-g', '--gpu', type=int, default=0,\
                    help='gpu device id')

parser.add_argument('-t', '--is_train', type=int, default=1,\
                    help='use 1 to train model')

parser.add_argument('-e', '--epochs', type=int, default=500,\
                    help='number of training epochs')

parser.add_argument('-b', '--batchsize', type=int, default=16,\
                    help='number of samples per training batch')

parser.add_argument('-m', '--nthreads', type=int, default=4,\
                    help='pytorch data loader threads')

parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5,\
                    help='learning rate')

parser.add_argument('-vf', '--val_freq', type=float, default=25,\
                    help='number of epochs before testing validation set')

args = parser.parse_args()


def train(epoch, train_loader, optimizer_align, model_align, loss_fn):
    accs = []
    losses = []
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


def test(epoch, test_loader, model_align, loss_fn, repeat = 5):
    accs = []
    losses = []
    for i in range(repeat):
        for batch_idx, (images, sounds, labels) in enumerate(test_loader):
            with torch.no_grad():
                images_v = Variable(images.type(torch.FloatTensor)).cuda()
                sounds_v = Variable(sounds.type(torch.FloatTensor)).cuda()
                labels_v = Variable(labels).cuda()

                aligned_res, _ = model_align(sounds_v, images_v)
                loss = loss_fn(aligned_res, labels_v)
                losses.append(loss.item())
                accs.append(np.mean((torch.argmax(aligned_res,1) == labels_v).detach().cpu().numpy()))
    print("Validation :", epoch, np.mean(losses), np.mean(accs))

def activation(feature_map, weights, label):
    output = np.zeros((224,224))
    for i in range(512):
        output += imresize(feature_map[i], (224,224))*weights[label,i]
    return output

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # gpu device

    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    train_dataset = AudioDataset(train=True,transform=transform, h5_file='/media/jeff/Backup/CS598PS/data.h5')
    test_dataset = AudioDataset(train=False,transform=transform, h5_file='/media/jeff/Backup/CS598PS/data.h5')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    model_align = alignment().cuda()
#     checkpoint = torch.load("fixed_500.pth")
#     model_align.load_state_dict(checkpoint.state_dict())

    loss_fn = nn.CrossEntropyLoss()
    optimizer_align = optim.Adam(model_align.parameters(), lr = args.learning_rate)
    
    if (args.is_train == 1): 
        for epoch in range(args.epochs):
            train(epoch, train_loader, optimizer_align, model_align, loss_fn)
            if (epoch + 1)%args.val_freq == 0:
                test(epoch, test_loader, model_align, loss_fn)
        torch.save(model_align, args.model + '.pth')
        
    output = activation(feature_maps[0,:,0].detach().cpu().numpy(), weight.detach().cpu().numpy(),0)


   
