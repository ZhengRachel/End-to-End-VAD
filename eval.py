from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from utils import utils as utils
from torch.utils.data import DataLoader
import time
import torch.nn.utils as torchutils
from torch.autograd import Variable
from utils.logger import Logger
from sklearn import metrics
import matplotlib.pylab as plt
import os
import numpy as np

if __name__ == '__main__':

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--time_depth', type=int, default=15, help='number of time frames in each video\audio sample')
    parser.add_argument('--workers', type=int, default=0, help='num workers for data loading')
    parser.add_argument('--print_freq', type=int, default=50, help='freq of printing stats')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--use_mcb', action='store_true', help='wether to use MCB or concat')
    parser.add_argument('--mcb_output_size', type=int, default=1024, help='the size of the MCB outputl')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    parser.add_argument('--arch', type=str, default='Video', help='which modality to train - Video\Audio\AV')
    parser.add_argument('--pre_train', type=str, default='/home/rczheng/End-to-End-VAD/saved_models/Video/acc_0.696_epoch_027_arch_Video.pkl', help='path to the pre-trained network')
    args = parser.parse_args()
    print(args, end='\n\n')

    # create test dataset
    dataset = utils.import_dataset(args)

    test_dataset = dataset(DataDir='data/test/', timeDepth = args.time_depth, is_train=False)

    # create the data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)

    # create optimizer and loss
    criterion = nn.CrossEntropyLoss().cuda()

    # import network
    net = utils.import_network(args)

    # init from a saved checkpoint
    if args.pre_train is not '':
        model_name = os.path.join('pre_trained',args.arch,args.pre_train)

        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(args.pre_train))
            checkpoint = torch.load(args.pre_train)
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre_train, checkpoint['epoch']))
        else:
            print('Couldn\'t load model from {}'.format(model_name))
    else:
        print('Training the model from scratch.')


    # perform test
    test_acc = utils.AverageMeter()
    test_loss = utils.AverageMeter()

    net.eval()
    print('Test started.')

    all_pred = []
    all_gt = []
    all_pre_index = []

    for i, data in enumerate(test_loader):

        states_test = net.init_hidden(is_train=False)

        if args.arch == 'Video' or args.arch == 'Audio':  # single modality

            input, target = data  # input is of shape torch.Size([batch, channels, frames, width, height])
            input_var = Variable(input.unsqueeze(1)).cuda()
            target_var = Variable(target.squeeze()).cuda()

            output = net(input_var, states_test)

        else:  # multiple modalities

            audio, video, target = data
            audio_var = Variable(audio.unsqueeze(1)).cuda()
            video_var = Variable(video.unsqueeze(1)).cuda()
            target_var = Variable(target.squeeze()).cuda()

            output = net(audio_var, video_var, states_test)

        loss = criterion(output.squeeze(), target_var)

        # measure accuracy and record loss
        pre_index, predicted = torch.max(output.data, 1)
        # print(pre_index)
        accuracy = (predicted == target.squeeze().cuda()).sum().type(torch.FloatTensor)
        accuracy.mul_((100.0 / args.test_batch_size))
        test_loss.update(loss.item(), args.test_batch_size)
        test_acc.update(accuracy.item(), args.test_batch_size)

        target = target.squeeze().cpu()
        targer = target.tolist()
        predicted = predicted.cpu()
        predicted = predicted.tolist()
        pre_index = pre_index.cpu().tolist()
        all_pred.extend(predicted)
        all_gt.extend(target)
        all_pre_index.extend(pre_index)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]'.format(i, len(test_loader)))

    print('Test finished.')
    print('final loss on test set is {} and final accuracy is {}'.format(test_loss.avg,test_acc.avg))

    fpr, tpr, thresholds = metrics.roc_curve(all_gt, all_pre_index, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.savefig('ROC2.png')
    # plt.show()