from __future__ import print_function, division
import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from networks.Video_Net import DeepVAD_video
from utils.logger import Logger
from datasets import VideoDataset
from utils import utils as utils

if __name__ == '__main__':

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs', help='log directory')
    parser.add_argument('--outdir', type=str, default='outputs', help='output directory')
    parser.add_argument('--num_epochs', type=int, default=51, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--time_depth', type=int, default=15, help='number of time frames in each video/audio sample')
    parser.add_argument('--workers', type=int, default=0, help='num workers for data loading')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum factor')
    parser.add_argument('--save_freq', type=int, default=1, help='freq of saving the model')
    parser.add_argument('--print_freq', type=int, default=50, help='freq of printing the results')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--pre_train', type=str, default='', help='path to a pre-trained network')
    args = parser.parse_args()
    print(args, end='\n\n')

    # Settings
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Use Device: ', device)

    model = DeepVAD_video(device)
    model = model.to(device)

    # create train + val datasets
    train_dataset = VideoDataset(DataDir='E2EVAD/train/', timeDepth = args.time_depth, is_train=True)
    val_dataset = VideoDataset(DataDir='E2EVAD/test/', timeDepth = args.time_depth, is_train=False)
    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)
    print('{} batches found, {} train batches and {} test batches.'.format(len(val_loader)+len(train_loader),
                                                                           len(train_loader),
                                                                           len(val_loader)))

    # set the logger
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = Logger(args.logdir)

    # create a saved models folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # create loss (optionaly assign each class with different weight
    weight = torch.FloatTensor(2)
    weight[0] = 1  # class 0 - non-speech
    weight[1] = 1  # class 1 - speech
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)

    # init from a saved checkpoint
    if args.pre_train is not '':
        model_name = args.pre_train
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            pretrained = checkpoint['state_dict']
            model.load_state_dict(pretrained,strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_name, checkpoint['epoch']))
        else:
            print('Couldn\'t load model from {}'.format(model_name))
    else:
        print('Training the model from scratch.')

    # def test method
    def test():
        test_acc  = utils.AverageMeter()
        test_loss = utils.AverageMeter()
        test_prec = utils.AverageMeter()
        test_reca = utils.AverageMeter()
        test_f1 = utils.AverageMeter()

        model.eval()
        print('Test started.')
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                input, target = model.parse_batch(data)  
                output = model(input)

                loss = criterion(output.squeeze(), target)

                # measure accuracy and record loss
                _, predicted = torch.max(output.data, 1)
                target = target.squeeze().cpu()
                predicted = predicted.cpu()
                accuracy = accuracy_score(target, predicted)
                precision = precision_score(target, predicted)
                recall = recall_score(target, predicted)
                f1score = f1_score(target, predicted)

                test_loss.update(loss.item(), args.batch_size)
                test_acc.update(accuracy.item(), args.batch_size)
                test_prec.update(precision.item(), args.batch_size)
                test_reca.update(recall.item(), args.batch_size)
                test_f1.update(f1score.item(), args.batch_size)

                if i>0 and i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}] , \t'
                      'Loss {loss.avg:.4f} , \t'
                      'Acc {top1.avg:.3f}, \t'
                      'precision {prec.avg:.3f}, \t'
                      'Recall {reca.avg:.3f}, \t'
                      'F1_score {f1.avg:.3f}, \t'
                      .format(
                    epoch, i, len(val_loader), loss=test_loss, top1=test_acc, prec=test_prec, reca=test_reca, f1=test_f1))

        model.train()
        print('Test finished.')
        return test_acc.avg, test_loss.avg, test_prec.avg, test_reca.avg, test_f1.avg


    ### main training loop ###

    best_accuracy = 0
    best_epoch = 0
    step = 0

    for epoch in range(0,args.num_epochs):

        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        end = time.time()

        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        train_prec = utils.AverageMeter()
        train_reca = utils.AverageMeter()
        train_f1 = utils.AverageMeter()

        # train for one epoch
        for i, data in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()
            input, target = model.parse_batch(data)
        
            # measure data loading time
            data_time.update(time.time() - end)

            output = model(input)

            loss = criterion(output.squeeze(), target)
            loss.backward()

            # compute gradient and do SGD step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isnan(grad_norm):
                print("grad norm is nan. Do not update model.")
            else:
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end) # actually step time

            # measure accuracy and record loss
            _, predicted = torch.max(output.data, 1)
            target = target.squeeze().cpu()
            predicted = predicted.cpu()

            accuracy = accuracy_score(target, predicted)
            precision = precision_score(target, predicted)
            recall = recall_score(target, predicted)
            f1score = f1_score(target, predicted)

            train_loss.update(loss.item(), args.batch_size)
            train_acc.update(accuracy.item(), args.batch_size)
            train_prec.update(precision.item(), args.batch_size)
            train_reca.update(recall.item(), args.batch_size)
            train_f1.update(f1score.item(), args.batch_size)

            # tensorboard logging
            logger.scalar_summary('train loss', loss.item(), step + 1)
            logger.scalar_summary('train accuracy', accuracy.item(), step + 1)
            logger.scalar_summary('train precision', precision.item(), step + 1)
            logger.scalar_summary('train recall', recall.item(), step + 1)
            logger.scalar_summary('train f1score', f1score.item(), step + 1)
            step+=1

            end = time.time()

            if i > 0 and i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] , \t'
                      'LR {3} , \t'
                      'Time {batch_time.avg:.3f} , \t'
                      'Data {data_time.avg:.3f} , \t'
                      'Loss {loss.avg:.4f} , \t'
                      'Acc {top1.avg:.3f}, \t'
                      'precision {prec.avg:.3f}, \t'
                      'Recall {reca.avg:.3f}, \t'
                      'F1_score {f1.avg:.3f}, \t'
                      .format(
                    epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                    data_time=data_time, loss=train_loss, top1=train_acc, prec=train_prec, reca=train_reca, f1=train_f1))

        # evaluate on validation set
        accuracy, loss, precision, recall, f1score = test()
        # logger
        logger.scalar_summary('Test Accuracy', accuracy, epoch)
        logger.scalar_summary('Test Loss ', loss, epoch)
        logger.scalar_summary('Test Precision ', precision, epoch)
        logger.scalar_summary('Test Recall ', recall, epoch)
        logger.scalar_summary('Test F1Score ', f1score, epoch)
        logger.scalar_summary('LR ', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = False
        if accuracy > best_accuracy:
            is_best = True
            best_epoch = epoch

        best_accuracy = max(accuracy, best_accuracy)

        print('Average accuracy on validation set is: {}%'.format(accuracy))
        print('Best accuracy so far is: {}% , at epoch #{}'.format(best_accuracy,best_epoch))

        if epoch % args.save_freq == 0:
            checkpoint_name = "%s/acc_%.3f_epoch_%03d.pkl" % (args.outdir, accuracy, epoch)
            utils.save_checkpoint(state={
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_name)
            