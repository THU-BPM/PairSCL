import os
from posixpath import split
import sys
import argparse
import time
import math
import pickle
import random
import warnings
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from util import NLIProcessor, adjust_learning_rate, warmup_learning_rate, load_and_cache_examples, save_model, AverageMeter, ProgressMeter
from bert_model import PairSupConBert, BertForCL
from losses import SupConLoss


CUDA_VISIBLE_DEVICES=2,3

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument("--max_seq_length", default=128, type=int, 
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--model', type=str, default='BERT')
    parser.add_argument('--dataset', type=str, default='SNLI',
                        choices=['SNLI', 'MNLI'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets/preprocessed', help='path to custom dataset')

    # training
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,8',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')

    # distribute
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # parameters
    parser.add_argument('--alpha', type=float, default=1.0, help="the parameter to balance the training objective")
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset)

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}'.\
        format(args.dataset, args.model, args.learning_rate,
               args.weight_decay, args.batch_size, args.temp)

    if args.cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    # warm-up for large-batch training,
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model_name)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def train(train_loader, model, criterion_sup, criterion_ce, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)

        if args.gpu is not None:
            for i in range(len(batch)):
                batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        batch = tuple(t.cuda() for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
        features = model(**inputs)
        # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        loss = criterion_sup(features, batch[3]) + criterion_ce(features, batch[3])

        # update metric
        losses.update(loss.item(), bsz)
        
        # AdamW
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)
    return losses.avg


def main():
    args = parse_option()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    model = PairSupConBert(BertForCL.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=128,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion_supcon = SupConLoss(temperature=args.temp).cuda(args.gpu) 
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # construct data loader
    if args.dataset == 'SNLI':
        train_file = os.path.join(args.data_folder, args.dataset, "train_data.pkl") 
    elif args.dataset == 'MNLI':
        train_file = os.path.join(args.data_folder, args.dataset, "train_data.pkl") 
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))
    
    print("load dataset")
    with open(train_file, "rb") as pkl:
        processor = NLIProcessor(pickle.load(pkl))

    train_dataset = load_and_cache_examples(args, processor, tokenizer, "train", args.dataset)
    # dataset_size = len(train_dataset)
    # split_size = int(0.01*dataset_size)
    # train_dataset, _ = torch.utils.data.random_split(train_dataset, [split_size,dataset_size-split_size])
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        time1 = time.time()
        loss = train(train_loader, model, criterion_supcon, criterion_ce, optimizer, epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.2f}'.format(epoch, time2 - time1, loss))
        
    # save the last model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        save_file = os.path.join(args.save_folder, 'last.pth')
        save_model(model, optimizer, args, args.epochs, save_file, False)  
 

if __name__ == '__main__':
    main()