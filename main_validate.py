import os
import sys
import argparse
import time
import math
import pickle
import random
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from util import NLIProcessor, adjust_learning_rate, accuracy, warmup_learning_rate, load_and_cache_examples, save_model, AverageMeter, ProgressMeter
from bert_model import BertForCL, LinearClassifier, PairSupConBert

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
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
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
    parser.add_argument('--lr_decay_epochs', type=str, default='10,15',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')

    # distribute
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')


    # parameters
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ckpt', type=str, default='', help="path to pre-trained model")
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


def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
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
    best_acc1 = 0
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
    ), is_train=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    classifier = LinearClassifier(BertForCL.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=3,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ))
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()


    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    model.load_state_dict(state_dict)

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
        validate_file = os.path.join(args.data_folder, args.dataset, "dev_data.pkl")
        print("load dataset")
        with open(train_file, "rb") as pkl:
            train_processor = NLIProcessor(pickle.load(pkl))
        
        train_dataset = load_and_cache_examples(args, train_processor, tokenizer, "train", args.dataset)
        
        with open(validate_file, "rb") as pkl:
            validate_processor = NLIProcessor(pickle.load(pkl))

        validate_dataset = load_and_cache_examples(args, validate_processor, tokenizer, "validate", args.dataset)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            validate_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)
        else:
            train_sampler = None
            validate_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        validate_loader = torch.utils.data.DataLoader(
            validate_dataset, batch_size=args.batch_size, shuffle=(validate_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=validate_sampler)
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(args, optimizer, epoch)

            time1 = time.time()
            loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, args)
            time2 = time.time()
            print('epoch {}, total time {:.2f}, loss {:.2f}, accuracy {:.2f}'.format(epoch, time2 - time1, loss, train_acc))
            
            _, acc = validate(validate_loader, model, classifier, criterion, epoch, args)
            if acc > best_acc1:
                best_acc1 = acc
                print('best accuracy: {:.2f}'.format(best_acc1))
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
            save_file = os.path.join(args.save_folder, 'classifier_last.pth')
            save_model(classifier, optimizer, args, epoch, save_file, False)
        print("best accuracy: {:.2f}".format(best_acc1))
    

    elif args.dataset == 'MNLI':
        train_file = os.path.join(args.data_folder, args.dataset, "train_data.pkl")
        validate_match = os.path.join(args.data_folder, args.dataset, "matched_dev_data.pkl")
        validate_mismatch = os.path.join(args.data_folder, args.dataset, "mismatched_dev_data.pkl")

        print("load dataset")
        with open(train_file, "rb") as pkl:
            train_processor = NLIProcessor(pickle.load(pkl))
        
        train_dataset = load_and_cache_examples(args, train_processor, tokenizer, "train", args.dataset)
        
        with open(validate_match, "rb") as pkl:
            match_processor = NLIProcessor(pickle.load(pkl))

        match_dataset = load_and_cache_examples(args, match_processor, tokenizer, "validate_match", args.dataset)

        with open(validate_mismatch, "rb") as pkl:
            mismatch_processor = NLIProcessor(pickle.load(pkl))

        mismatch_dataset = load_and_cache_examples(args, mismatch_processor, tokenizer, "validate_match", args.dataset)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            match_sampler = torch.utils.data.distributed.DistributedSampler(match_dataset)
            mismatch_sampler = torch.utils.data.distributed.DistributedSampler(mismatch_dataset)

        else:
            train_sampler = None
            match_sampler = None
            mismatch_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        match_loader = torch.utils.data.DataLoader(
            match_dataset, batch_size=args.batch_size, shuffle=(match_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=match_sampler)
        mismatch_loader = torch.utils.data.DataLoader(
            mismatch_dataset, batch_size=args.batch_size, shuffle=(mismatch_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=mismatch_sampler)
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(args, optimizer, epoch)

            time1 = time.time()
            loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, args)
            time2 = time.time()
            print('epoch {}, total time {:.2f}, loss {:.2f}, accuracy {:.2f}'.format(epoch, time2 - time1, loss, train_acc))
            
            _, acc1 = validate(match_loader, model, classifier, criterion, epoch, args)
            _, acc2 = validate(mismatch_loader, model, classifier, criterion, epoch, args)

            if acc1 > best_acc1:
                best_acc1 = acc1
                print('best accuracy: {:.2f}'.format(best_acc1))
            if acc2 > best_acc1:
                best_acc1 = acc2
                print('best accuracy: {:.2f}'.format(best_acc1))
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
            save_file = os.path.join(args.save_folder, 'classifier_last.pth')
            save_model(classifier, optimizer, args, epoch, save_file, False)
        print("best accuracy: {:.2f}".format(best_acc1))

    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))



def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top = AverageMeter('Accuracy', ':.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    classifier.train()
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
        with torch.no_grad():
            features = model(**inputs)
        logits = classifier(features.detach())
        labels = batch[3]
        loss = criterion(logits.view(-1, 3), labels.view(-1))
        losses.update(loss.item(), bsz)

        # update metric
        acc1 = accuracy(logits, labels)
        top.update(acc1[0].item(), bsz)

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
    return losses.avg, top.avg

def validate(val_loader, model, classifier, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top = AverageMeter('Accuracy', ':.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top],
        prefix="Epoch: [{}]".format(epoch))

    # switch to validate mode
    model.eval()
    classifier.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            bsz = batch[0].size(0)

            if args.gpu is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

            # compute loss
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            labels = batch[3]
            features = model(**inputs)
            logits = classifier(features.detach())
            loss = criterion(logits.view(-1, 3), labels.view(-1))

            # update metric
            # print(logits)
            losses.update(loss.item(), bsz)
            acc1 = accuracy(logits, batch[3])
            top.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)
    return losses.avg, top.avg



if __name__ == '__main__':
    main()
