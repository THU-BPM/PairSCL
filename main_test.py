from logging import log
import os
import sys
import argparse
import time
import math
import pickle
import random
import csv
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

label_map = ["contradiction", "entailment", "neutral"]

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument("--max_seq_length", default=128, type=int, 
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--model', type=str, default='BERT')
    parser.add_argument('--dataset', type=str, default='SNLI',
                        choices=['SNLI', 'MNLI'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets/preprocessed', help='path to custom dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')


    # distribute
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # ckpt
    parser.add_argument('--ckpt_bert', type=str, default='', help="path to pre-trained model")
    parser.add_argument('--ckpt_classifier', type=str, default='', help="path to pre-trained model")
    args = parser.parse_args()

    return args

def test(val_loader, model, classifier, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter('Accuracy', ':.2f')

    # switch to validate mode
    model.eval()
    classifier.eval()
    res = {}
    res["label"] = []
    res["fc"] = []
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
            features = model(**inputs)
            logits = classifier(features.detach())
            # update metric
            _, pred = logits.topk(1, 1, True, True)
            res["label"] += pred.t().cpu().numpy().tolist()[0]
            res["fc"] += features[1].cpu().numpy().tolist()
            acc1 = accuracy(logits, batch[3])
            top.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    np.save(os.path.join('/'.join(args.ckpt_bert.split('/')[:-1]), 'fc.npy'), np.array(res["fc"]))
    np.save(os.path.join('/'.join(args.ckpt_bert.split('/')[:-1]), 'label.npy'), np.array(res["label"]))
    return top.avg


def test_mnli(val_loader, model, classifier, args):
    
    # switch to validate mode
    model.eval()
    classifier.eval()
    res = {}
    res["id"] = []
    res["label"] = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if args.gpu is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

            # compute loss
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            features = model.encoder(**inputs)
            logits = classifier(features[1])
            # update metric
            _, pred = logits.topk(1, 1, True, True)
            res["id"] += batch[4].cpu().numpy().tolist()
            res["label"] += pred.t().cpu().numpy().tolist()[0]
    return res



def main():
    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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
    ckpt_bert = torch.load(args.ckpt_bert, map_location='cpu')
    ckpt_classifier = torch.load(args.ckpt_classifier, map_location='cpu')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)

    model.load_state_dict(ckpt_bert['model'])
    classifier.load_state_dict(ckpt_classifier['model'])

    cudnn.benchmark = True

    # construct data loader
    if args.dataset == 'SNLI':
        test_file = os.path.join(args.data_folder, args.dataset, "test_data.pkl")
        print("load dataset")
        with open(test_file, "rb") as pkl:
            test_processor = NLIProcessor(pickle.load(pkl))
        test_dataset = load_and_cache_examples(args, test_processor, tokenizer, "test", args.dataset) 
        test_sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)
        acc = test(test_loader, model, classifier, args)
        print("Accuracy: {:.2f}".format(acc))

    elif args.dataset == 'MNLI':
        test_match = os.path.join(args.data_folder, args.dataset, "matched_test_data.pkl")
        test_mismatch = os.path.join(args.data_folder, args.dataset, "mismatched_test_data.pkl")

        print("load dataset")
        with open(test_match, "rb") as pkl:
            pkls = pickle.load(pkl)
            match_processor = NLIProcessor(pkls)

        match_dataset = load_and_cache_examples(args, match_processor, tokenizer, "test_match", args.dataset) 
        match_sampler = None

        with open(test_mismatch, "rb") as pkl:
            mismatch_processor = NLIProcessor(pickle.load(pkl))
        mismatch_dataset = load_and_cache_examples(args, mismatch_processor, tokenizer, "test_mismatch", args.dataset) 
        mismatch_sampler = None

        match_loader = torch.utils.data.DataLoader(
            match_dataset, batch_size=args.batch_size, shuffle=(match_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=match_sampler)
        mismatch_loader = torch.utils.data.DataLoader(
            mismatch_dataset, batch_size=args.batch_size, shuffle=(mismatch_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=mismatch_sampler)
        res1 = test_mnli(match_loader, model, classifier, args)
        res2 = test_mnli(mismatch_loader, model, classifier, args)
        csvFile1 = open('./matched_test_submission.csv', 'w', newline='')
        writer1 = csv.writer(csvFile1)
        writer1.writerow(["pairID", "gold_label"])
        for i in range(len(res1["id"])):
            writer1.writerow([res1["id"][i], label_map[res1["label"][i]]])
        csvFile1.close()
        csvFile2 = open('./mismatched_test_submission.csv', 'w', newline='')
        writer2 = csv.writer(csvFile2)
        writer2.writerow(["pairID", "gold_label"])
        for i in range(len(res2["id"])):
            writer2.writerow([res2["id"][i], label_map[res2["label"][i]]])
        csvFile2.close()
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))


if __name__ == '__main__':
    main()