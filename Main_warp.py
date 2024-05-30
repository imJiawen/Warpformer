import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import Utils
from Utils import *

from Dataset_MM import get_clints_hii_data, get_activity_data, get_physionet_data
from warpformer.Models import Classifier, Hie_Encoder
from tqdm import tqdm
import os
import sys
import gc

eps=1e-7

def train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier):
    """ Epoch operation in training phase. """

    model.train()
    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0,0,0

    for train_batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        train_batch, labels = map(lambda x: x.to(opt.device), train_batch)

        observed_tp = train_batch[:, :, 2 * opt.num_types]

        if opt.task != "active":
            limit_length = torch.argmax(observed_tp, -1).detach().cpu().numpy()
            max_len = max(limit_length)
        else:
            max_len = train_batch.shape[1]

        observed_data, observed_mask, observed_tp, tau = \
            train_batch[:, :max_len+1, :opt.num_types], train_batch[:, :max_len+1, opt.num_types:2 * opt.num_types], train_batch[:, :max_len+1, 2 * opt.num_types], \
                train_batch[:, :max_len+1, (2 * opt.num_types)+1: (3 * opt.num_types)+1]
        del train_batch

        """ forward """
        optimizer.zero_grad()

        out = model(observed_tp, observed_data, observed_mask, tau=tau, return_almat=False) # [B,L,K,D]
        sup_pred = classifier(out)
        
        if sup_pred.dim() == 1:
            sup_pred = sup_pred.unsqueeze(0)


        if opt.task == "active":
            sup_pred = sup_pred.view(-1, opt.n_classes)
            labels = labels.view(-1)

        if opt.task == "wbm":
            loss = torch.sum(pred_loss_func((sup_pred), labels.float()))
            sup_pred = torch.sigmoid(sup_pred)

        else:
            loss = torch.sum(pred_loss_func((sup_pred), labels))
            sup_pred = torch.softmax(sup_pred, dim=-1)

        if torch.any(torch.isnan(loss)):
            print("exit nan in pred loss!!!")
            print("sup_pred\n", sup_pred)
            sys.exit(0)
        
        losses.append(loss.item())
        loss.backward()
        
        
        sup_preds.append(sup_pred.detach().cpu().numpy())
        sup_labels.append(labels.detach().cpu().numpy())
        
        del out, loss, sup_pred, labels

        B, L = observed_mask.size(0), observed_mask.size(1)
        
        # loss.backward()
        optimizer.step()

        del observed_data, observed_mask, observed_tp, tau
        gc.collect()
        torch.cuda.empty_cache()

    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)
        
        if opt.task == 'wbm':
            acc, auroc, auprc = evaluate_ml(sup_labels, sup_preds)
        else:
            acc, auroc, auprc = evaluate_mc(sup_labels, sup_preds, opt.n_classes)

    return acc, auroc, auprc, train_loss



def eval_epoch(model, validation_data, pred_loss_func, opt, classifier, save_res=False):
    """ Epoch operation in evaluation phase. """

    model.eval()

    valid_losses = []
    sup_preds = []
    sup_labels = []
    acc, auroc, auprc = 0,0,0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ prepare data """
             #mTAN
            train_batch, labels = map(lambda x: x.to(opt.device), batch)
            observed_tp = train_batch[:, :, 2 * opt.num_types]
            
            if opt.task != "active":
                limit_length = torch.argmax(observed_tp, -1).detach().cpu().numpy()
                max_len = max(limit_length)
            else:
                max_len = train_batch.shape[1]

            # [B,L,K], [B,L,K], [B,L]
            observed_data, observed_mask, observed_tp, tau = \
                train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2 * opt.num_types], train_batch[:, :max_len, 2 * opt.num_types], \
                    train_batch[:, :max_len, (2 * opt.num_types)+1: (3 * opt.num_types)+1]
            del train_batch
            
            
            out = model(observed_tp, observed_data, observed_mask, tau=tau) # [B,L,K,D]

            sup_pred = classifier(out)
            if sup_pred.dim() == 1:
                sup_pred = sup_pred.unsqueeze(0)

            if opt.task == "active":
                sup_pred = sup_pred.view(-1, opt.n_classes)
                labels = labels.view(-1)

            if opt.task == "wbm":
                valid_loss = torch.sum(pred_loss_func((sup_pred + eps), labels.float()))
                sup_pred = torch.sigmoid(sup_pred)
            else:
                valid_loss = torch.sum(pred_loss_func((sup_pred + eps), labels))
                sup_pred = torch.softmax(sup_pred, dim=-1)

            sup_preds.append(sup_pred.detach().cpu().numpy())
            sup_labels.append(labels.detach().cpu().numpy())

            if valid_loss != 0:
                valid_losses.append(valid_loss.item())

            del out, observed_data, observed_mask, observed_tp, tau, valid_loss

            gc.collect()
            torch.cuda.empty_cache()

    valid_loss = np.average(valid_losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels, axis=0)
        sup_preds = np.concatenate(sup_preds, axis=0)
        sup_preds = np.nan_to_num(sup_preds)

        # save prediction results
        if save_res:
            np.save(opt.save_res + '_prediction.npy',sup_preds)
        
        if opt.task == 'wbm':
            acc, auroc, auprc = evaluate_ml(sup_labels, sup_preds)
        else:
            acc, auroc, auprc = evaluate_mc(sup_labels, sup_preds, opt.n_classes)
        
    return acc, auroc, auprc, valid_loss

def train(model, training_data, validation_data, testing_data, optimizer, scheduler, pred_loss_func, opt, \
                        early_stopping=None, classifier=None, save_path=None):

    epoch = 0
    if not opt.test_only:
        """ Start training. """
        for epoch_i in range(opt.epoch):
            
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_acc, train_auroc, train_auprc, train_loss = train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier)
            log_info(opt, 'Train', epoch, train_acc, start=start, auroc=train_auroc, auprc=train_auprc, loss=train_loss, save=True)

            if not opt.retrain:
                start = time.time()
                valid_acc, valid_auroc, valid_auprc, valid_loss = eval_epoch(model, validation_data, pred_loss_func, opt, classifier)
                log_info(opt, 'Valid', epoch, valid_acc, start=start, auroc=valid_auroc, auprc=valid_auprc, loss=valid_loss, save=True)

                if early_stopping is not None:
                    early_stopping(-valid_auroc, model, classifier, epoch=epoch)
                    # early_stopping(valid_loss, model, classifier)

                    if early_stopping.early_stop: #and not opt.pretrain:
                        print("Early stopping. Training Done.")
                        break
            else:
                start = time.time()
                test_acc, test_auroc, test_auprc, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)

                log_info(opt, 'Testing', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, save=True)

            scheduler.step()

    if not opt.retrain and save_path is not None:
        print("Testing...")
        model, classifier, _, _ = load_checkpoints(save_path, model, classifier=classifier, dp_flag=opt.dp_flag)

        start = time.time()
        test_acc, auroc, auprc, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)

        if early_stopping is not None and early_stopping.best_epoch > 0:
            best_epoch = early_stopping.best_epoch
        else:
            best_epoch = epoch
            
        log_info(opt, 'Testing', best_epoch, test_acc, start=start, auroc=auroc, auprc=auprc, save=True)


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='path/to/datasets')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_types', type=int, default=23)

    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_inner_hid', type=int, default=32)
    parser.add_argument('--d_k', type=int, default=8)
    parser.add_argument('--d_v', type=int, default=8)

    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--task', type=str, default='nan')

    parser.add_argument('--debug_flag', action='store_true')
    parser.add_argument('--dp_flag', action='store_true')
    parser.add_argument('--load_in_batch', action='store_true')
    parser.add_argument('--hourly', action='store_true')
    parser.add_argument('--warp_num', type=str, default=None)
    parser.add_argument('--nonneg_trans', type=str, default='sigmoid')
    parser.add_argument('--input_only', action='store_true')
    parser.add_argument('--dec_only', action='store_true')
    parser.add_argument('--remove_rep', type=str, default=None, help='select from type, abs, rel, tem')

    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--median_len', type=int, default=50)

    parser.add_argument('--warpfunc', type=str, default='l2')
    parser.add_argument('--warpact', type=str, default='relu')
    parser.add_argument('--only_down', action='store_true')
    parser.add_argument('--full_attn', action='store_true')

    opt = parser.parse_args()
    seed = opt.seed

    opt.device = torch.device('cuda')
    # opt.device = torch.device('cpu')
    
    opt.n_classes = 2

    if opt.task == "mor":
        opt.median_len = 63

    elif opt.task == "decom":
        opt.median_len = 34

    elif opt.task == 'vent' or opt.task == "vaso":
        opt.n_classes = 4
        opt.median_len = 10

    elif opt.task == 'wbm':
        opt.n_classes = 54
        opt.median_len = 78

    elif opt.task == 'los':
        opt.n_classes = 9
        opt.median_len = 34

    Utils.setup_seed(seed)

    warp_str = ''
    
    if opt.only_down:
        warp_str += 'onlydown_'

    if opt.remove_rep is not None:
        warp_str = warp_str + 'rm'+ opt.remove_rep + '_'


    """ prepare dataloader """
    if opt.task == 'active':
        trainloader, validloader, testloader, opt.num_types = get_activity_data(opt, opt.device)
        opt.n_classes = 7
        opt.median_len = 50
        opt.full_attn = True
        # opt.d_model = 64
        # opt.n_head = 8
        # opt.n_layers = 3


    elif opt.task == 'physio':
        trainloader, validloader, testloader, opt.num_types = get_physionet_data(opt, opt.device)
        opt.n_classes = 2
        opt.median_len = 72
        # opt.d_model = 32
        # opt.n_head = 1
        # opt.n_layers = 2

    else:
        trainloader, validloader, testloader = get_clints_hii_data(opt)

    opt.log = opt.root_path + opt.log
    

    if opt.save_path is not None:
        opt.save_path = opt.root_path + opt.save_path
    
    if opt.load_path is not None:
        opt.load_path = opt.root_path + opt.load_path
    

    if opt.hourly:
        opt.log = opt.log + opt.task + '_hourly' 
        save_name = opt.task + '_hourly' 
    else:
        opt.log = opt.log + warp_str + opt.task
        save_name = warp_str + opt.task

    str_warp_num = opt.warp_num

    if opt.warp_num is not None:
        opt.warp_num = [float(i) for i in opt.warp_num.split("_")]
        
        if opt.warp_num[-1] > 3:
            opt.warp_num = [int(i) for i in opt.warp_num]
        else:
            opt.warp_num = [int(i*opt.median_len) for i in opt.warp_num]
    else:
        opt.warp_num = []

    """ prepare model """
    model = Hie_Encoder(
        opt=opt,
        num_types=opt.num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        )

    para_list = list(model.parameters())

    mort_classifier = Classifier(opt.d_model, opt.num_types, opt.n_classes)
    para_list += list(mort_classifier.parameters())
    
    # load model
    if opt.load_path is not None:
        print("Loading checkpoints...")
        model, mort_classifier, _, _ = load_checkpoints(opt.load_path, model, classifier=mort_classifier, dp_flag=False)

    
    model = model.to(opt.device)
    
    for mod in [model, mort_classifier]:
        if mod is not None:
            mod = mod.to(opt.device)
    
    if opt.dp_flag:
        model = nn.DataParallel(model)
    
    if not opt.hourly:
        if opt.warp_num is not None and len(opt.warp_num) > 0:
            opt.log = opt.log + '_warp' + str_warp_num
            save_name = save_name + '_warp' + str_warp_num

    if opt.input_only:
        opt.log = opt.log + '_onlyinput'
        save_name = save_name + '_onlyinput'

    if opt.dec_only:
        opt.log = opt.log + '_dec_only'
        save_name = save_name + '_dec_only'


    if opt.retrain:
        print("Re-trianing...")
        opt.log = opt.log + '_retrain'
        save_name = save_name + '_retrain'

    opt.log = opt.log + "_seed" + str(seed) + '.log'

    if opt.save_path is not None:
        opt.save_res = opt.save_path + save_name + '_seed' + str(opt.seed)
        save_path = opt.save_path + save_name + '_seed' + str(opt.seed) + '.h5'
    else:
        save_path = None
    
    """ optimizer and scheduler """

    params = (para_list)
    optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function """
    if opt.task == 'wbm':
        pred_loss_func = nn.BCEWithLogitsLoss(reduction='none').to(opt.device)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(opt.device)


    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('[Info] parameters: {}\n'.format(opt))

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    print('[Info] parameters: {}'.format(opt))

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, save_path=save_path, dp_flag=opt.dp_flag)

    """ train the model """
    train(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, early_stopping, mort_classifier, save_path=save_path)


if __name__ == '__main__':
    main()
