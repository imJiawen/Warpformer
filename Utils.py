import numpy as np
import pandas as pd
import os
import math
import sys
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import time
import random
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


def pred_loss(prediction, truth, loss_func):
    """ supervised prediction loss, cross entropy or label smoothing. 
    prediction: [B, 2]
    label: [B]
    """
    loss = loss_func(prediction, truth)
    loss = torch.sum(loss)
    return loss



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path=None, dp_flag=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.dp_flag = dp_flag
        self.best_epoch = -1

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None,epoch=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        classifier_state_dict = None

        if dp_flag:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        if classifier is not None:
            classifier_state_dict = classifier.state_dict()
            
        if self.save_path is not None:
            torch.save({
                'model_state_dict':model_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, self.save_path)
        else:
            print("no path assigned")  

        self.val_loss_min = val_loss


def log_info(opt, phase, epoch, acc, rmse=0.0, start=0.0, value_rmse=0.0, auroc=0.0, auprc=0.0, loss=0.0, save=False):
    print('  -(', phase, ') epoch: {epoch}, RMSE: {rmse: 8.5f}, acc: {type: 8.5f}, '
                'AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, Value_RMSE: {value_rmse: 8.5f}, loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'
                .format(epoch=epoch, type=acc, rmse=rmse, auroc=auroc, auprc=auprc, value_rmse=value_rmse, loss=loss, elapse=(time.time() - start) / 60))

    if save and opt.log is not None:
        with open(opt.log, 'a') as f:
            f.write(phase + ':\t{epoch}, TimeRMSE: {rmse: 8.5f},  ACC: {acc: 8.5f}, AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, ValueRMSE: {value_rmse: 8.5f}, Loss: {loss: 8.5f}\n'
                    .format(epoch=epoch, acc=acc, rmse=rmse, auroc=auroc, auprc=auprc, value_rmse=value_rmse, loss=loss))
                

def load_checkpoints(save_path, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False, use_cpu=False):
    if not os.path.getsize(save_path) > 0: 
        print(save_path, " is None file")
        sys.exit(0)

    if use_cpu:
        checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(save_path)
    
    if dp_flag:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])


    if classifier is not None and checkpoint['classifier_state_dict'] is not None:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

    if time_predictor is not None and checkpoint['time_predictor_state_dict'] is not None:
        time_predictor.load_state_dict(checkpoint['time_predictor_state_dict'])

    if decoder is not None and checkpoint['decoder_state_dict'] is not None:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    return model, classifier, time_predictor, decoder


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)  # gpu
    

def evaluate_mc(label, pred, n_class):
    if n_class > 2:
        labels_classes = label_binarize(label, classes=range(n_class))
        pred_scores = pred
        idx = np.argmax(pred_scores, axis=-1)
        preds_label = np.zeros(pred_scores.shape)
        preds_label[np.arange(preds_label.shape[0]), idx] = 1
        acc = metrics.accuracy_score(labels_classes, preds_label)
    else:
        labels_classes = label
        pred_scores = pred[:, 1]
        acc = np.mean(pred.argmax(1) == label)

    try:
        auroc = metrics.roc_auc_score(labels_classes, pred_scores, average='macro')
        auprc = metrics.average_precision_score(labels_classes, pred_scores, average='macro')
    except ValueError:
        auroc = 0
        auprc = 0
    

    return acc, auroc, auprc

def evaluate_ml(true, pred):
    auroc = metrics.roc_auc_score(true, pred, average='macro')
    auprc = metrics.average_precision_score(true, pred, average='macro')
    
    preds_label = np.array(pred > 0.5, dtype=float)
    acc = metrics.accuracy_score(true, preds_label)

    return acc, auroc, auprc
