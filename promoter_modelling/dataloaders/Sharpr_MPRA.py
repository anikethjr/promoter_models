import numpy as np
import pdb
import pandas as pd
import os
import scipy.stats as stats
from tqdm import tqdm
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import torchmetrics

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

def read_data(data_dir, split):
    filename = os.path.join(data_dir, "{}.hdf5".format(split))
    f = h5py.File(filename, "r")

    # Get the data
    X = np.array(f["X"]["sequence"])
    Y = np.array(f["Y"]["output"])
    
    print("{} set X shape = {}".format(split, X.shape))
    print("{} set Y shape = {}".format(split, Y.shape))
    
    return X, Y

class SharprMPRADataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()

        self.X = X
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):      
        return self.X[idx], self.Y[idx]

# class used to read, process and build train, val, test sets using the Sharpr MPRA dataset preprocessed by Movva et al. (2019)
class SharprMPRADataLoader(pl.LightningDataModule):
    def download_data(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        if not os.path.exists(os.path.join(self.data_dir, "train.hdf5")):
            os.system("wget --no-check-certificate https://mitra.stanford.edu/kundaje/projects/mpra/data/train.hdf5 -O {}".format(os.path.join(self.data_dir, "train.hdf5")))
            assert os.path.exists(os.path.join(self.data_dir, "train.hdf5")), "Could not download train.hdf5"
        if not os.path.exists(os.path.join(self.data_dir, "valid.hdf5")):
            os.system("wget --no-check-certificate https://mitra.stanford.edu/kundaje/projects/mpra/data/valid.hdf5 -O {}".format(os.path.join(self.data_dir, "valid.hdf5")))
            assert os.path.exists(os.path.join(self.data_dir, "valid.hdf5")), "Could not download valid.hdf5"
        if not os.path.exists(os.path.join(self.data_dir, "test.hdf5")):
            os.system("wget --no-check-certificate https://mitra.stanford.edu/kundaje/projects/mpra/data/test.hdf5 -O {}".format(os.path.join(self.data_dir, "test.hdf5")))
            assert os.path.exists(os.path.join(self.data_dir, "test.hdf5")), "Could not download test.hdf5"
        
    def update_metrics(self, y_hat, y, loss, split):
        self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].update(loss)
        for i, output in enumerate(self.output_names):
            for met in ["MSE", "MAE", "R2", "PearsonR", "SpearmanR"]:
                self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].update(y_hat[:, i], y[:, i])
    
    def compute_metrics(self, split):
        metrics_dict = {}

        metrics_dict["{}_{}_avg_epoch_loss".format(split, self.name)] = self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].compute()

        metrics_set = ["MSE", "MAE", "R2", "PearsonR", "SpearmanR"]
        
        for met in metrics_set:
            for i, output in enumerate(self.output_names):
                metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)] = self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].compute()
                
                if "{}_{}_mean_{}".format(split, self.name, met) not in metrics_dict:
                    metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] = 0
                
                if "rep" not in output: # we don't want to include the replicates in the mean
                    metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] += metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)]
            
            metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] /= 4 # divide by 4 because there are 4 unique conditions
        
        self.all_metrics[split].reset()
                
        return metrics_dict
    
    def __init__(self, \
                 batch_size, \
                 data_dir, \
                 n_cpus = 8):
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)
        
        print("Creating Sharpr MPRA DataLoader object")
        
        self.name = "SharprMPRA"

        self.task = "regression"
        self.loss_fn = nn.MSELoss()
        self.with_mask = False

        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.data_dir = data_dir

        # download data if not already downloaded
        self.download_data()
        
        self.num_outputs = 12
        self.output_names = ["k562_minp_rep1", "k562_minp_rep2", "k562_minp_avg", \
                             "k562_sv40p_rep1", "k562_sv40p_rep1", "k562_sv40p_avg", \
                             "hepg2_minp_rep1", "hepg2_minp_rep2", "hepg2_minp_avg", \
                             "hepg2_sv40p_rep1", "hepg2_sv40p_rep2", "hepg2_sv40p_avg"]
        self.promoter_windows_relative_to_TSS = [] # dummy attribute, needed to maintain uniformity with other datasets

        # specify metrics to track for this dataloader
        self.metrics = {}
        for i, cell in enumerate(self.output_names):
            self.metrics["{}_{}_MSE".format(self.name, cell)] = torchmetrics.MeanSquaredError()
            self.metrics["{}_{}_MAE".format(self.name, cell)] = torchmetrics.MeanAbsoluteError()
            self.metrics["{}_{}_R2".format(self.name, cell)] = torchmetrics.R2Score()
            self.metrics["{}_{}_PearsonR".format(self.name, cell)] = torchmetrics.PearsonCorrCoef()
            self.metrics["{}_{}_SpearmanR".format(self.name, cell)] = torchmetrics.SpearmanCorrCoef() 
        self.metrics["{}_avg_epoch_loss".format(self.name)] = torchmetrics.MeanMetric()       
        self.metrics = torchmetrics.MetricCollection(self.metrics)

        self.all_metrics = {}
        for split in ["train", "val", "test"]:
            self.all_metrics[split] = self.metrics.clone(prefix=split + "_")
        
        print("Creating train dataset")
        train_x, train_y = read_data(self.data_dir, "train")
        self.train_dataset = SharprMPRADataset(train_x, train_y)
        print("Creating test dataset")
        val_x, val_y = read_data(self.data_dir, "valid")
        self.val_dataset = SharprMPRADataset(val_x, val_y)
        print("Creating val dataset")
        test_x, test_y = read_data(self.data_dir, "test")
        self.test_dataset = SharprMPRADataset(test_x, test_y)
        
        print("Completed Instantiation of Sharpr MPRA DataLoader")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)