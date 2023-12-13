import numpy as np
import pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import gc
from Bio.Seq import Seq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

import torchmetrics

from joblib import Parallel, delayed

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

class STARRSeqDataset(Dataset):
    def __init__(self, df, split, num_cells, cell_names, \
                 cache_dir, use_cache=True):
        super().__init__()

        self.df = df
        self.split = split
        self.num_cells = num_cells
        self.cell_names = cell_names
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # create outputs
        self.all_outputs = {}
        for cell in self.cell_names:
            self.all_outputs[cell] = np.array(self.df[cell].values, dtype=np.float32)
        
        # create mask since not all sequences have outputs for all cells
        self.valid_outputs_mask = []
        for cell in self.cell_names:
            self.valid_outputs_mask.append(~np.isnan(self.all_outputs[cell]))
        self.valid_outputs_mask = np.stack(self.valid_outputs_mask, axis=1)
        print("Valid outputs mask shape = {}".format(self.valid_outputs_mask.shape))
        
        print("Number of valid outputs for each cell type:")
        for cell, mask in zip(self.cell_names, self.valid_outputs_mask.T):
            print(cell, np.sum(mask))
        
        # set invalid outputs to -100000
        for cell in self.cell_names:
            self.all_outputs[cell][np.isnan(self.all_outputs[cell])] = -100000
        self.all_outputs = np.stack([self.all_outputs[cell] for cell in self.cell_names], axis=1)
        print("All outputs shape = {}".format(self.all_outputs.shape))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = fasta_utils.one_hot_encode(row["sequence"]).astype(np.float32)
        return torch.tensor(seq), torch.tensor(self.all_outputs[idx]), torch.tensor(self.valid_outputs_mask[idx])
    
def pad_collate(batch):
    (seq, targets, mask) = zip(*batch)
    seq_lens = [x.shape[0] for x in seq]

    seq = pad_sequence(seq, batch_first=True, padding_value=0)
    targets = torch.vstack(targets)
    mask = torch.vstack(mask)
    
    return seq, targets, mask
    
class STARRSeqDataLoader(pl.LightningDataModule):
    def download_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        self.K562_peaks_bed_path = os.path.join(self.cache_dir, "K562_repMerged_starrpeaker.peak.final.bed")
        self.HepG2_peaks_bed_path = os.path.join(self.cache_dir, "HepG2_repMerged_starrpeaker.peak.final.bed")
        self.fasta_file = os.path.join(self.common_cache_dir, "hg38.fa")
        
        if not os.path.exists(self.K562_peaks_bed_path):
            os.system("wget 'https://www.encodeproject.org/files/ENCFF045TVA/@@download/ENCFF045TVA.bed.gz' -O {}".format(self.K562_peaks_bed_path + ".gz"))
            os.system("gunzip {}".format(self.K562_peaks_bed_path  + ".gz"))
            assert os.path.exists(self.K562_peaks_bed_path)

        if not os.path.exists(self.HepG2_peaks_bed_path):
            os.system("wget 'https://www.encodeproject.org/files/ENCFF047LDJ/@@download/ENCFF047LDJ.bed.gz' -O {}".format(self.HepG2_peaks_bed_path + ".gz"))
            os.system("gunzip {}".format(self.HepG2_peaks_bed_path  + ".gz"))
            assert os.path.exists(self.HepG2_peaks_bed_path)
            
        if not os.path.exists(self.fasta_file):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O {}".format(self.fasta_file + ".gz"))
            os.system("gunzip {}".format(self.fasta_file + ".gz"))
            assert os.path.exists(self.fasta_file)
        
    def update_metrics(self, y_hat, y, loss, split):
        self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].update(loss)
        for i, output in enumerate(self.output_names):
            valid_y_hat = y_hat[:, i][~(y[:, i] == -100000)]
            valid_y = y[:, i][~(y[:, i] == -100000)]            
            if len(valid_y_hat) == 0:
                continue
            for met in ["MSE", "MAE", "R2", "PearsonR", "SpearmanR"]:
                self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].update(valid_y_hat, valid_y)
    
    def compute_metrics(self, split):
        metrics_dict = {}

        metrics_dict["{}_{}_avg_epoch_loss".format(split, self.name)] = self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].compute()

        metrics_set = ["MSE", "MAE", "R2", "PearsonR", "SpearmanR"]

        for met in metrics_set:
            for i, output in enumerate(self.output_names):
                try:
                    metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)] = self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].compute()
                except:
                    print("WARNING metric could not be computed {}_{}_{}_{}".format(split, self.name, output, met))
                    metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)] = 0
                
                if "{}_{}_mean_{}".format(split, self.name, met) not in metrics_dict:
                    metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] = 0
                
                metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] += metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)]
            
            metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] /= len(self.output_names)

        self.all_metrics[split].reset()
        
        return metrics_dict
    
    def __init__(self, \
                 batch_size, \
                 cache_dir, \
                 common_cache_dir, \
                 n_cpus = 0, \
                 train_chromosomes = ['1', '3', '5', '6', '7', '8', '11', '12', '14', '15', '16', '18', '19', '22', 'X', 'Y'], \
                 test_chromosomes = ['2', '9', '10', '13', '20', '21'], \
                 val_chromosomes = ['4', '17'], \
                 use_cache = True):
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)
        
        print("Creating STARRSeq DataLoader object")

        self.name = "STARRSeq"

        self.task = "regression"
        self.loss_fn = nn.MSELoss()
        self.with_mask = True

        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        self.train_chromosomes = train_chromosomes
        self.test_chromosomes = test_chromosomes
        self.val_chromosomes = val_chromosomes

        self.promoter_windows_relative_to_TSS = [] # dummy

        self.cache_dir = cache_dir
        self.common_cache_dir = common_cache_dir
        self.download_data()

        self.cell_names = ["K562", "HepG2"]
        self.num_cells = len(self.cell_names)
        self.output_names = self.cell_names
        self.num_outputs = len(self.output_names)

        self.final_dataset_path = os.path.join(self.cache_dir, "final_dataset.tsv")
        
        if not os.path.exists(self.final_dataset_path):
            print("Creating final dataset")
            
            self.fasta_file = os.path.join(self.common_cache_dir, "hg38.fa")
            fasta_extractor = fasta_utils.FastaStringExtractor(self.fasta_file)
            self.final_dataset = None
            for cell in self.cell_names:
                peaks_bed_path = os.path.join(self.cache_dir, f"{cell}_repMerged_starrpeaker.peak.final.bed")
            
                peaks = pd.read_csv(peaks_bed_path, sep="\t", 
                                    names=["chromosome", "start", "end", "name", "score", "strand", 
                                           "fold_change", "output_coverage", "input_coverage", "log_pval", "log_qval"], 
                                    header=None)
                print(f"Read {peaks.shape[0]} {cell} peaks")
                
                # first extract sequences
                all_seqs = []
                for i in tqdm(range(peaks.shape[0])):
                    row = peaks.iloc[i]
                    seq = fasta_utils.get_sequence(row["chromosome"], \
                                                   row["start"], \
                                                   row["end"], \
                                                   row["strand"], \
                                                   fasta_extractor)
                    all_seqs.append(seq)
                    
                peaks["sequence"] = all_seqs
                peaks["length"] = peaks["end"] - peaks["start"]

                # # filter out peaks with length > 1000
                # print(f"Filtering out {peaks.shape[0] - peaks[peaks['length'] <= 1000].shape[0]} peaks with length > 1000")
                # peaks = peaks[peaks["length"] <= 1000].reset_index(drop=True)
                # print(f"Filtered out peaks with length > 1000, {peaks.shape[0]} peaks remaining")

                if self.final_dataset is None:
                    self.final_dataset = peaks[["sequence", "chromosome", "start", "end", "fold_change"]].copy()
                else:
                    self.final_dataset = self.final_dataset.merge(peaks[["sequence", "chromosome", "start", "end", "fold_change"]], on=["sequence", "chromosome", "start", "end"], how="outer")
                self.final_dataset = self.final_dataset.rename({"fold_change": cell}, axis=1).reset_index(drop=True)
                print(f"Combined dataset size after merging = {self.final_dataset.shape[0]}")
            
            print(f"Final dataset size = {self.final_dataset.shape[0]}")
            self.final_dataset.to_csv(self.final_dataset_path, sep="\t", index=False)
        
        self.final_dataset = pd.read_csv(self.final_dataset_path, sep="\t")
        self.final_dataset["is_train"] = self.final_dataset.apply(lambda x: x["chromosome"][3:] in self.train_chromosomes, axis=1)
        self.final_dataset["is_test"] = self.final_dataset.apply(lambda x: x["chromosome"][3:] in self.test_chromosomes, axis=1)
        self.final_dataset["is_val"] = self.final_dataset.apply(lambda x: x["chromosome"][3:] in self.val_chromosomes, axis=1)

        for cell in self.cell_names:
            print("Number of measurements in {} = {}".format(cell, np.sum(~np.isnan(self.final_dataset[cell]))))
            
        self.train_set = self.final_dataset[self.final_dataset["is_train"]].reset_index(drop=True)
        self.test_set = self.final_dataset[self.final_dataset["is_test"]].reset_index(drop=True)
        self.val_set = self.final_dataset[self.final_dataset["is_val"]].reset_index(drop=True)

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
        self.train_dataset = STARRSeqDataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating test dataset")
        self.test_dataset = STARRSeqDataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating val dataset")
        self.val_dataset = STARRSeqDataset(self.val_set, "val", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache)
        
        print("Train set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                                        100.0*self.train_set.shape[0]/self.final_dataset.shape[0]))
        print("Test set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                                        100.0*self.test_set.shape[0]/self.final_dataset.shape[0]))
        print("Val set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                                        100.0*self.val_set.shape[0]/self.final_dataset.shape[0]))
        print("Completed Instantiation of STARRSeq DataLoader")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)