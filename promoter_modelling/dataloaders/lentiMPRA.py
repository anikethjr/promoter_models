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
import pytorch_lightning as pl

import torchmetrics

from joblib import Parallel, delayed

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

class lentiMPRADataset(Dataset):
    def __init__(self, df, split, num_cells, cell_names, \
                 cache_dir, use_cache=True, shrink_set=False):
        super().__init__()

        self.df = df
        self.split = split
        self.num_cells = num_cells
        self.cell_names = cell_names
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # create/load one-hot encoded input sequences
        if self.use_cache and os.path.exists(os.path.join(self.cache_dir, f"{split}_seqs.npy")):
            self.all_seqs = np.load(os.path.join(self.cache_dir, f"{split}_seqs.npy"))
            print("Loaded cached one-hot encoded sequences, shape = {}".format(self.all_seqs.shape))
        else:
            print("Creating one-hot encoded sequences")
            self.all_seqs = []

            for seq in tqdm(self.df['sequence'].values, ncols=0):
                self.all_seqs.append(fasta_utils.one_hot_encode(seq).astype(np.float32))
            
            self.all_seqs = np.array(self.all_seqs)
            np.save(os.path.join(self.cache_dir, f"{split}_seqs.npy"), self.all_seqs)
            print("Done! Shape = {}".format(self.all_seqs.shape))
        
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

        if shrink_set:
            self.all_seqs = self.all_seqs[:10]
            self.all_outputs = self.all_outputs[:10]
            self.valid_outputs_mask = self.valid_outputs_mask[:10]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.all_seqs[idx], self.all_outputs[idx], self.valid_outputs_mask[idx]


class lentiMPRADataLoader(pl.LightningDataModule):
    def download_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        self.large_scale_positions_path = os.path.join(self.cache_dir, "large_scale_positions.ods")
        self.large_scale_exps_path = os.path.join(self.cache_dir, "large_scale_exps.ods")
        self.designed_sequences_path = os.path.join(self.cache_dir, "designed_sequences.ods")
        self.designed_sequences_exps_path = os.path.join(self.cache_dir, "designed_sequences_exps.ods")

        if not os.path.exists(self.large_scale_positions_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ME80jtw2QGt4nUo_pfNVktyt-zv9EYvg' -O {}".format(self.large_scale_positions_path))
            assert os.path.exists(self.large_scale_positions_path)
        
        if not os.path.exists(self.large_scale_exps_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=19YQGNWg8zHxRcF92cb_wYhkZUCxrjsMS' -O {}".format(self.large_scale_exps_path))
            assert os.path.exists(self.large_scale_exps_path)
        
        if not os.path.exists(self.designed_sequences_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LPeGz_-cAidxgPsuHAQDny1Ie-FDYZ_4' -O {}".format(self.designed_sequences_path))
            assert os.path.exists(self.designed_sequences_path)
        
        if not os.path.exists(self.designed_sequences_exps_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1QJIU5dWe8PQnPyGeXiqJnYGDkZPp1j8q' -O {}".format(self.designed_sequences_exps_path))
            assert os.path.exists(self.designed_sequences_exps_path)
        
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
                 n_cpus = 8, \
                 train_chromosomes = ['1', '3', '5', '6', '7', '8', '11', '12', '14', '15', '16', '18', '19', '22', 'X', 'Y', 'specialchr'], \
                 test_chromosomes = ['2', '9', '10', '13', '20', '21'], \
                 val_chromosomes = ['4', '17'], \
                 use_cache = True, 
                 shrink_test_set=False):
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)
        
        print("Creating lentiMPRA DataLoader object")

        self.name = "lentiMPRA"

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
        self.download_data()

        self.cell_names = ["K562", "HepG2", "WTC11"]
        self.num_cells = len(self.cell_names)
        self.output_names = self.cell_names
        self.num_outputs = len(self.output_names)

        self.final_dataset_path = os.path.join(self.cache_dir, "final_dataset.tsv")
        
        if not os.path.exists(self.final_dataset_path):
            print("Creating final dataset")

            self.large_scale_positions_path = os.path.join(self.cache_dir, "large_scale_positions.ods")
            self.large_scale_exps_path = os.path.join(self.cache_dir, "large_scale_exps.ods")
            self.designed_sequences_path = os.path.join(self.cache_dir, "designed_sequences.ods")
            self.designed_sequences_exps_path = os.path.join(self.cache_dir, "designed_sequences_exps.ods")

            # load data published by the authors
            large_scale_positions = pd.read_excel(self.large_scale_positions_path, sheet_name=None)
            large_scale_exps = pd.read_excel(self.large_scale_exps_path, sheet_name=None)
            designed_sequences = pd.read_excel(self.designed_sequences_path, sheet_name=None)
            designed_sequences_exps = pd.read_excel(self.designed_sequences_exps_path, sheet_name=None)
            
            final_dataset = {}
            final_dataset["sequence"] = []
            final_dataset["chromosome"] = []
            final_dataset["orientation"] = []

            all_cells = self.cell_names

            for cell in all_cells:
                final_dataset[cell] = []
            
            for cell in all_cells:
                print(cell)
                other_cells = [i for i in all_cells if i != cell]
                large_scale_seqs = large_scale_positions["{} large-scale".format(cell)][["name", "chr.hg38", "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"]]
                large_scale_seqs = large_scale_seqs.rename({"chr.hg38": "chromosome", "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)": "sequence"}, axis=1)
                large_scale_seqs["sequence"] = large_scale_seqs.apply(lambda x: x["sequence"][15:-15], axis=1)
                if cell == "WTC11":
                    large_scale_seqs["chromosome"] = large_scale_seqs.apply(lambda x: np.nan if str(x["chromosome"])=="nan" else "chr"+str(x["chromosome"]), axis=1)
                
                large_scale_exps_forward = large_scale_exps["{}_summary_data".format(cell)][["name", "mean"]]
                large_scale_exps_forward = large_scale_exps_forward.merge(large_scale_seqs, on="name", how="inner")

                large_scale_exps_forward_and_reverse = large_scale_exps["{}_forward_reverse".format(cell)]
                if cell == "WTC11":
                    large_scale_exps_forward_and_reverse["name"] = large_scale_exps_forward_and_reverse.apply(lambda x: x["name"] + "_F", axis=1)
                large_scale_exps_forward_and_reverse = large_scale_exps_forward_and_reverse.merge(large_scale_seqs, on="name", how="inner")
                large_scale_exps_forward_and_reverse["rc_sequence"] = large_scale_exps_forward_and_reverse.apply(lambda x: str(Seq(x["sequence"]).reverse_complement()), axis=1)

                final_dataset["sequence"].extend(large_scale_exps_forward["sequence"])
                final_dataset["chromosome"].extend(large_scale_exps_forward["chromosome"])
                final_dataset["orientation"].extend(["forward"]*len(large_scale_exps_forward))
                final_dataset[cell].extend(large_scale_exps_forward["mean"])
                for c in other_cells:
                    final_dataset[c].extend([np.nan]*len(large_scale_exps_forward))
                print("Forward orientation sequences = {}".format(len(large_scale_exps_forward)))

                final_dataset["sequence"].extend(large_scale_exps_forward_and_reverse["rc_sequence"])
                final_dataset["chromosome"].extend(large_scale_exps_forward_and_reverse["chromosome"])
                final_dataset["orientation"].extend(["reverse"]*len(large_scale_exps_forward_and_reverse))
                final_dataset[cell].extend(large_scale_exps_forward_and_reverse["reverse [log2(rna/dna)]"])
                for c in other_cells:
                    final_dataset[c].extend([np.nan]*len(large_scale_exps_forward_and_reverse))
                print("Reverse orientation sequences = {}".format(len(large_scale_exps_forward_and_reverse)))

            common_sequences = designed_sequences_exps["all_cell_types_summary"].merge(designed_sequences["joint library"][["name", "chr.hg38", "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"]], on="name", how="inner")
            common_sequences = common_sequences.rename({"chr.hg38": "chromosome", "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)": "sequence"}, axis=1)
            common_sequences["sequence"] = common_sequences.apply(lambda x: x["sequence"][15:-15], axis=1)
            common_sequences["chromosome"] = common_sequences.apply(lambda x: np.nan if str(x["chromosome"])=="nan" else "chr"+str(x["chromosome"]), axis=1)
            final_dataset["sequence"].extend(common_sequences["sequence"])
            final_dataset["chromosome"].extend(common_sequences["chromosome"])
            final_dataset["orientation"].extend(["forward"]*len(common_sequences))
            for cell in all_cells:
                final_dataset[cell].extend(common_sequences["{} [log2(rna/dna)]".format(cell)])
            print("\nNumber of common sequences = {}".format(len(common_sequences)))

            final_dataset = pd.DataFrame(final_dataset)
            final_dataset["chromosome"] = final_dataset.apply(lambda x: "undefined" if str(x["chromosome"]) == "nan" else x["chromosome"], axis=1)
            final_dataset["chromosome"] = final_dataset.apply(lambda x: "specialchr" if len(x["chromosome"].split("_")) > 1 else x["chromosome"], axis=1)
            print("Final number of sequences = {}".format(final_dataset.shape))
            for cell in all_cells:
                print("Number of measurements in {} = {}".format(cell, np.sum(~np.isnan(final_dataset[cell]))))

            # create train, test and val datasets
            final_dataset["is_train"] = final_dataset.apply(lambda x: (x["chromosome"][3:] in self.train_chromosomes) or (x["chromosome"] in self.train_chromosomes), axis=1)
            final_dataset["is_test"] = final_dataset.apply(lambda x: (x["chromosome"][3:] in self.test_chromosomes) or (x["chromosome"] in self.test_chromosomes), axis=1)
            final_dataset["is_val"] = final_dataset.apply(lambda x: (x["chromosome"][3:] in self.val_chromosomes) or (x["chromosome"] in self.val_chromosomes), axis=1)

            # split undefined chromosome samples into train, test and val
            undefined_chromosome_indices = final_dataset[final_dataset["chromosome"] == "undefined"].index
            num_undefined_chromosome = len(undefined_chromosome_indices)
            train_num_undefined_chromosome = int(np.ceil(0.7 * num_undefined_chromosome))
            test_num_undefined_chromosome = int(np.ceil(0.2 * num_undefined_chromosome))
            val_num_undefined_chromosome = num_undefined_chromosome - train_num_undefined_chromosome - test_num_undefined_chromosome

            print("Train num undefined chromosome = {}".format(train_num_undefined_chromosome))
            print("Test num undefined chromosome = {}".format(test_num_undefined_chromosome))
            print("Val num undefined chromosome = {}".format(val_num_undefined_chromosome))

            shuffled_ind = undefined_chromosome_indices[np.random.permutation(num_undefined_chromosome)]
            train_ind = shuffled_ind[:train_num_undefined_chromosome]
            test_ind = shuffled_ind[train_num_undefined_chromosome:train_num_undefined_chromosome+test_num_undefined_chromosome]
            val_ind = shuffled_ind[train_num_undefined_chromosome+test_num_undefined_chromosome:]

            final_dataset.loc[train_ind, "is_train"] = True
            final_dataset.loc[test_ind, "is_test"] = True
            final_dataset.loc[val_ind, "is_val"] = True

            assert final_dataset["is_train"].sum() + final_dataset["is_test"].sum() + final_dataset["is_val"].sum() == final_dataset.shape[0]

            final_dataset.to_csv(self.final_dataset_path, sep="\t", index=None)
        else:
            print("Loading final dataset from cache...")

        self.final_dataset = pd.read_csv(self.final_dataset_path, sep="\t")

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
        self.train_dataset = lentiMPRADataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating test dataset")
        self.test_dataset = lentiMPRADataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache, shrink_set=shrink_test_set)
        print("Creating val dataset")
        self.val_dataset = lentiMPRADataset(self.val_set, "val", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache, shrink_set=shrink_test_set)
        
        print("Train set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                                        100.0*self.train_set.shape[0]/self.final_dataset.shape[0]))
        print("Test set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                                        100.0*self.test_set.shape[0]/self.final_dataset.shape[0]))
        print("Val set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                                        100.0*self.val_set.shape[0]/self.final_dataset.shape[0]))
        print("Completed Instantiation of lentiMPRA DataLoader")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)