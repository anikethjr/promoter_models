import numpy as np
import pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

import torchmetrics

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

def determine_cell_specific_class(row):
    if row["class"] == "ClassIII":
        return "ClassIII_" + row["final_provenance"].split(" ~ ")[1][len("type="):]
    elif row["class"] == "ClassII":
        return "ClassII_" + row["final_provenance"].split(" ~ ")[0][len("cell="):] + "_" + row["final_provenance"].split(" ~ ")[1][len("type="):] + "_" + row["final_provenance"].split(" ~ ")[2][len("total_num_motifs="):]
    else:
        log2FoldChange = float(row["final_provenance"].split(" ~ ")[2][len("log2FoldChange="):])
        first_description = row["final_provenance"].split(" ~ ")[4][len("description="):].split(";")[0]
        all_cells = ["NK92", "THP1", "JURKAT"]
        out = "ClassI"
        for c in all_cells:
            if c in first_description:
                if "upregulated" in first_description:
                    out += "_{}_up".format(c)
                    assert log2FoldChange > 0
                else:
                    out += "_{}_down".format(c)
                    assert log2FoldChange < 0
                break
        return out
    
def GC_content(seq):
    num_GC = len([seq[i] for i in range(len(seq)) if seq[i] == "G" or seq[i] == "C"])
    return float(num_GC) / len(seq)

class FluorescenceDataset(Dataset):
    def __init__(self, df, split_name, num_cells, cell_names, cache_dir, use_cache):
        super().__init__()

        self.df = df
        self.num_cells = num_cells
        self.cell_names = cell_names
                
        # create/load one-hot encoded input sequences
        if use_cache and os.path.exists(os.path.join(cache_dir, "{}_seqs.npy".format(split_name))):
            self.all_seqs = np.load(os.path.join(cache_dir, "{}_seqs.npy".format(split_name)))
            print("Loaded cached one-hot encoded sequences, shape = {}".format(self.all_seqs.shape))
        else:
            print("Creating one-hot encoded sequences")
            self.all_seqs = []
            for i in tqdm(range(self.df.shape[0])):
                row = df.iloc[i]
                promoter_seq = row["sequence"]
                onehot_seq = fasta_utils.one_hot_encode(promoter_seq).astype(np.float32)                  
                self.all_seqs.append(onehot_seq)
            self.all_seqs = np.array(self.all_seqs)
            np.save(os.path.join(cache_dir, "{}_seqs.npy".format(split_name)), self.all_seqs)
            print("Done! Shape = {}".format(self.all_seqs.shape))
        
        # create MTL targets
        self.y = []        
        for cell in self.cell_names:
            self.y.append(df[cell].to_numpy().astype(np.float32))
        self.y = np.array(self.y).T        
        print("Targets shape = {}".format(self.y.shape))    
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):                
        return self.all_seqs[idx], \
               self.y[idx]

class FluorescenceDataLoader(pl.LightningDataModule):    
    def download_data(self):
        if not os.path.exists(os.path.join(self.cache_dir, "Raw_Promoter_Counts.csv")):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=15p6GhDop5BsUPryZ6pfKgwJ2XEVHRAYq' -O {}".format(os.path.join(self.cache_dir, "Raw_Promoter_Counts.csv")))
            assert os.path.exists(os.path.join(self.cache_dir, "Raw_Promoter_Counts.csv"))
        if not os.path.exists(os.path.join(self.cache_dir, "final_list_of_all_promoter_sequences_fixed.tsv")):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1kTfsZvsCz7EWUhl-UZgK0B31LtxJH4qG' -O {}".format(os.path.join(self.cache_dir, "final_list_of_all_promoter_sequences_fixed.tsv")))
            assert os.path.exists(os.path.join(self.cache_dir, "final_list_of_all_promoter_sequences_fixed.tsv"))
    
    def update_metrics(self, y_hat, y, loss, split):
        self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].update(loss)
        for i, output in enumerate(self.output_names):
            for met in ["Accuracy", "Precision", "Recall", "F1"]:
                self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].update(y_hat[:, i], y[:, i])
    
    def compute_metrics(self, split):
        metrics_dict = {}

        metrics_dict["{}_{}_avg_epoch_loss".format(split, self.name)] = self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].compute()

        metrics_set = ["Accuracy", "Precision", "Recall", "F1"]
        
        for met in metrics_set:
            for i, output in enumerate(self.output_names):
                metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)] = self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].compute()
                
                if "{}_{}_mean_{}".format(split, self.name, met) not in metrics_dict:
                    metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] = 0
                
                metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] += metrics_dict["{}_{}_{}_{}".format(split, self.name, output, met)]
            
            metrics_dict["{}_{}_mean_{}".format(split, self.name, met)] /= len(self.output_names)
        
        self.all_metrics[split].reset()
        
        return metrics_dict
        
    def __init__(self, \
                 batch_size, \
                 cache_dir, \
                 seed = None, \
                 n_cpus = 0, \
                 min_reads = 5, \
                 train_fraction = 0.7, \
                 val_fraction = 0.1, \
                 use_cache = True, \
                 return_specified_cells = None, \
                 predict_DE = False):
        
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.seed = seed
            cache_dir = os.path.join(cache_dir + "_seed_{}".format(seed))
            print("Using seed = {}".format(seed))
        
        print("Creating Fluorescence classification DataLoader object")

        self.name = "Fluorescence_classification"

        self.task = "classification"
        self.use_1hot_for_classification = False
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.with_mask = False

        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        # mandatory param
        self.promoter_windows_relative_to_TSS = []
        
        self.num_cells = 3
        self.num_replicates = 2
        self.min_reads = min_reads
        self.cell_names = np.array(["JURKAT", "K562", "THP1"])
        self.num_outputs = self.num_cells
        self.output_names = self.cell_names
                
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        # download data if necessary
        self.download_data()
        
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = 1.0 - train_fraction - val_fraction

        self.predict_DE = predict_DE

        if self.predict_DE:
            self.merged_cache_path = os.path.join(self.cache_dir, "merged_DE.tsv")
        else:        
            self.merged_cache_path = os.path.join(self.cache_dir, "merged.tsv")

        if not os.path.exists(self.merged_cache_path):
            self.measurements = pd.read_csv(os.path.join(self.cache_dir, "Raw_Promoter_Counts.csv"))
            
            self.measurements["keep"] = True
            for col in self.measurements.columns:
                if col.endswith("_sum") and col != "cum_sum":
                    self.measurements["keep"] = self.measurements["keep"] & (self.measurements[col] >= min_reads)
            self.measurements = self.measurements[self.measurements["keep"]].drop("keep", axis=1).reset_index(drop=True)
            
            for cell in self.cell_names:
                first_letter_of_cell_name = cell[:1]
                self.measurements[cell] = 0
                for rep in range(self.num_replicates):
                    if self.predict_DE:
                        other_cells = [c for c in self.cell_names if c != cell]
                        other_cells_first_letters = [c[:1] for c in other_cells]
                        avg_ratio = 0
                        for other_cell, other_cell_first_letter in zip(other_cells, other_cells_first_letters):
                            avg_ratio += (self.measurements["{}{}_P4".format(other_cell_first_letter, rep+1)] + 1) / (self.measurements["{}{}_P7".format(other_cell_first_letter, rep+1)] + 1)
                        avg_ratio /= len(other_cells)

                        cur_ratio = (self.measurements["{}{}_P4".format(first_letter_of_cell_name, rep+1)] + 1) / (self.measurements["{}{}_P7".format(first_letter_of_cell_name, rep+1)] + 1)

                        # DE = ratio of P4 to P7 in cell of interest / ratio of P4 to P7 in other cells
                        self.measurements[cell] += np.log2(cur_ratio / avg_ratio)
                    else:
                        self.measurements[cell] += np.log2((self.measurements["{}{}_P4".format(first_letter_of_cell_name, rep+1)] + 1) / (self.measurements["{}{}_P7".format(first_letter_of_cell_name, rep+1)] + 1))
                self.measurements[cell] /= self.num_replicates

            # binarize measurements by assigning 1 to all measurements above the median
            for cell in self.cell_names:
                self.measurements["numerical_" + cell] = self.measurements[cell]
                self.measurements[cell] = self.measurements[cell] > np.median(self.measurements[cell])

            self.sequence_properties = pd.read_csv(os.path.join(self.cache_dir, "final_list_of_all_promoter_sequences_fixed.tsv"), sep="\t")
            self.sequence_properties["cell_specific_class"] = self.sequence_properties.apply(lambda x: determine_cell_specific_class(x), axis=1)

            self.merged = self.measurements.merge(self.sequence_properties, on="sequence", how="inner")
            
            # divide data into train, val, test while also balancing class membership and GC content
            self.merged["GC_content"] = self.merged.apply(lambda x: GC_content(x["sequence"]), axis=1)
            self.merged["GC_content_bin"] = (np.floor(self.merged["GC_content"] / 0.05)).astype(int)
            
            self.all_classes = sorted(list(set(self.merged["cell_specific_class"])))
            
            all_train_inds = []
            all_val_inds = []
            all_test_inds = []
            
            np.random.seed(97)
            if seed is not None:
                np.random.seed(seed)

            for cl in self.all_classes:
                class_subset = self.merged[self.merged["cell_specific_class"] == cl]
                for i in range(20):
                    bin_subset = class_subset[class_subset["GC_content_bin"] == i]
                    bin_subset_inds = bin_subset.index.to_numpy()

                    bin_subset_inds_shuffled = bin_subset_inds[np.random.permutation(bin_subset_inds.shape[0])]

                    train_inds = bin_subset_inds_shuffled[0: int(np.ceil(bin_subset_inds_shuffled.shape[0] * self.train_fraction))]
                    val_inds = bin_subset_inds_shuffled[int(np.ceil(bin_subset_inds_shuffled.shape[0] * self.train_fraction)): \
                                                        int(np.ceil(bin_subset_inds_shuffled.shape[0] * self.train_fraction) \
                                                        + np.floor(bin_subset_inds_shuffled.shape[0] * self.val_fraction))]
                    test_inds = bin_subset_inds_shuffled[int(np.ceil(bin_subset_inds_shuffled.shape[0] * self.train_fraction) \
                                                        + np.floor(bin_subset_inds_shuffled.shape[0] * self.val_fraction)):]

                    all_train_inds += list(train_inds)
                    all_val_inds += list(val_inds)
                    all_test_inds += list(test_inds)

            all_train_inds = np.array(all_train_inds)
            all_val_inds = np.array(all_val_inds)
            all_test_inds = np.array(all_test_inds)

            assert (len(all_train_inds) + len(all_val_inds) + len(all_test_inds)) == self.merged.shape[0]

            self.merged["is_train"] = False
            self.merged["is_train"].iloc[all_train_inds] = True

            self.merged["is_val"] = False
            self.merged["is_val"].iloc[all_val_inds] = True

            self.merged["is_test"] = False
            self.merged["is_test"].iloc[all_test_inds] = True
            
            self.merged.to_csv(self.merged_cache_path, sep="\t", index=False)
        
        self.merged = pd.read_csv(self.merged_cache_path, sep="\t")
        self.all_classes = sorted(list(set(self.merged["cell_specific_class"])))
        
        self.train_set = self.merged[self.merged["is_train"]].reset_index(drop=True)
        self.test_set = self.merged[self.merged["is_test"]].reset_index(drop=True)
        self.val_set = self.merged[self.merged["is_val"]].reset_index(drop=True)
        
        self.return_specified_cells = return_specified_cells
        if not self.return_specified_cells is None:
            print("Keeping only specified cells data")
            self.num_cells = len(self.return_specified_cells)
            self.cell_names = self.cell_names[self.return_specified_cells]
            self.num_outputs = self.num_cells
            self.output_names = self.cell_names

        # specify metrics to track for this dataloader
        self.metrics = {}
        for i, cell in enumerate(self.output_names):            
            self.metrics["{}_{}_Accuracy".format(self.name, cell)] = torchmetrics.Accuracy(task='binary')
            self.metrics["{}_{}_Precision".format(self.name, cell)] = torchmetrics.Precision(task='binary')
            self.metrics["{}_{}_Recall".format(self.name, cell)] = torchmetrics.Recall(task='binary')
            self.metrics["{}_{}_F1".format(self.name, cell)] = torchmetrics.F1Score(task='binary')
        self.metrics["{}_avg_epoch_loss".format(self.name)] = torchmetrics.MeanMetric()
        self.metrics = torchmetrics.MetricCollection(self.metrics)

        self.all_metrics = {}
        for split in ["train", "val", "test"]:
            self.all_metrics[split] = self.metrics.clone(prefix=split + "_")
                
        print("Creating train dataset")
        self.train_dataset = FluorescenceDataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                                 cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating test dataset")
        self.test_dataset = FluorescenceDataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating val dataset")
        self.val_dataset = FluorescenceDataset(self.val_set, "val", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache)
        
        print("Train set has {} promoter-expression pairs from {} total pairs ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                                                           self.merged.shape[0], \
                                                                                                           100.0*self.train_set.shape[0]/self.merged.shape[0]))
        print("Test set has {} promoter-expression data from {} total pairs ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                                                         self.merged.shape[0], \
                                                                                                         100.0*self.test_set.shape[0]/self.merged.shape[0]))
        print("Val set has {} promoter-expression data from {} total pairs ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                                                        self.merged.shape[0], \
                                                                                                        100.0*self.val_set.shape[0]/self.merged.shape[0]))
        
        # print number of positive and negative examples in each dataset
        for split in ["train", "test", "val"]:
            split_set = getattr(self, split + "_set")
            for cell in self.cell_names:
                num_positive = (split_set[cell] == 1).sum()
                num_negative = (split_set[cell] == 0).sum()
                percent_positive = 100.0 * num_positive / (num_positive + num_negative)
                percent_negative = 100.0 * num_negative / (num_positive + num_negative)
                print("{} set has {} positive examples ({:.2f}%) and {} negative examples ({:.2f}%) for cell {}".format(split, num_positive, percent_positive, num_negative, percent_negative, cell))
        
        print("Completed Instantiation of Fluorescence DataLoader")        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)