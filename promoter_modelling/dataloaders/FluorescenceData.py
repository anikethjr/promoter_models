import numpy as np
import pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import shlex

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L

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
    def __init__(self, df, split_name, num_cells, cell_names, cache_dir, use_cache, use_construct):
        super().__init__()

        self.df = df
        self.num_cells = num_cells
        self.cell_names = cell_names
                
        # create/load one-hot encoded input sequences
        if use_construct:
            file_name = "{}_seqs_construct.npy".format(split_name)
        else:
            file_name = "{}_seqs.npy".format(split_name)

        if use_cache and os.path.exists(os.path.join(cache_dir, file_name)):
            self.all_seqs = np.load(os.path.join(cache_dir, file_name))
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
            np.save(os.path.join(cache_dir, file_name), self.all_seqs)
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

class FluorescenceDataLoader(L.LightningDataModule):    
    def download_data(self):
        self.cache_dir_counts = shlex.quote(os.path.join(self.cache_dir, "Raw_Promoter_Counts.csv"))
        if not os.path.exists(self.cache_dir_counts):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=15p6GhDop5BsUPryZ6pfKgwJ2XEVHRAYq' -O {}".format(self.cache_dir_counts))
            assert os.path.exists(self.cache_dir_counts)

        self.cache_dir_seq_list = shlex.quote(os.path.join(self.cache_dir, "final_list_of_all_promoter_sequences_fixed.tsv"))
        if not os.path.exists(self.cache_dir_seq_list):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1kTfsZvsCz7EWUhl-UZgK0B31LtxJH4qG' -O {}".format(self.cache_dir_seq_list))
            assert os.path.exists(self.cache_dir_seq_list)
    
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
                 zscore = False, \
                 use_cache = True, \
                 return_specified_cells = None, \
                 predict_DE = False, 
                 use_construct=False):
        
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.seed = seed
            cache_dir = os.path.join(cache_dir + "_seed_{}".format(seed))
            print("Using seed = {}".format(seed))
        
        print("Creating Fluorescence DataLoader object")

        self.name = "Fluorescence"

        self.task = "regression"
        self.loss_fn = nn.MSELoss()
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
        
        self.zscore = zscore
        self.predict_DE = predict_DE
        self.use_construct = use_construct

        if self.predict_DE:
            self.merged_cache_path = os.path.join(self.cache_dir, "merged_DE.tsv")
        else:        
            self.merged_cache_path = os.path.join(self.cache_dir, "merged.tsv")

        if not os.path.exists(self.merged_cache_path):
            print("Processing sequencing data from Fluorescence Assay...")
            self.measurements = pd.read_csv(os.path.join(self.cache_dir, "Raw_Promoter_Counts.csv"))
            
            self.measurements["keep"] = True
            for col in self.measurements.columns:
                if col.endswith("_sum") and col != "cum_sum":
                    self.measurements["keep"] = self.measurements["keep"] & (self.measurements[col] >= min_reads)
            self.measurements = self.measurements[self.measurements["keep"]].drop("keep", axis=1).reset_index(drop=True)

            # divide the read counts by the total number of reads across sequences
            for col in self.measurements.columns:
                if not (col.endswith("_sum") or col == "sequence"):
                    print("{}, sum before normalizing = {}".format(col, self.measurements[col].sum()))
                    self.measurements[col] = self.measurements[col] + 1.0 # pseudocount
                    self.measurements[col] = self.measurements[col] / self.measurements[col].sum() # normalize
                    print("{}, sum after normalizing = {}".format(col, self.measurements[col].sum()))
            
            for cell in self.cell_names:
                first_letter_of_cell_name = cell[:1]
                self.measurements[cell] = 0
                for rep in range(self.num_replicates):
                    if self.predict_DE:
                        other_cells = [c for c in self.cell_names if c != cell]
                        other_cells_first_letters = [c[:1] for c in other_cells]
                        avg_ratio = 0
                        for other_cell, other_cell_first_letter in zip(other_cells, other_cells_first_letters):
                            avg_ratio += (self.measurements["{}{}_P4".format(other_cell_first_letter, rep+1)]) / (self.measurements["{}{}_P7".format(other_cell_first_letter, rep+1)])
                        avg_ratio /= len(other_cells)

                        cur_ratio = (self.measurements["{}{}_P4".format(first_letter_of_cell_name, rep+1)]) / (self.measurements["{}{}_P7".format(first_letter_of_cell_name, rep+1)])

                        # DE = ratio of P4 to P7 in cell of interest / ratio of P4 to P7 in other cells
                        self.measurements[cell] += np.log2(cur_ratio / avg_ratio)
                    else:
                        self.measurements[f"log2({cell}{rep+1}_R)"] = np.log2((self.measurements["{}{}_P4".format(first_letter_of_cell_name, rep+1)]) / (self.measurements["{}{}_P7".format(first_letter_of_cell_name, rep+1)]))
                        self.measurements[cell] += np.log2((self.measurements["{}{}_P4".format(first_letter_of_cell_name, rep+1)]) / (self.measurements["{}{}_P7".format(first_letter_of_cell_name, rep+1)]))
                self.measurements[cell] /= self.num_replicates

            if self.zscore:
                for cell in self.cell_names:
                    self.measurements[cell] = stats.zscore(self.measurements[cell])

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

            print("Finished processing sequencing data from Fluorescence Assay.")
        
        self.merged = pd.read_csv(self.merged_cache_path, sep="\t")

        # add scaffolding to the sequence if using constructs
        if self.use_construct:
            full_construct = "GGGTCTCTCTGGTTAGACCAGATCTGAGCCTGGGAGCTCTCTGGCTAACTAGGGAACCCACTGCTTAAGCCTCAATAAAGCTTGCCTTGAGTGCTTCAAGTAGTGTGTGCCCGTCTGTTGTGTGACTCTGGTAACTAGAGATCCCTCAGACCCTTTTAGTCAGTGTGGAAAATCTCTAGCAGTGGCGCCCGAACAGGGACTTGAAAGCGAAAGGGAAACCAGAGGAGCTCTCTCGACGCAGGACTCGGCTTGCTGAAGCGCGCACGGCAAGAGGCGAGGGGCGGCGACTGGTGAGTACGCCAAAAATTTTGACTAGCGGAGGCTAGAAGGAGAGAGATGGGTGCGAGAGCGTCAGTATTAAGCGGGGGAGAATTAGATCGCGATGGGAAAAAATTCGGTTAAGGCCAGGGGGAAAGAAAAAATATAAATTAAAACATATAGTATGGGCAAGCAGGGAGCTAGAACGATTCGCAGTTAATCCTGGCCTGTTAGAAACATCAGAAGGCTGTAGACAAATACTGGGACAGCTACAACCATCCCTTCAGACAGGATCAGAAGAACTTAGATCATTATATAATACAGTAGCAACCCTCTATTGTGTGCATCAAAGGATAGAGATAAAAGACACCAAGGAAGCTTTAGACAAGATAGAGGAAGAGCAAAACAAAAGTAAGACCACCGCACAGCAAGCGGCCGCTGATCTTCAGACCTGGAGGAGGAGATATGAGGGACAATTGGAGAAGTGAATTATATAAATATAAAGTAGTAAAAATTGAACCATTAGGAGTAGCACCCACCAAGGCAAAGAGAAGAGTGGTGCAGAGAGAAAAAAGAGCAGTGGGAATAGGAGCTTTGTTCCTTGGGTTCTTGGGAGCAGCAGGAAGCACTATGGGCGCAGCGTCAATGACGCTGACGGTACAGGCCAGACAATTATTGTCTGGTATAGTGCAGCAGCAGAACAATTTGCTGAGGGCTATTGAGGCGCAACAGCATCTGTTGCAACTCACAGTCTGGGGCATCAAGCAGCTCCAGGCAAGAATCCTGGCTGTGGAAAGATACCTAAAGGATCAACAGCTCCTGGGGATTTGGGGTTGCTCTGGAAAACTCATTTGCACCACTGCTGTGCCTTGGAATGCTAGTTGGAGTAATAAATCTCTGGAACAGATTTGGAATCACACGACCTGGATGGAGTGGGACAGAGAAATTAACAATTACACAAGCTTAATACACTCCTTAATTGAAGAATCGCAAAACCAGCAAGAAAAGAATGAACAAGAATTATTGGAATTAGATAAATGGGCAAGTTTGTGGAATTGGTTTAACATAACAAATTGGCTGTGGTATATAAAATTATTCATAATGATAGTAGGAGGCTTGGTAGGTTTAAGAATAGTTTTTGCTGTACTTTCTATAGTGAATAGAGTTAGGCAGGGATATTCACCATTATCGTTTCAGACCCACCTCCCAACCCCGAGGGGACCCGACAGGCCCGAAGGAATAGAAGAAGAAGGTGGAGAGAGAGACAGAGACAGATCCATTCGATTAGTGAACGGATCGGCACTGCGTGCGCCAATTCTGCAGACAAATGGCAGTATTCATCCACAATTTTAAAAGAAAAGGGGGGATTGGGGGGTACAGTGCAGGGGAAAGAATAGTAGACATAATAGCAACAGACATACAAACTAAAGAATTACAAAAACAAATTACAAAAATTCAAAATTTTCGGGTTTATTACAGGGAATGGACTAACTACGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGTAGGCGTGTACGGTGGGAGGCCTATATAAGCAGAGCTGGACTTAGCCTTTAGTGAACCGTCAGAATTAATTCAGATCGATCTACCAGAACCGTCAGATCCGCTAGAGATTACGCCAACCGCCACCATGGGCAGCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGTGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTCAGGCTAATAACAGCTTCCGGACTCTAGAACATCCCTACAGGTGATATCCTCGCTACGTGTGTCAGTTAGGGTGTGGAAAGTCCCCAGGCTCCCCAGCAGGCAGAAGTATGCAAAGCATGCATCTCAATTAGTCAGCAACCAGGTGTGGAAAGTCCCCAGGCTCCCCAGCAGGCAGAAGTATGCAAAGCATGCATCTCAATTAGTCAGCAACCATAGTCCCGCCCCTAACTCCGCCCATCCCGCCCCTAACTCCGCCCAGTTCCGCCCATTCTCCGCCCCATGGCTGACTAATTTTTTTTATTTATGCAGAGGCCGAGGCCGCCTCTGCCTCTGAGCTATTCCAGAAGTAGTGAGGAGGCTTTTTTGGAGGCCTAGGCTTTTGCAAAAAGCTCCCGGGAGCTTGTATATCCATTTTCGGATCTGATCAAGAGACAGGGCAACCGCCACCATGACCGAGTACAAGCCCACGGTGCGCCTCGCCACCCGCGACGACGTCCCCAGGGCCGTACGCACCCTCGCCGCCGCGTTCGCCGACTACCCCGCCACGCGCCACACCGTCGATCCGGACCGCCACATCGAGCGGGTCACCGAGCTGCAAGAACTCTTCCTCACGCGCGTCGGGCTCGACATCGGCAAGGTGTGGGTCGCGGACGACGGCGCCGCGGTGGCGGTCTGGACCACGCCGGAGAGCGTCGAAGCGGGGGCGGTGTTCGCCGAGATCGGCCCGCGCATGGCCGAGTTGAGCGGTTCCCGGCTGGCCGCGCAGCAACAGATGGAAGGCCTCCTGGCGCCGCACCGGCCCAAGGAGCCCGCGTGGTTCCTGGCCACCGTCGGCGTGTCGCCCGACCACCAGGGCAAGGGTCTGGGCAGCGCCGTCGTGCTCCCCGGAGTGGAGGCGGCCGAGCGCGCCGGGGTGCCCGCCTTCCTGGAAACCTCCGCGCCCCGCAACCTCCCCTTCTACGAGCGGCTCGGCTTCACCGTCACCGCCGACGTCGAGGTGCCCGAAGGACCGCGCACCTGGTGCATGACCCGCAAGCCCGGTGCCTAATAACCCTGCTAATAATTCACTCCTCCTGGTCGCCTCTTCATACTCCCTGGATAGTTAAGTCGACAATCAACCTCTGGATTACAAAATTTGTGAAAGATTGACTGGTATTCTTAACTATGTTGCTCCTTTTACGCTATGTGGATACGCTGCTTTAATGCCTTTGTATCATGCTATTGCTTCCCGTATGGCTTTCATTTTCTCCTCCTTGTATAAATCCTGGTTGCTGTCTCTTTATGAGGAGTTGTGGCCCGTTGTCAGGCAACGTGGCGTGGTGTGCACTGTGTTTGCTGACGCAACCCCCACTGGTTGGGGCATTGCCACCACCTGTCAGCTCCTTTCCGGGACTTTCGCTTTCCCCCTCCCTATTGCCACGGCGGAACTCATCGCCGCCTGCCTTGCCCGCTGCTGGACAGGGGCTCGGCTGTTGGGCACTGACAATTCCGTGGTGTTGTCGGGGAAATCATCGTCCTTTCCTTGGCTGCTCGCCTGTGTTGCCACCTGGATTCTGCGCGGGACGTCCTTCTGCTACGTCCCTTCGGCCCTCAATCCAGCGGACCTTCCTTCCCGCGGCCTGCTGCCGGCTCTGCGGCCTCTTCCGCGTCTTCGCCTTCGCCCTCAGACGAGTCGGATCTCCCTTTGGGCCGCCTCCCCGCGTCGACTTTAAGACCAATGACTTACAAGGCAGCTGTAGATCTTAGCCACTTTTTAAAAGAAAAGGGGGGACTGGAAGGGCTAATTCACTCCCAACGAAGACAAGATCTGCTTTTTGCTTGTACTGGGTCTCTCTGGTTAGACCAGATCTGAGCCTGGGAGCTCTCTGGCTAACTAGGGAACCCACTGCTTAAGCCTCAATAAAGCTTGCCTTGAGTGCTTCAAGTAGTGTGTGCCCGTCTGTTGTGTGACTCTGGTAACTAGAGATCCCTCAGACCCTTTTAGTCAGTGTGGAAAATCTCTAGCA"
            NGS_F_start_ind = 1734
            NGS_F_end_ind = 1753

            designed_promoter_start_ind = 1754
            designed_promoter_end_ind = 1754 + 250 - 1

            assert full_construct.find("N") == designed_promoter_start_ind
            assert full_construct.rfind("N") == designed_promoter_end_ind

            EGFP_end_ind = 2910 # defined as end of conn H-N

            # trim off parts that are not relevant for regulation of EGFP
            construct = full_construct[NGS_F_start_ind:EGFP_end_ind + 1]

            # update designed_promoter_start_ind and designed_promoter_end_ind after trimming
            designed_promoter_start_ind = designed_promoter_start_ind - NGS_F_start_ind
            designed_promoter_end_ind = designed_promoter_end_ind - NGS_F_start_ind

            assert construct.find("N") == designed_promoter_start_ind
            assert construct.rfind("N") == designed_promoter_end_ind

            print(f"Length of functional construct to be supplied to models = {len(construct)}")

            base_input = construct

            print(f"Length of final base input to be supplied to models = {len(base_input)}")

            self.merged["sequence"] = self.merged.apply(lambda x: base_input[:designed_promoter_start_ind] + x["sequence"] + base_input[designed_promoter_end_ind+1:], axis=1)

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
        self.train_dataset = FluorescenceDataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                                 cache_dir=self.cache_dir, use_cache=use_cache, use_construct=self.use_construct)
        print("Creating test dataset")
        self.test_dataset = FluorescenceDataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache, use_construct=self.use_construct)
        print("Creating val dataset")
        self.val_dataset = FluorescenceDataset(self.val_set, "val", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache, use_construct=self.use_construct)
        print("Creating full dataset")
        self.full_dataset = FluorescenceDataset(self.merged, "full", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache, use_construct=self.use_construct)
        
        print("Train set has {} promoter-expression pairs from {} total pairs ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                                                           self.merged.shape[0], \
                                                                                                           100.0*self.train_set.shape[0]/self.merged.shape[0]))
        print("Test set has {} promoter-expression data from {} total pairs ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                                                         self.merged.shape[0], \
                                                                                                         100.0*self.test_set.shape[0]/self.merged.shape[0]))
        print("Val set has {} promoter-expression data from {} total pairs ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                                                        self.merged.shape[0], \
                                                                                                        100.0*self.val_set.shape[0]/self.merged.shape[0]))
        print("Full set has {} promoter-expression data from {} total pairs ({:.2f}% of dataset)".format(len(self.full_dataset), \
                                                                                                            self.merged.shape[0], \
                                                                                                            100.0*self.merged.shape[0]/self.merged.shape[0]))
        print("Completed Instantiation of Fluorescence DataLoader")        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def full_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)