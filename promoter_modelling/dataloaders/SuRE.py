import numpy as np
import pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import gc

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


def preprocess_file(file, fasta_file, cur_cache_dir, cur_datasets_save_dir, cur_stats_save_dir):
    print("Pre-processing {}".format(file))
    
    if os.path.exists(os.path.join(cur_cache_dir, file)):
        print("Pre-processed file already exists. Skipping...")
    else:
        df = pd.read_csv(os.path.join(cur_datasets_save_dir, file), sep="\t")

        fasta_extractor = fasta_utils.FastaStringExtractor(fasta_file)

        read_counts_cols = [i for i in df.columns if i.startswith("SuRE")]
        HepG2_cols = sorted([i for i in read_counts_cols if "HEPG2" in i])
        K562_cols = sorted(list(set(read_counts_cols) - set(HepG2_cols)))

        df["avg_K562_exp"] = df[K562_cols].mean(axis=1)
        df["avg_HepG2_exp"] = df[HepG2_cols].mean(axis=1)

        valid_seqs = []
        all_seqs = []
        all_GC_content = []

        ambiguous_bases = ["R", "Y", "M", "K", "S", "W", "H", "B", "V", "D", "N"]
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i]

            check = [b for b in ambiguous_bases if b in str(row["SNPbaseInf"])]
            if len(check) > 0:
                continue            

            valid_seqs.append(i)

            seq = fasta_utils.get_sequence(row["chr"], \
                                           row["start"], \
                                           # the end position is included, so +1 is needed
                                           row["end"] + 1, \
                                           row["strand"], \
                                           fasta_extractor)
            GC_content = float(seq.count("G") + seq.count("C")) / float(len(seq))

            if ("," in str(row["SNPrelpos"])) or (not np.isnan(float(row["SNPrelpos"]))): 
                ori_seq = seq
                all_pos = str(row["SNPrelpos"]).split(",")
                all_bases = str(row["SNPbaseInf"]).split(",")

                ori_len = len(seq)
                for i, pos in enumerate(all_pos):
                    pos = int(float(pos))
                    if row["strand"] == "+":
                        seq = seq[:pos] + all_bases[i] + seq[(pos + 1):]
                    else:
                        if all_bases[i] == "A":
                            c = "T"
                        if all_bases[i] == "T":
                            c = "A"
                        if all_bases[i] == "C":
                            c = "G"
                        if all_bases[i] == "G":
                            c = "C"

                        seq = seq[:(len(seq) - pos - 1)] + all_bases[i] + seq[(len(seq) - pos):]

                assert len(seq) == ori_len

            all_seqs.append(seq)
            all_GC_content.append(GC_content)

        print("Number of seqs = {}".format(df.shape[0]))
        print("Number of valid seqs with unambiguous variant calls = {}".format(len(valid_seqs)))

        df = df.iloc[valid_seqs].reset_index(drop=True)
        df["sequence"] = all_seqs
        df["GC_content"] = all_GC_content
        
        # bin expression values and split promoters into files associated with each bin
        # bins are:
        # 0
        # 1-10
        # 11-20
        # ...
        # 91-100
        # > 100        
        print("Binning expression values")
        for cell in ["K562", "HepG2"]:
            print(cell)
            df["{}_bin".format(cell)] = np.ceil(df["avg_{}_exp".format(cell)] / 10).astype(int)
            df["{}_bin".format(cell)][df["{}_bin".format(cell)] > 10] = 11
            
            for bin_num in range(0, 12):                
                subset = df[df["{}_bin".format(cell)] == bin_num].reset_index(drop=True)
                print("{} {} bin {} samples = {}".format(file, cell, bin_num, subset.shape[0]))
                
        df.to_csv(os.path.join(cur_cache_dir, file), sep="\t", index=False)        

        random_samples = np.random.choice(df.shape[0], 10000, replace=False)

        sns.histplot(data=df.iloc[random_samples], x="GC_content", binwidth=0.05)
        plt.xlim(0, 1)
        plt.savefig(os.path.join(cur_stats_save_dir, "{}_GC_content.png".format(file)))
        plt.clf()

        sns.histplot(data=df.iloc[random_samples], x="avg_K562_exp", binwidth=1)
        plt.xlim(0, 20)
        plt.savefig(os.path.join(cur_stats_save_dir, "{}_avg_K562_exp.png".format(file)))
        plt.clf()

        sns.histplot(data=df.iloc[random_samples], x="avg_HepG2_exp", binwidth=1)
        plt.xlim(0, 20)
        plt.savefig(os.path.join(cur_stats_save_dir, "{}_avg_HepG2_exp.png".format(file)))
        plt.clf()

        sns.countplot(data=df.iloc[random_samples], x="K562_bin")
        plt.savefig(os.path.join(cur_stats_save_dir, "{}_K562_bin.png".format(file)))
        plt.clf()

        sns.countplot(data=df.iloc[random_samples], x="HepG2_bin")
        plt.savefig(os.path.join(cur_stats_save_dir, "{}_HepG2_bin.png".format(file)))
        plt.clf()
        
def get_overall_GC_content_histogram(all_files, cur_cache_dir, cur_stats_save_dir):
    GC_content_bin_width = 0.05
    num_GC_content_bins = int(1.0 / GC_content_bin_width)
    counts_in_each_bin = np.zeros(num_GC_content_bins)
    
    if os.path.exists(os.path.join(cur_stats_save_dir, \
                         "frac_num_promoters_in_each_GC_bin_width_{}.npy".format(GC_content_bin_width))):
        print("Overall GC content histogram already exists.")
        return
    
    for file in all_files:
        print(file)
        df = pd.read_csv(os.path.join(cur_cache_dir, file), sep="\t")
        
        # compute GC content bin
        df["GC_content_bin"] = np.floor(df["GC_content"] / GC_content_bin_width).astype(int)
        
        for i in range(num_GC_content_bins):
            counts_in_each_bin[i] += (df["GC_content_bin"] == i).sum()
    
    frac_in_each_bin = counts_in_each_bin / np.sum(counts_in_each_bin)
    
    np.save(os.path.join(cur_stats_save_dir, \
                         "num_promoters_in_each_GC_bin_width_{}".format(GC_content_bin_width)), \
            counts_in_each_bin)
    
    sns.barplot(x=np.arange(num_GC_content_bins), y=counts_in_each_bin)
    plt.savefig(os.path.join(cur_stats_save_dir, "num_promoters_in_each_GC_bin_width_{}.png".format(GC_content_bin_width)))
    plt.show()
    
    np.save(os.path.join(cur_stats_save_dir, \
                         "frac_num_promoters_in_each_GC_bin_width_{}".format(GC_content_bin_width)), \
            frac_in_each_bin)
    
    sns.barplot(x=np.arange(num_GC_content_bins), y=frac_in_each_bin)
    plt.savefig(os.path.join(cur_stats_save_dir, "frac_num_promoters_in_each_GC_bin_width_{}.png".format(GC_content_bin_width)))
    plt.show()

def bin_promoters_based_on_diff_exp(file, \
                                    cur_cache_dir):
    print(file) 
    binned_values_dir = os.path.join(cur_cache_dir, "diff_exp_binned_values")
    if not os.path.exists(binned_values_dir):
        try:
            os.mkdir(binned_values_dir)
        except:
            print("Dir already created")
    
    if os.path.exists(os.path.join(binned_values_dir, "done_{}".format(file))):
        print("Binned promoters already exist. Skipping...")
        return
            
    df = pd.read_csv(os.path.join(cur_cache_dir, file), sep="\t")
    # combine bins 4-11 because they don't have too many sequences
    for cell in ["K562", "HepG2"]:
        df["{}_bin".format(cell)][df["{}_bin".format(cell)] >= 4] = 4
    df["combined_bins"] = df["K562_bin"].astype(str) + "_" + df["HepG2_bin"].astype(str)
    df["diff_exp_bins"] = df["K562_bin"] - df["HepG2_bin"]
    
    for grp in df.groupby("combined_bins"):
        print("Combined bin {} num promoters = {}".format(grp[0], grp[1].shape))
        grp[1].to_csv(os.path.join(binned_values_dir, "combined_bins_{}_{}".format(grp[0], file)), sep="\t", index=False)
        
    for grp in df.groupby("diff_exp_bins"):
        print("Diff exp bin {} num promoters = {}".format(grp[0], grp[1].shape))
        grp[1].to_csv(os.path.join(binned_values_dir, "diff_exp_bins_{}_{}".format(grp[0], file)), sep="\t", index=False)
        
    done = open(os.path.join(binned_values_dir, "done_{}".format(file)), "w+")
    done.close()
    
def read_file(file):
    return pd.read_csv(file, sep="\t")

def subsample_sequences_based_on_GC_content(all_files, \
                                            num_train, \
                                            num_val, \
                                            num_test, \
                                            cur_cache_dir, \
                                            cur_datasets_save_dir, \
                                            cur_stats_save_dir):
    final_sets_dir = os.path.join(cur_cache_dir, "final_sets")
    if not os.path.exists(final_sets_dir):
        try:
            os.mkdir(final_sets_dir)
        except:
            print()
    binned_values_dir = os.path.join(cur_cache_dir, "diff_exp_binned_values")
    
    # prepare GC content related stuff
    GC_content_bin_width = 0.05
    num_GC_content_bins = int(1.0 / GC_content_bin_width)    
    overall_frac_promoters_in_each_GC_bin = np.load(os.path.join(cur_stats_save_dir, \
                                                                 "frac_num_promoters_in_each_GC_bin_width_{}.npy".format(GC_content_bin_width)))
    
    train_set = []
    val_set = []
    test_set = []
    
    desired_num_promoters_per_exp_bin_train = int(np.floor(num_train / 25))
    desired_num_promoters_per_exp_bin_val = int(np.floor(num_val / 25))
    desired_num_promoters_per_exp_bin_test = int(np.floor(num_test / 25))
    
    for i in range(0, 5):
        for j in range(0, 5):                
            print("Getting subsampled sequences for combined bin K562 {} HepG2 {}".format(i, j))
            
            if os.path.exists(os.path.join(final_sets_dir, "combined_bins_test_set_{}_{}.tsv".format(i, j))):
                train_set.append(pd.read_csv(os.path.join(final_sets_dir, "combined_bins_train_set_{}_{}.tsv".format(i, j)), sep="\t"))
                val_set.append(pd.read_csv(os.path.join(final_sets_dir, "combined_bins_val_set_{}_{}.tsv".format(i, j)), sep="\t"))
                test_set.append(pd.read_csv(os.path.join(final_sets_dir, "combined_bins_test_set_{}_{}.tsv".format(i, j)), sep="\t"))
                print("Used cached subsampled sequences for this bin")
            else:
                promoters_in_exp_bin = []

                all_paths = []
                for file in tqdm(all_files):
                    path_to_promoters = os.path.join(binned_values_dir, "combined_bins_{}_{}_{}".format(i, j, file))
                    if os.path.exists(path_to_promoters):
                        all_paths.append(path_to_promoters)

                promoters_in_exp_bin = Parallel(n_jobs=-1)(delayed(read_file)(file) for file in tqdm(all_paths))
                promoters_in_exp_bin = pd.concat(promoters_in_exp_bin).reset_index(drop=True)

                # compute GC content bin
                promoters_in_exp_bin["GC_content_bin"] = np.floor(promoters_in_exp_bin["GC_content"] / GC_content_bin_width).astype(int)

                num_promoters_in_exp_bin = promoters_in_exp_bin.shape[0]
                print("There are {} promoters in this bin".format(num_promoters_in_exp_bin))

                np.random.seed(97)
                promoters_in_exp_bin = promoters_in_exp_bin.iloc[np.random.permutation(promoters_in_exp_bin.shape[0])]
                
                cur_train_set = []
                cur_val_set = []
                cur_test_set = []
                
                for k in range(overall_frac_promoters_in_each_GC_bin.shape[0]):
                    subset = promoters_in_exp_bin[promoters_in_exp_bin["GC_content_bin"] == k]
                    subset_frac = subset.shape[0] / num_promoters_in_exp_bin

                    print("Number of promoters in GC content bin {} = {} ({}% of total)".format(k, subset.shape[0], subset_frac*100))

                    print("We want this GC bin to be {}% of the total".format(overall_frac_promoters_in_each_GC_bin[k]*100))

                    n_train = int(np.around(overall_frac_promoters_in_each_GC_bin[k] * desired_num_promoters_per_exp_bin_train))
                    n_val = int(np.around(overall_frac_promoters_in_each_GC_bin[k] * desired_num_promoters_per_exp_bin_val))
                    n_test = int(np.around(overall_frac_promoters_in_each_GC_bin[k] * desired_num_promoters_per_exp_bin_test))

                    print("n_train = {}".format(n_train))
                    print("n_val = {}".format(n_val))
                    print("n_test = {}".format(n_test))

                    try:
                        assert (n_train + n_val + n_test) <= subset.shape[0]
                    except:
                        print("Didn't have enough samples, so reducing n_train to {} from {}".format(subset.shape[0] - n_val - n_test,\
                                                                                                     n_train))
                        n_train = subset.shape[0] - n_val - n_test
                        if n_train < 0:
                            print("Still not enough samples, dividing the samples equally among train, test and val sets")
                            n_train = int(np.floor(subset.shape[0] / 3))
                            n_val = int(np.floor(subset.shape[0] / 3))
                            n_test = subset.shape[0] - n_train - n_val

                    cur_train_set.append(subset.iloc[np.arange(0, n_train)])
                    cur_val_set.append(subset.iloc[np.arange(n_train, n_train + n_val)])
                    cur_test_set.append(subset.iloc[np.arange(n_train + n_val, n_train + n_val + n_test)])
                
                cur_train_set = pd.concat(cur_train_set)
                cur_val_set = pd.concat(cur_val_set)
                cur_test_set = pd.concat(cur_test_set)
                
                cur_train_set.to_csv(os.path.join(final_sets_dir, "combined_bins_train_set_{}_{}.tsv".format(i, j)), sep="\t", index=False)
                cur_val_set.to_csv(os.path.join(final_sets_dir, "combined_bins_val_set_{}_{}.tsv".format(i, j)), sep="\t", index=False)
                cur_test_set.to_csv(os.path.join(final_sets_dir, "combined_bins_test_set_{}_{}.tsv".format(i, j)), sep="\t", index=False)
                
                train_set.append(cur_train_set)
                val_set.append(cur_val_set)
                test_set.append(cur_test_set)
                
                promoters_in_exp_bin = []
                del promoters_in_exp_bin
                gc.collect()
                
    train_set = pd.concat(train_set)
    val_set = pd.concat(val_set)
    test_set = pd.concat(test_set)
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_set, x="combined_bins")
    plt.savefig(os.path.join(cur_stats_save_dir, "train_set_combined_bins_distribution.png"))
    plt.show()
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_set, x="GC_content_bin")
    plt.savefig(os.path.join(cur_stats_save_dir, "train_set_GC_content_bin_distribution.png"))
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=val_set, x="combined_bins")
    plt.savefig(os.path.join(cur_stats_save_dir, "val_set_combined_bins_distribution.png"))
    plt.show()
    plt.figure(figsize=(10, 5))
    sns.countplot(data=val_set, x="GC_content_bin")
    plt.savefig(os.path.join(cur_stats_save_dir, "val_set_GC_content_bin_distribution.png"))
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=test_set, x="combined_bins")
    plt.savefig(os.path.join(cur_stats_save_dir, "test_set_combined_bins_distribution.png"))
    plt.show()
    plt.figure(figsize=(10, 5))
    sns.countplot(data=test_set, x="GC_content_bin")
    plt.savefig(os.path.join(cur_stats_save_dir, "test_set_GC_content_bin_distribution.png"))
    plt.show()
    
    train_set.to_csv(os.path.join(final_sets_dir, "combined_bins_train_set.tsv"), sep="\t", index=False)
    val_set.to_csv(os.path.join(final_sets_dir, "combined_bins_val_set.tsv"), sep="\t", index=False)
    test_set.to_csv(os.path.join(final_sets_dir, "combined_bins_test_set.tsv"), sep="\t", index=False)

# class to create batches for SuRE data
class SuREDataset(Dataset):
    def __init__(self, path_to_x, split_name, num_outputs, num_classes_per_output, output_names, task, shrink_set=False):
        super().__init__()

        self.split_name = split_name
        self.num_outputs = num_outputs
        self.num_classes_per_output = num_classes_per_output
        self.output_names = output_names
        self.x = pd.read_csv(path_to_x, sep="\t")        
        self.num_windows = 1
        self.task = task
        
        if shrink_set:
            self.x = self.x.iloc[0:10]
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        row = self.x.iloc[idx]
        seq = fasta_utils.one_hot_encode(row["sequence"]).astype(np.float32)        
        targets = torch.zeros(len(self.output_names)) # even for classification, we only have one output per cell
        
        for i, output in enumerate(self.output_names):
            targets[i] = row[output]
        
        if self.task == "classification":
            targets = targets.long()
        
        return torch.tensor(seq), targets
    
def pad_collate(batch):
    (seq, targets) = zip(*batch)
    seq_lens = [x.shape[0] for x in seq]

    seq = pad_sequence(seq, batch_first=True, padding_value=0)

    return seq, torch.vstack(targets)
            
# class used to read, process and build train, val, test sets using the SuRE 2019 datasets
# From GEO (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128325):
# SuRE-seq assays with 8 libraries obtained from four heterozygous genomes from the 1000-genomes project, 
# each in 2 biological replicates. Each library is transfected in K562, using 3 biological replicates, 
# and in HepG2 cell-lines, using 2 biological replicates.

# Here HG02601, GM18983, HG01241 and HG03464 are genomes (from 4 different individuals) 
# from the 1000-genomes project.
# Data for each genome is in a different subfolder.
# Each subfolder has 1-2 (2 if the chromosome is big, labelled by lib1, lib2) .txt.gz files for each chromosome

# Each file is a TSV - there are columns which describe the SNP (+ its context) tested 
# and there are multiple data columns with read counts (depending on how many replicates were used) - 
# SuRE*_*HEPG2* are for HepG2
# other SuRE*_* columns are for K562
class SuREDataLoader(pl.LightningDataModule):
    def download_data(self):
        download_path = None
        if self.genome_id == "SuRE42_HG02601":
            download_path = "https://files.de-1.osf.io/v1/resources/h3m49/providers/osfstorage/5bdaf1369764d2001958cc9d/?zip="
        elif self.genome_id == "SuRE43_GM18983":
            download_path = "https://files.de-1.osf.io/v1/resources/h3m49/providers/osfstorage/5bdaef10a28e68001ba2e11d/?zip="
        elif self.genome_id == "SuRE44_HG01241":
            download_path = "https://files.de-1.osf.io/v1/resources/h3m49/providers/osfstorage/5bdaf5889764d2001958d39f/?zip="
        elif self.genome_id == "SuRE45_HG03464":
            download_path = "https://files.de-1.osf.io/v1/resources/h3m49/providers/osfstorage/5bdaf3499764d2001a581441/?zip="
        else:
            raise Exception("ERROR: invalid genome specified. Must be one of SuRE42_HG02601, SuRE43_GM18983, SuRE44_HG01241 or SuRE45_HG03464")

        genome_zipped_file_path = os.path.join(self.datasets_save_dir, self.genome_id + ".zip")
        if not os.path.exists(genome_zipped_file_path):
            os.system("wget {} -O {}".format(download_path, genome_zipped_file_path))
            assert os.path.exists(genome_zipped_file_path)

        if not os.path.exists(self.cur_datasets_save_dir):
            os.system("unzip {}".format(genome_zipped_file_path))
            assert os.path.exists(self.cur_datasets_save_dir)

        self.fasta_file = os.path.join(self.common_cache_dir, "hg19.fa")
        if not os.path.exists(self.fasta_file):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz -O {}".format(self.fasta_file + ".gz"))
            os.system("gunzip {}".format(self.fasta_file + ".gz"))
            assert os.path.exists(self.fasta_file)

    def update_metrics(self, y_hat, y, loss, split):
        self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].update(loss)
        if self.task == "classification":
            s = 0
            for i, output in enumerate(self.output_names):
                num_outputs_for_this = self.num_classes_per_output[i]
                for met in ["Accuracy", "Precision", "Recall", "F1"]:
                    self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].update(y_hat[:, s:s+num_outputs_for_this], y[:, i])
                s += num_outputs_for_this
        else:
            for i, output in enumerate(self.output_names):
                for met in ["MSE", "MAE", "R2", "PearsonR", "SpearmanR"]:
                    self.all_metrics[split]["{}_{}_{}".format(self.name, output, met)].update(y_hat[:, i], y[:, i])
    
    def compute_metrics(self, split):
        metrics_dict = {}

        metrics_dict["{}_{}_avg_epoch_loss".format(split, self.name)] = self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].compute()

        if self.task == "classification":
            metrics_set = ["Accuracy", "Precision", "Recall", "F1"]
        else:
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
                 genome_id, \
                 cache_dir, \
                 common_cache_dir, \
                 datasets_save_dir, \
                 task="classification", \
                 n_cpus=8, \
                 num_train=250000*3, \
                 num_val=25000*3, \
                 num_test=25000*3, \
                 use_cache = True, \
                 shrink_test_set=False):

        super().__init__()
        
        np.random.seed(97)
        torch.manual_seed(97)

        print("Creating SuRE DataLoader object using data from {}".format(genome_id))

        self.name = "SuRE_{}".format(genome_id)
        
        self.batch_size = batch_size
        self.genome_id = genome_id
        self.task = task
        self.n_cpus = n_cpus

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        
        self.metrics = None
        if self.task == "classification":
            self.output_names = ["K562_bin", "HepG2_bin"]
            self.num_classes_per_output = [5, 5]
            self.num_outputs = np.sum(self.num_classes_per_output)
            self.use_1hot_for_classification = True

            # specify metrics to track for this dataloader
            self.metrics = {}
            for i, cell in enumerate(self.output_names):
                self.metrics["{}_{}_Accuracy".format(self.name, cell)] = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes_per_output[i])
                self.metrics["{}_{}_Precision".format(self.name, cell)] = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes_per_output[i])
                self.metrics["{}_{}_Recall".format(self.name, cell)] = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes_per_output[i])
                self.metrics["{}_{}_F1".format(self.name, cell)] = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes_per_output[i])
            self.metrics["{}_avg_epoch_loss".format(self.name)] = torchmetrics.MeanMetric()
            self.metrics = torchmetrics.MetricCollection(self.metrics)

        elif self.task == "regression":
            self.output_names = ["avg_K562_exp", "avg_HepG2_exp"]
            self.num_outputs = 2

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

        else:
            raise Exception("task must be one of 'classification' or 'regression'")

        self.all_metrics = {}
        for split in ["train", "val", "test"]:
            self.all_metrics[split] = self.metrics.clone(prefix=split + "_")
        
        self.loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        self.with_mask = False
               
        self.promoter_windows_relative_to_TSS = [] # dummy
        
        # cache_dir is the directory where we will save the preprocessed data
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.cur_cache_dir = os.path.join(self.cache_dir, genome_id)
        if not os.path.exists(self.cur_cache_dir):
            os.mkdir(self.cur_cache_dir)
        self.common_cache_dir = common_cache_dir

        # datasets_save_dir is the directory where we will save the raw data
        self.datasets_save_dir = datasets_save_dir
        if not os.path.exists(self.datasets_save_dir):
            os.mkdir(self.datasets_save_dir)
        self.cur_datasets_save_dir = os.path.join(datasets_save_dir, genome_id)
        
        self.cur_stats_save_dir = os.path.join(self.cur_cache_dir, "stats")
        if not os.path.exists(self.cur_stats_save_dir):
            os.mkdir(self.cur_stats_save_dir)
        
        self.final_sets_dir = os.path.join(self.cur_cache_dir, "final_sets")
        
        self.train_set = os.path.join(self.final_sets_dir, "combined_bins_train_set.tsv")
        self.val_set = os.path.join(self.final_sets_dir, "combined_bins_val_set.tsv")
        self.test_set = os.path.join(self.final_sets_dir, "combined_bins_test_set.tsv")
        
        if (not (os.path.exists(self.train_set) and os.path.exists(self.val_set) and os.path.exists(self.test_set))) or (not use_cache):
            # download gene expression and other data if needed
            self.fasta_file = None
            self.download_data()
            self.fasta_extractor = fasta_utils.FastaStringExtractor(self.fasta_file)

            self.all_files = sorted(os.listdir(self.cur_datasets_save_dir))

            # preprocess each file - uncomment following lines to run in parallel
    #         Parallel(n_jobs=-1)(delayed(preprocess_file)(file, \
    #                                                      self.fasta_file, \
    #                                                      self.cur_cache_dir, \
    #                                                      self.cur_datasets_save_dir, \
    #                                                      self.cur_stats_save_dir) for file in self.all_files)
            for file in self.all_files:
                preprocess_file(file, \
                                self.fasta_file, \
                                self.cur_cache_dir, \
                                self.cur_datasets_save_dir, \
                                self.cur_stats_save_dir)

            get_overall_GC_content_histogram(self.all_files, self.cur_cache_dir, self.cur_stats_save_dir)

            # bin promoters based on differential expression - uncomment following lines to run in parallel
    #         Parallel(n_jobs=3)(delayed(bin_promoters_based_on_diff_exp)(file, \
    #                                                                      self.cur_cache_dir) for file in self.all_files)

            for file in self.all_files:
                bin_promoters_based_on_diff_exp(file, \
                                                self.cur_cache_dir)

            # subsample sequences based on GC content
            subsample_sequences_based_on_GC_content(self.all_files, \
                                                self.num_train, \
                                                self.num_val, \
                                                self.num_test, \
                                                self.cur_cache_dir, \
                                                self.cur_datasets_save_dir, \
                                                self.cur_stats_save_dir)
        
#         self.train_set = pd.read_csv(self.train_set, sep="\t")
#         self.val_set = pd.read_csv(self.val_set, sep="\t")
#         self.test_set = pd.read_csv(self.test_set, sep="\t")
        
#         plt.figure(figsize=(10, 5))
#         sns.countplot(data=self.train_set, x="combined_bins")
#         plt.title("train set combined bins distribution")
#         plt.show()
        
#         plt.figure(figsize=(10, 5))
#         sns.countplot(data=self.train_set, x="GC_content_bin")
#         plt.title("train set GC content bins distribution")
#         plt.show()
        
#         plt.figure(figsize=(10, 5))
#         sns.countplot(data=self.val_set, x="combined_bins")
#         plt.title("val set combined bins distribution")
#         plt.show()
        
#         plt.figure(figsize=(10, 5))
#         sns.countplot(data=self.val_set, x="GC_content_bin")
#         plt.title("val set GC content bins distribution")
#         plt.show()
        
#         plt.figure(figsize=(10, 5))
#         sns.countplot(data=self.test_set, x="combined_bins")
#         plt.title("test set combined bins distribution")
#         plt.show()
        
#         plt.figure(figsize=(10, 5))
#         sns.countplot(data=self.test_set, x="GC_content_bin")
#         plt.title("test set GC content bins distribution")
#         plt.show()
        
#         print("Final size of train set = {}".format(self.train_set.shape))
#         print("Final size of val set = {}".format(self.val_set.shape))
#         print("Final size of test set = {}".format(self.test_set.shape))
        
        print("Creating train dataset")
        self.train_dataset = SuREDataset(self.train_set, "train", self.num_outputs, self.num_classes_per_output, self.output_names, self.task)
        
        print("Creating test dataset")
        self.test_dataset = SuREDataset(self.test_set, "test", self.num_outputs, self.num_classes_per_output, self.output_names, self.task, shrink_set=shrink_test_set)
        
        print("Creating val dataset")
        self.val_dataset = SuREDataset(self.val_set, "val", self.num_outputs, self.num_classes_per_output, self.output_names, self.task, shrink_set=shrink_test_set)
        
        self.num_train = len(self.train_dataset)
        self.num_val = len(self.val_dataset)
        self.num_test = len(self.test_dataset)
        
        total_num_examples = len(self.train_dataset) + len(self.test_dataset) + len(self.val_dataset)
        
        print("Train set has {} examples ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                   100.0*len(self.train_dataset)/total_num_examples))
        print("Test set has {} examples ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                  100.0*len(self.test_dataset)/total_num_examples))
        print("Validation set has {} examples ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                        100.0*len(self.val_dataset)/total_num_examples))
        
        print("Completed Instantiation of SuRE_{} DataLoader".format(genome_id))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True, collate_fn=pad_collate)