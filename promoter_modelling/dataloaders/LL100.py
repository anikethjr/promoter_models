import numpy as np
import pdb
import pandas as pd
import os
import scipy.stats as stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import torchmetrics

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

# class to create training batches for LL-100 data
class LL100TrainDataset(Dataset):
    def __init__(self, gene_df, split_name, num_cells, cell_names, \
                 cache_dir, use_cache=True):
        super().__init__()

        self.num_genes = gene_df.shape[0]
        self.num_cells = num_cells
        self.cell_names = cell_names
        
        # create/load one-hot encoded input sequences
        if use_cache and os.path.exists(os.path.join(cache_dir, "{}_seqs.npy".format(split_name))):
            self.all_seqs = np.load(os.path.join(cache_dir, "{}_seqs.npy".format(split_name)))
            print("Loaded cached one-hot encoded sequences, shape = {}".format(self.all_seqs.shape))
        else:
            print("Creating one-hot encoded sequences")
            self.all_seqs = []
            for i in tqdm(range(self.num_genes)):
                row = gene_df.iloc[i]
                gene_seqs = []
                for promoter_seq in row["promoter_sequences"].split("_"):
                    gene_seqs.append(fasta_utils.one_hot_encode(promoter_seq).astype(np.float32))                    
                self.all_seqs.append(gene_seqs)
            self.all_seqs = np.array(self.all_seqs)
            np.save(os.path.join(cache_dir, "{}_seqs.npy".format(split_name)), self.all_seqs)
            print("Done! Shape = {}".format(self.all_seqs.shape))
        
        # create MTL targets
        self.y = []        
        for cell in self.cell_names:
            self.y.append(gene_df[cell].to_numpy().astype(np.float32))
        self.y = np.array(self.y).T        
        print("Targets shape = {}".format(self.y.shape))
        
        self.num_windows = self.all_seqs.shape[1]
        
        self.gene_identifier = []
        for i in range(self.num_genes):
            gid = gene_df["Gene stable ID"].iloc[i]
            self.gene_identifier.append([])
            for j in range(self.num_windows):
                self.gene_identifier[-1].append(gid + ".seq{}".format(j))
        self.gene_identifier = np.array(self.gene_identifier)
        
    def __len__(self):
        return self.y.shape[0]*self.num_windows

    def __getitem__(self, idx):
        gene_idx = idx // self.num_windows
        window_idx = idx % self.num_windows
                
        return self.all_seqs[gene_idx, window_idx], \
               self.y[gene_idx]
#     , \
#                [self.gene_identifier[gene_idx, window_idx]]
    
    
# class to create predict batches for LL-100 data
class LL100PredictDataset(Dataset):
    def __init__(self, gene_df, split_name, num_cells, cell_names, \
                 cache_dir, use_cache=True):  
        super().__init__()

        self.num_genes = gene_df.shape[0]
        self.num_cells = num_cells
        self.cell_names = cell_names
        
        # create/load one-hot encoded input sequences
        if use_cache and os.path.exists(os.path.join(cache_dir, "{}_seqs.npy".format(split_name))):
            self.all_seqs = np.load(os.path.join(cache_dir, "{}_seqs.npy".format(split_name)))
            print("Loaded cached one-hot encoded sequences, shape = {}".format(self.all_seqs.shape))
        else:
            print("Creating one-hot encoded sequences")
            self.all_seqs = []
            for i in tqdm(range(self.num_genes)):
                row = gene_df.iloc[i]
                gene_seqs = []
                for promoter_seq in row["promoter_sequences"].split("_"):
                    gene_seqs.append(fasta_utils.one_hot_encode(promoter_seq).astype(np.float32))                    
                self.all_seqs.append(gene_seqs)
            self.all_seqs = np.array(self.all_seqs)
            np.save(os.path.join(cache_dir, "{}_seqs.npy".format(split_name)), self.all_seqs)
            print("Done! Shape = {}".format(self.all_seqs.shape))
        
        # create MTL targets
        self.y = []        
        for cell in self.cell_names:
            self.y.append(gene_df[cell].to_numpy().astype(np.float32))
        self.y = np.array(self.y).T        
        print("Targets shape = {}".format(self.y.shape))
        
        self.num_windows = self.all_seqs.shape[1]
        
        self.gene_identifier = []
        for i in range(self.num_genes):
            gid = gene_df["Gene stable ID"].iloc[i]
            self.gene_identifier.append([])
            for j in range(self.num_windows):
                self.gene_identifier[-1].append(gid + ".seq{}".format(j))
        self.gene_identifier = np.array(self.gene_identifier)
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):               
        return self.all_seqs[idx], \
               self.y[idx]
#     , \
#                [self.gene_identifier[idx]]


# class used to read, process and build train, val, test sets using the LL-100 dataset
class LL100DataLoader(pl.LightningDataModule):
    def download_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.exists(self.common_cache_dir):
            os.mkdir(self.common_cache_dir)

        self.rnaseq_TPM_path = os.path.join(self.cache_dir, "rsem.merged.gene_tpm.tsv")
        self.ensembl_gene_info_path = os.path.join(self.common_cache_dir, "ensembl_data.txt")
        self.fasta_file = os.path.join(self.common_cache_dir, "hg38.fa")

        if not os.path.exists(self.rnaseq_TPM_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=12_WFTVbkm4gLp-qE1nPK-Jb8tSKX4giD' -O {}".format(self.rnaseq_TPM_path))
            assert os.path.exists(self.rnaseq_TPM_path)

        if not os.path.exists(self.ensembl_gene_info_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ufpCW9Qb6r9tDohMIQTcuA-HdlcwjH1C' -O {}".format(self.ensembl_gene_info_path))
            assert os.path.exists(self.ensembl_gene_info_path)

        if not os.path.exists(self.fasta_file):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O {}".format(self.fasta_file + ".gz"))
            os.system("gunzip {}".format(self.fasta_file + ".gz"))
            assert os.path.exists(self.fasta_file)

    def extract_promoter_sequences(self):        
        print("Getting all promoter sequences")
        all_sequences = []
        for i in tqdm(range(self.TPM.shape[0])):
            row = self.TPM.iloc[i]
            gene_id = row["Gene stable ID"]        
            seqs = fasta_utils.get_promoter_sequences(gene_id, self.promoter_windows_relative_to_TSS, \
                                                      self.ensembl_gene_info, self.fasta_extractor)
            all_sequences.append("_".join(seqs))
        self.TPM["promoter_sequences"] = all_sequences

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
                 common_cache_dir, \
                 n_cpus = 0, \
                 train_chromosomes = ['1', '3', '5', '6', '7', '8', '11', '12', '14', '15', '16', '18', '19', '22'], \
                 test_chromosomes = ['2', '9', '10', '13', '20', '21'], \
                 val_chromosomes = ['4', '17'], \
                 tpm_thres = 1, \
                 zscore = True, \
                 promoter_windows_relative_to_TSS = [[-300, -50], [-100, 150], [100, 350]], \
                 use_cache = True, \
                 return_specified_cells = None):
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)
        
        print("Creating LL-100 DataLoader object")

        self.name = "LL100"

        self.task = "regression"
        self.loss_fn = nn.MSELoss()
        self.with_mask = False

        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        self.train_chromosomes = train_chromosomes
        self.test_chromosomes = test_chromosomes
        self.val_chromosomes = val_chromosomes

        self.zscore = zscore
        self.promoter_windows_relative_to_TSS = promoter_windows_relative_to_TSS
        
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "LL100Dataset.tsv")
        self.common_cache_dir = common_cache_dir

        # download gene expression and other data if needed
        self.rnaseq_TPM_path = None
        self.ensembl_gene_info_path = None
        self.fasta_file = None
        self.download_data()
        self.fasta_extractor = fasta_utils.FastaStringExtractor(self.fasta_file)
        
        # read TPM values
        self.TPM = pd.read_csv(self.rnaseq_TPM_path, sep="\t")
        self.num_cells = self.TPM.shape[1] - 2
        self.num_outputs = self.num_cells
        self.cell_names = self.TPM.columns[2:]
        self.output_names = self.cell_names

        print("We have TPM values of {} genes from {} cells".format(self.TPM.shape[0], self.num_cells))
        
        # read Ensembl gene info
        self.ensembl_gene_info = pd.read_csv(self.ensembl_gene_info_path, sep="\t")
        
        if use_cache and os.path.exists(self.cache_path):
            print("Using cached info")
            self.TPM = pd.read_csv(self.cache_path, sep="\t")
        else:
            # remove genes that have low expression in all cells
            self.TPM["filter_out_due_to_low_exp"] = self.TPM.apply(lambda x: x[self.TPM.columns[2:]].sum() < tpm_thres*self.num_cells, \
                                                                    axis=1)
            print("{} genes are being filtered out because they have low expression in all cells"\
                  .format(self.TPM["filter_out_due_to_low_exp"].sum()))
            self.TPM = self.TPM[np.logical_not(self.TPM["filter_out_due_to_low_exp"])].reset_index(drop=True)
            print("Left with {} genes".format(self.TPM.shape[0]))

            # convert TPM values to log2TPM values. pseudo-count of 1 is used while computing the log
            for col in self.cell_names:
                self.TPM[col] = np.log2(self.TPM[col] + 1)

            # only keep protein-coding genes found in Ensembl
            self.TPM = self.TPM.merge(self.ensembl_gene_info, how="inner", left_on="gene_id", right_on="Gene name")
            print("Out of these {} are protein-coding genes found in Ensembl".format(self.TPM.shape[0]))

            # remove genes on chromosomes not used in train, val or test sets
            self.TPM["filter_out_due_to_being_on_bad_chromosome"] = self.TPM.apply(lambda x: (x["Chromosome/scaffold name"] not in train_chromosomes) and \
                                                                                             (x["Chromosome/scaffold name"] not in test_chromosomes) and \
                                                                                             (x["Chromosome/scaffold name"] not in val_chromosomes), \
                                                                                   axis=1)
            print("{} genes are being filtered out because they are on chromosomes not used in train, val or test sets"\
                  .format(self.TPM["filter_out_due_to_being_on_bad_chromosome"].sum()))
            self.TPM = self.TPM[np.logical_not(self.TPM["filter_out_due_to_being_on_bad_chromosome"])].reset_index(drop=True)
            print("Left with {} genes".format(self.TPM.shape[0]))

            # zscore log2TPM values if desired
            if self.zscore:
                for cell in self.cell_names:
                    self.TPM[cell] = stats.zscore(self.TPM[cell])

            # get all promoter sequences
            self.extract_promoter_sequences()

            # create cache
            self.TPM.to_csv(self.cache_path, sep="\t", index=False)
        
        self.TPM["Chromosome/scaffold name"] = self.TPM["Chromosome/scaffold name"].astype(str)
        self.TPM["is_train"] = self.TPM.apply(lambda x: x["Chromosome/scaffold name"] in train_chromosomes, axis=1)
        self.TPM["is_test"] = self.TPM.apply(lambda x: x["Chromosome/scaffold name"] in test_chromosomes, axis=1)
        self.TPM["is_val"] = self.TPM.apply(lambda x: x["Chromosome/scaffold name"] in val_chromosomes, axis=1)
        
        self.train_set = self.TPM[self.TPM["is_train"]].reset_index(drop=True)
        self.test_set = self.TPM[self.TPM["is_test"]].reset_index(drop=True)
        self.val_set = self.TPM[self.TPM["is_val"]].reset_index(drop=True)
        
        self.num_genes = self.TPM.shape[0]
        
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
        self.train_dataset = LL100TrainDataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating test dataset")
        self.test_dataset = LL100PredictDataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating val dataset")
        self.val_dataset = LL100PredictDataset(self.val_set, "val", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache)
        
        print("Train set has {} promoter-expression pairs from {} genes ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                                                  self.train_set.shape[0], \
                                                                                                  100.0*self.train_set.shape[0]/self.num_genes))
        print("Test set has {} promoter-expression data (not split among different promoter windows) from {} genes ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                                                                                             self.test_set.shape[0], \
                                                                                                                                             100.0*self.test_set.shape[0]/self.num_genes))
        print("Val set has {} promoter-expression data (not split among different promoter windows) from {} genes ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                                                                                            self.val_set.shape[0], \
                                                                                                                                            100.0*self.val_set.shape[0]/self.num_genes))
        
        print("Completed Instantiation of LL-100 DataLoader")        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)