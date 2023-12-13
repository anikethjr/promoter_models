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
import lightning.pytorch as pl

from promoter_modelling.utils import fasta_utils

import torchmetrics

np.random.seed(97)
torch.manual_seed(97)

# class to create training batches for Roadmap data
class RoadmapTrainDataset(Dataset):
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
    
    
# class to create predict batches for Roadmap data
class RoadmapPredictDataset(Dataset):
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


# class used to read, process and build train, val, test sets using the Roadmap dataset
class RoadmapDataLoader(pl.LightningDataModule):
    def download_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.exists(self.common_cache_dir):
            os.mkdir(self.common_cache_dir)

        self.rnaseq_RPKM_path = os.path.join(self.cache_dir, "57epigenomes.RPKM.pc")
        self.ensembl_gene_info_path = os.path.join(self.common_cache_dir, "ensembl_data.txt")
        self.fasta_file = os.path.join(self.common_cache_dir, "hg38.fa")

        if not os.path.exists(self.rnaseq_RPKM_path):
            os.system("wget https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression/57epigenomes.RPKM.pc.gz -O {}".format(self.rnaseq_RPKM_path + ".gz"))
            os.system("gunzip {}".format(self.rnaseq_RPKM_path + ".gz"))
            assert os.path.exists(self.rnaseq_RPKM_path)

        if not os.path.exists(self.ensembl_gene_info_path):
            os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ufpCW9Qb6r9tDohMIQTcuA-HdlcwjH1C' -O {}".format(self.ensembl_gene_info_path))
            assert os.path.exists(self.ensembl_gene_info_path)

        if not os.path.exists(self.fasta_file):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O {}".format(self.fasta_file + ".gz"))
            os.system("gunzip {}".format(self.fasta_file + ".gz"))
            assert os.path.exists(self.fasta_file)

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

    def extract_promoter_sequences(self):        
        print("Getting all promoter sequences")
        all_sequences = []
        for i in tqdm(range(self.RPKM.shape[0])):
            row = self.RPKM.iloc[i]
            gene_id = row["Gene stable ID"]        
            seqs = fasta_utils.get_promoter_sequences(gene_id, self.promoter_windows_relative_to_TSS, \
                                          self.ensembl_gene_info, self.fasta_extractor)
            all_sequences.append("_".join(seqs))
        self.RPKM["promoter_sequences"] = all_sequences
    
    def __init__(self, \
                 batch_size, \
                 cache_dir, \
                 common_cache_dir, \
                 n_cpus = 0, \
                 train_chromosomes = ['1', '3', '5', '6', '7', '8', '11', '12', '14', '15', '16', '18', '19', '22'], \
                 test_chromosomes = ['2', '9', '10', '13', '20', '21'], \
                 val_chromosomes = ['4', '17'], \
                 RPKM_thres = 1, \
                 zscore = True, \
                 promoter_windows_relative_to_TSS = [[-300, -50], [-100, 150], [100, 350]], \
                 use_cache = True, \
                 return_specified_cells = None):
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)
        
        print("Creating Roadmap DataLoader object")

        self.name = "Roadmap"

        self.task = "regression"
        self.loss_fn = nn.MSELoss()
        self.with_mask = False

        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        cell_type_to_roadmap_id = {"Universal_Human_Reference": "E000",
                            "H1_Cell_Line": "E003",
                            "H1_BMP4_Derived_Mesendoderm_Cultured_Cells": "E004",
                            "H1_BMP4_Derived_Trophoblast_Cultured_Cells": "E005",
                            "H1_Derived_Mesenchymal_Stem_Cells": "E006",
                            "H1_Derived_Neuronal_Progenitor_Cultured_Cells": "E007",
                            "hESC_Derived_CD184+_Endoderm_Cultured_Cells": "E011",
                            "hESC_Derived_CD56+_Ectoderm_Cultured_Cells": "E012",
                            "hESC_Derived_CD56+_Mesoderm_Cultured_Cells": "E013",
                            "HUES64_Cell_Line": "E016",
                            "4star": "E024",
                            "Breast_Myoepithelial_Cells": "E027",
                            "Breast_vHMEC": "E028",
                            "CD4_Memory_Primary_Cells": "E037",
                            "CD4_Naive_Primary_Cells": "E038",
                            "CD8_Naive_Primary_Cells": "E047",
                            "Mobilized_CD34_Primary_Cells_Female": "E050",
                            "Neurosphere_Cultured_Cells_Cortex_Derived": "E053",
                            "Neurosphere_Cultured_Cells_Ganglionic_Eminence_Derived": "E054",
                            "Penis_Foreskin_Fibroblast_Primary_Cells_skin01": "E055",
                            "Penis_Foreskin_Fibroblast_Primary_Cells_skin02": "E056",
                            "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02": "E057",
                            "Penis_Foreskin_Keratinocyte_Primary_Cells_skin03": "E058",
                            "Penis_Foreskin_Melanocyte_Primary_Cells_skin01": "E059",
                            "Penis_Foreskin_Melanocyte_Primary_Cells_skin03": "E061",
                            "Peripheral_Blood_Mononuclear_Primary_Cells": "E062",
                            "Aorta": "E065",
                            "Adult_Liver": "E066",
                            "Brain_Germinal_Matrix": "E070",
                            "Brain_Hippocampus_Middle": "E071",
                            "Esophagus": "E079",
                            "Fetal_Brain_Female": "E082",
                            "Fetal_Intestine_Large": "E084",
                            "Fetal_Intestine_Small": "E085",
                            "Pancreatic_Islets": "E087",
                            "Gastric": "E094",
                            "Left_Ventricle": "E095",
                            "Lung": "E096",
                            "Ovary": "E097",
                            "Pancreas": "E098",
                            "Psoas_Muscle": "E100",
                            "Right_Atrium": "E104",
                            "Right_Ventricle": "E105",
                            "Sigmoid_Colon": "E106",
                            "Small_Intestine": "E109",
                            "Thymus": "E112",
                            "Spleen": "E113",
                            "A549": "E114",
                            "GM12878": "E116",
                            "HELA": "E117",
                            "HEPG2": "E118",
                            "HMEC": "E119",
                            "HSMM": "E120",
                            "HUVEC": "E122",
                            "K562": "E123",
                            "NHEK": "E127",
                            "NHLF": "E128"}
        roadmap_id_to_cell_type = {}
        for key in cell_type_to_roadmap_id:
            roadmap_id_to_cell_type[cell_type_to_roadmap_id[key]] = key
        
        self.train_chromosomes = train_chromosomes
        self.test_chromosomes = test_chromosomes
        self.val_chromosomes = val_chromosomes
        self.zscore = zscore
        self.promoter_windows_relative_to_TSS = promoter_windows_relative_to_TSS
        
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "RoadmapDataset.tsv")
        self.common_cache_dir = common_cache_dir

        # download gene expression and other data if needed
        self.rnaseq_RPKM_path = None
        self.ensembl_gene_info_path = None
        self.fasta_file = None
        self.download_data()
        self.fasta_extractor = fasta_utils.FastaStringExtractor(self.fasta_file)
        
        # read RPKM values
        self.RPKM = pd.read_csv(self.rnaseq_RPKM_path, sep="\t", index_col=False).rename(roadmap_id_to_cell_type, axis=1)
        self.num_cells = self.RPKM.shape[1] - 1
        self.num_outputs = self.num_cells
        self.cell_names = self.RPKM.columns[1:]
        self.output_names = self.cell_names

        print("We have RPKM values of {} genes from {} cells".format(self.RPKM.shape[0], self.num_cells))

        # read Ensembl gene info
        self.ensembl_gene_info = pd.read_csv(self.ensembl_gene_info_path, sep="\t")
        
        if use_cache and os.path.exists(self.cache_path):
            print("Using cached info")
            self.RPKM = pd.read_csv(self.cache_path, sep="\t")
        else:
            # remove genes that have low expression in all cells
            self.RPKM["filter_out_due_to_low_exp"] = self.RPKM.apply(lambda x: x[self.output_names].sum() < RPKM_thres*self.num_cells, \
                                                                    axis=1)
            print("{} genes are being filtered out because they have low expression in all cells"\
                  .format(self.RPKM["filter_out_due_to_low_exp"].sum()))
            self.RPKM = self.RPKM[np.logical_not(self.RPKM["filter_out_due_to_low_exp"])].reset_index(drop=True)
            print("Left with {} genes".format(self.RPKM.shape[0]))

            # convert RPKM values to log2RPKM values. pseudo-count of 1 is used while computing the log
            for col in self.cell_names:
                self.RPKM[col] = np.log2(self.RPKM[col] + 1)

            # only keep protein-coding genes found in Ensembl
            self.RPKM = self.RPKM.merge(self.ensembl_gene_info, how="inner", left_on="gene_id", right_on="Gene stable ID")
            print("Out of these {} are protein-coding genes found in Ensembl".format(self.RPKM.shape[0]))

            # remove genes on chromosomes not used in train, val or test sets
            self.RPKM["filter_out_due_to_being_on_bad_chromosome"] = self.RPKM.apply(lambda x: (x["Chromosome/scaffold name"] not in train_chromosomes) and \
                                                                                   (x["Chromosome/scaffold name"] not in test_chromosomes) and \
                                                                                   (x["Chromosome/scaffold name"] not in val_chromosomes), \
                                                                                   axis=1)
            print("{} genes are being filtered out because they are on chromosomes not used in train, val or test sets"\
                  .format(self.RPKM["filter_out_due_to_being_on_bad_chromosome"].sum()))
            self.RPKM = self.RPKM[np.logical_not(self.RPKM["filter_out_due_to_being_on_bad_chromosome"])].reset_index(drop=True)
            print("Left with {} genes".format(self.RPKM.shape[0]))

            # zscore log2RPKM values if desired
            if self.zscore:
                for cell in self.cell_names:
                    self.RPKM[cell] = stats.zscore(self.RPKM[cell])

            # get all promoter sequences
            self.extract_promoter_sequences()

            # create cache
            self.RPKM.to_csv(self.cache_path, sep="\t", index=False)
        
        self.RPKM["Chromosome/scaffold name"] = self.RPKM["Chromosome/scaffold name"].astype(str)
        self.RPKM["is_train"] = self.RPKM.apply(lambda x: x["Chromosome/scaffold name"] in train_chromosomes, axis=1)
        self.RPKM["is_test"] = self.RPKM.apply(lambda x: x["Chromosome/scaffold name"] in test_chromosomes, axis=1)
        self.RPKM["is_val"] = self.RPKM.apply(lambda x: x["Chromosome/scaffold name"] in val_chromosomes, axis=1)
        
        self.train_set = self.RPKM[self.RPKM["is_train"]].reset_index(drop=True)
        self.test_set = self.RPKM[self.RPKM["is_test"]].reset_index(drop=True)
        self.val_set = self.RPKM[self.RPKM["is_val"]].reset_index(drop=True)
        
        self.num_genes = self.RPKM.shape[0]
        
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
        self.train_dataset = RoadmapTrainDataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                                 cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating test dataset")
        self.test_dataset = RoadmapPredictDataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                  cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating val dataset")
        self.val_dataset = RoadmapPredictDataset(self.val_set, "val", self.num_cells, self.cell_names, \
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
        
        print("Completed Instantiation of Roadmap DataLoader")        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)