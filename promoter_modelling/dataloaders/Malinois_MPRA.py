import numpy as np
import pdb
import pandas as pd
import os
import requests
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

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

class MalinoisMPRADataset(Dataset):
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
            self.all_outputs[cell] = np.array(self.df[cell + "_log2FoldChange"].values, dtype=np.float32)
        
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
            self.df = self.df.iloc[:10]
            self.all_seqs = self.all_seqs[:10]
            self.all_outputs = self.all_outputs[:10]
            self.valid_outputs_mask = self.valid_outputs_mask[:10]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.all_seqs[idx], self.all_outputs[idx], self.valid_outputs_mask[idx]


class MalinoisMPRADataLoader(pl.LightningDataModule):
    def download_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        # from supplementary table 1 of https://doi.org/10.1101/2023.08.08.552077, data for HepG2, K562, GM12878, SK-N-SH, A549
        all_count_file_urls = ["https://www.encodeproject.org/files/ENCFF996ECA", "https://www.encodeproject.org/files/ENCFF018AMJ", "https://www.encodeproject.org/files/ENCFF345ASG", "https://www.encodeproject.org/files/ENCFF970OLE", "https://www.encodeproject.org/files/ENCFF318XMJ", "https://www.encodeproject.org/files/ENCFF821XQZ", "https://www.encodeproject.org/files/ENCFF358MBK", "https://www.encodeproject.org/files/ENCFF379XWL", "https://www.encodeproject.org/files/ENCFF774CHX", "https://www.encodeproject.org/files/ENCFF138DJM", "https://www.encodeproject.org/files/ENCFF277DDE", "https://www.encodeproject.org/files/ENCFF334EKU", "https://www.encodeproject.org/files/ENCFF857FQR", "https://www.encodeproject.org/files/ENCFF259NMG", "https://www.encodeproject.org/files/ENCFF477LDL", "https://www.encodeproject.org/files/ENCFF484JFE", "https://www.encodeproject.org/files/ENCFF227KRF", "https://www.encodeproject.org/files/ENCFF102ZVT", "https://www.encodeproject.org/files/ENCFF418GRL", "https://www.encodeproject.org/files/ENCFF333BAD", "https://www.encodeproject.org/files/ENCFF307HBZ", "https://www.encodeproject.org/files/ENCFF771HPB", "https://www.encodeproject.org/files/ENCFF359KJL", "https://www.encodeproject.org/files/ENCFF035HKU", "https://www.encodeproject.org/files/ENCFF759PPO", "https://www.encodeproject.org/files/ENCFF705AES", "https://www.encodeproject.org/files/ENCFF256WKS", "https://www.encodeproject.org/files/ENCFF352JAC", "https://www.encodeproject.org/files/ENCFF147SMK", "https://www.encodeproject.org/files/ENCFF311DJW", "https://www.encodeproject.org/files/ENCFF350IJA", "https://www.encodeproject.org/files/ENCFF815ORW", "https://www.encodeproject.org/files/ENCFF402GOL", "https://www.encodeproject.org/files/ENCFF865LNO", "https://www.encodeproject.org/files/ENCFF755GRH", "https://www.encodeproject.org/files/ENCFF440YVF", "https://www.encodeproject.org/files/ENCFF703OIL", "https://www.encodeproject.org/files/ENCFF927USI", "https://www.encodeproject.org/files/ENCFF476FXK", "https://www.encodeproject.org/files/ENCFF742ENC", "https://www.encodeproject.org/files/ENCFF112HAT", "https://www.encodeproject.org/files/ENCFF792IHA", "https://www.encodeproject.org/files/ENCFF267VJA"]

        if not os.path.exists(os.path.join(self.cache_dir, "file_info.csv")):
            # Force return from the server in JSON format
            headers = {'accept': 'application/json'}

            data_df = {}
            data_df["cell"] = []
            data_df["dataset_id"] = []
            data_df["element_references_id"] = []
            data_df["element_quantifications_id"] = []

            for url in all_count_file_urls:
                url = url + "/?format=json"

                # GET the count file details
                response = requests.get(url, headers=headers).json()
                dataset_id = response["dataset"]

                data_df["cell"].append(response["biosample_ontology"]["term_name"])
                data_df["dataset_id"].append(dataset_id)

                dataset_url = f"https://www.encodeproject.org/{dataset_id}?format=json"

                # GET the dataset details
                response = requests.get(dataset_url, headers=headers).json()

                # extract element references
                element_references_id = response["elements_references"][0]["files"][0]["accession"]
                data_df["element_references_id"].append(element_references_id)

                # extract element quantifications
                element_quantifications_id = None
                assert_one = False
                for file in response["files"]:
                    if file["output_type"] == "element quantifications" and file["file_format"] == "tsv" and file["analysis_step_version"]["analysis_step"]["pipelines"][0]["title"] == "Tewhey Lab MPRA Pipeline (MPRAmodel)":
                        if "assembly" in file and file["assembly"] == "hg19":
                            continue
                        if assert_one:
                            raise Exception("ERROR: More than one TSV element quantification file found")
                        element_quantifications_id = file["accession"]
                        assert_one = True
                assert assert_one
                data_df["element_quantifications_id"].append(element_quantifications_id)

                # download references and quantifications
                element_references_download_url = f"https://www.encodeproject.org/files/{element_references_id}/@@download/{element_references_id}.fasta.gz"
                element_references_download_path = os.path.join(self.cache_dir, f"{element_references_id}.fasta.gz")
                os.system(f"wget -O {element_references_download_path} {element_references_download_url}")
                os.system(f"gzip -d {element_references_download_path}")
                assert os.path.exists(os.path.join(self.cache_dir, f"{element_references_id}.fasta"))

                element_quantifications_url = f"https://www.encodeproject.org/files/{element_quantifications_id}/@@download/{element_quantifications_id}.tsv"
                element_quantifications_download_path = os.path.join(self.cache_dir, f"{element_quantifications_id}.tsv")
                os.system(f"wget -O {element_quantifications_download_path} {element_quantifications_url}")
                assert os.path.exists(os.path.join(self.cache_dir, f"{element_quantifications_id}.tsv"))

            data_df = pd.DataFrame(data_df)
            data_df.to_csv(os.path.join(self.cache_dir, "file_info.csv"), index=False)
            
        self.data_df = pd.read_csv(os.path.join(self.cache_dir, "file_info.csv"))
        
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
                 test_chromosomes = ["7", "13"], \
                 val_chromosomes = ["19", "21", "X"], \
                 use_cache = True, 
                 shrink_test_set=False):
        super().__init__()

        np.random.seed(97)
        torch.manual_seed(97)
        
        print("Creating Malinois MPRA DataLoader object")

        self.name = "MalinoisMPRA"

        self.task = "regression"
        self.loss_fn = nn.MSELoss()
        self.with_mask = True

        self.batch_size = batch_size
        self.n_cpus = n_cpus

        self.test_chromosomes = test_chromosomes
        self.val_chromosomes = val_chromosomes

        self.promoter_windows_relative_to_TSS = [] # dummy

        self.cache_dir = cache_dir
        self.download_data()

        self.cell_names = ["K562", "HepG2", "GM12878", "SK-N-SH", "A549"]
        self.num_cells = len(self.cell_names)
        self.output_names = self.cell_names
        self.num_outputs = len(self.output_names)

        self.final_dataset_path = os.path.join(self.cache_dir, "final_dataset.tsv")
        
        if not os.path.exists(self.final_dataset_path):
            print("Creating final dataset")

            # create datasets for each cell type
            for cell in self.cell_names:
                print(cell)
                cell_data_df = self.data_df[self.data_df["cell"] == cell].reset_index(drop=True)
                print(f"Num datasets = {cell_data_df.shape[0]}")
                
                cell_element_references_df = {}
                cell_element_references_df["ID"] = []
                cell_element_references_df["sequence"] = []
                    
                for i in range(cell_data_df.shape[0]):
                    row = cell_data_df.iloc[i]
                    element_references_id = row['element_references_id'] 
                    f = open(os.path.join(self.cache_dir, f"{element_references_id}.fasta"), "r").readlines()
                    
                    for line in f:
                        if len(line.strip()) == 0:
                            continue
                        if line.startswith(">"):
                            cell_element_references_df["ID"].append(line[1:].strip())
                        else:
                            cell_element_references_df["sequence"].append(line.strip())
                
                cell_element_references_df = pd.DataFrame(cell_element_references_df).drop_duplicates().reset_index(drop=True)
                print(f"Num element references = {cell_element_references_df.shape[0]}")
                
                cell_element_quantifications_df = {}
                cell_element_quantifications_df["ID"] = []
                cell_element_quantifications_df[f"{cell}_log2FoldChange"] = []
                
                for i in range(cell_data_df.shape[0]):
                    row = cell_data_df.iloc[i]
                    element_quantifications_id = row['element_quantifications_id'] 
                    df = pd.read_csv(os.path.join(self.cache_dir, f"{element_quantifications_id}.tsv"), sep="\t")
                    cell_element_quantifications_df["ID"].extend(df["ID"].tolist())
                    cell_element_quantifications_df[f"{cell}_log2FoldChange"].extend(df["log2FoldChange"].tolist())
                
                cell_element_quantifications_df = pd.DataFrame(cell_element_quantifications_df).drop_duplicates().reset_index(drop=True)
                cell_element_quantifications_df = cell_element_quantifications_df[(~np.isnan(cell_element_quantifications_df[f"{cell}_log2FoldChange"])) & (~np.isinf(cell_element_quantifications_df[f"{cell}_log2FoldChange"]))].drop_duplicates().reset_index(drop=True)
                print(f"Num element quantifications = {cell_element_quantifications_df.shape[0]}")
                
                cell_df = cell_element_quantifications_df.merge(cell_element_references_df, on="ID", how="inner").reset_index(drop=True)
                cell_df = cell_df.groupby(by=["ID", "sequence"]).mean()
                cell_df = pd.DataFrame(cell_df.to_records())
                
                print(f"Final num element quantifications = {cell_df.shape[0]}")
                cell_df.to_csv(os.path.join(self.cache_dir, f"{cell}_data.csv"), index=False)

            # create intersection df containing all sequences that have outputs for all cell types
            print("Creating intersection df")
            intersection_df = None
            for cell in self.cell_names:
                print(cell)
                cell_df = pd.read_csv(os.path.join(self.cache_dir, f"{cell}_data.csv"))
                print(f"Num measurements = {cell_df.shape[0]}")
                if intersection_df is None:
                    intersection_df = cell_df.copy()
                else:
                    intersection_df = intersection_df.merge(cell_df, on=["ID", "sequence"], how="inner").reset_index(drop=True)
                print(f"Num common measurements so far = {intersection_df.shape[0]}")
            intersection_df.to_csv(os.path.join(self.cache_dir, "common_sequences_data.csv"), index=False)
            
            # create union df containing all sequences that have outputs for at least one cell type
            print("Creating union df")
            union_df = None
            for cell in self.cell_names:
                print(cell)
                cell_df = pd.read_csv(os.path.join(self.cache_dir, f"{cell}_data.csv"))
                print(f"Num measurements = {cell_df.shape[0]}")
                if union_df is None:
                    union_df = cell_df.copy()
                else:
                    union_df = union_df.merge(cell_df, on=["ID", "sequence"], how="outer").reset_index(drop=True)
                print(f"Num total sequences so far = {union_df.shape[0]}")
            union_df.to_csv(os.path.join(self.cache_dir, "all_sequences_data.csv"), index=False)

            # create final dataset
            print("Creating final dataset")
            final_dataset = union_df
            final_dataset["chr"] = final_dataset.apply(lambda x: x["ID"].split(":")[0], axis=1)
            final_dataset["length"] = final_dataset.apply(lambda x: len(x["sequence"]), axis=1)
            final_dataset = final_dataset[final_dataset["length"] == 200].reset_index(drop=True) # filter out sequences that are not 200 bp long
            final_dataset["is_val"] = final_dataset.apply(lambda x: x["chr"] in self.val_chromosomes, axis=1)
            final_dataset["is_test"] = final_dataset.apply(lambda x: x["chr"] in self.test_chromosomes, axis=1)
            final_dataset["is_train"] = ~(final_dataset["is_val"] | final_dataset["is_test"])
            final_dataset.to_csv(self.final_dataset_path, sep="\t", index=False)
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
        self.train_dataset = MalinoisMPRADataset(self.train_set, "train", self.num_cells, self.cell_names, \
                                                 cache_dir=self.cache_dir, use_cache=use_cache)
        print("Creating test dataset")
        self.test_dataset = MalinoisMPRADataset(self.test_set, "test", self.num_cells, self.cell_names, \
                                                cache_dir=self.cache_dir, use_cache=use_cache, shrink_set=shrink_test_set)
        print("Creating val dataset")
        self.val_dataset = MalinoisMPRADataset(self.val_set, "val", self.num_cells, self.cell_names, \
                                               cache_dir=self.cache_dir, use_cache=use_cache, shrink_set=shrink_test_set)
        
        num_all_pairs = len(self.train_dataset) + len(self.test_dataset) + len(self.val_dataset)
        
        print("Train set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                                        100.0*self.train_set.shape[0]/num_all_pairs))
        print("Test set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                                        100.0*len(self.test_dataset)/num_all_pairs))
        print("Val set has {} promoter-expression pairs ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                                        100.0*len(self.val_dataset)/num_all_pairs))
        print("Completed Instantiation of Malinois MPRA DataLoader")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)