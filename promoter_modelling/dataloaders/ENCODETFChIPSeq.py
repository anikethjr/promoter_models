import numpy as np
import pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import joblib
import kipoiseq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import torchmetrics

from promoter_modelling.utils import fasta_utils

np.random.seed(97)
torch.manual_seed(97)

def merge_peaks(all_peaks, fasta_file, region_size, max_peak_offset_for_grouping):
    merged_peaks = {}
    merged_peaks["chromosome"] = []
    merged_peaks["ori_peak_position"] = []
    merged_peaks["all_peaks_in_this_group"] = []
    merged_peaks["all_peak_types_in_this_group"] = []
    merged_peaks["avg_peak_position"] = []
    merged_peaks["extraction_region_start"] = []
    merged_peaks["extraction_region_end"] = []
    merged_peaks["sequence"] = []
    
    fasta_extractor = fasta_utils.FastaStringExtractor(fasta_file)

    for i in range(all_peaks.shape[0]):
        row = all_peaks.iloc[i]
        if len(merged_peaks["ori_peak_position"]) == 0: # corner case for first peak
            # create new peak group
            merged_peaks["chromosome"].append(row["chromosome"])
            merged_peaks["ori_peak_position"].append(row["peak_position"])
            merged_peaks["all_peaks_in_this_group"].append([row["peak_position"]])    
            merged_peaks["all_peak_types_in_this_group"].append([row["name"]])
        elif (row["chromosome"] == merged_peaks["chromosome"][-1]) and (row["peak_position"] >= (merged_peaks["ori_peak_position"][-1] - max_peak_offset_for_grouping)) and (row["peak_position"] <= (merged_peaks["ori_peak_position"][-1] + max_peak_offset_for_grouping)): # add peak to previous peak group if within offsets
            merged_peaks["all_peaks_in_this_group"][-1].append(row["peak_position"])
            merged_peaks["all_peak_types_in_this_group"][-1].append(row["name"])
        else:
            # average peak positions of previous group to finish creation of peak group
            avg_peak_position = (np.around(np.mean(merged_peaks["all_peaks_in_this_group"][-1]))).astype(int)
            merged_peaks["avg_peak_position"].append(avg_peak_position)
            # compute extraction regions                    
            merged_peaks["extraction_region_start"].append((avg_peak_position - (region_size / 2)).astype(int))
            merged_peaks["extraction_region_end"].append((avg_peak_position + (region_size / 2)).astype(int))
            # compute sequence
            merged_peaks["sequence"].append(fasta_utils.get_sequence(merged_peaks["chromosome"][-1], \
                                                                   merged_peaks["extraction_region_start"][-1], \
                                                                   merged_peaks["extraction_region_end"][-1], \
                                                                   ".", fasta_extractor))
            # format peaks
            merged_peaks["all_peaks_in_this_group"][-1] = ",".join([str(j) for j in merged_peaks["all_peaks_in_this_group"][-1]])
            merged_peaks["all_peak_types_in_this_group"][-1] = ",".join([str(j) for j in merged_peaks["all_peak_types_in_this_group"][-1]])

            # create new peak group for current peak
            merged_peaks["chromosome"].append(row["chromosome"])
            merged_peaks["ori_peak_position"].append(row["peak_position"])
            merged_peaks["all_peaks_in_this_group"].append([row["peak_position"]])
            merged_peaks["all_peak_types_in_this_group"].append([row["name"]])

    if len(merged_peaks["chromosome"]) > len(merged_peaks["sequence"]):
        assert len(merged_peaks["sequence"]) == (len(merged_peaks["chromosome"]) - 1)

        # average peak positions of previous group to finish creation of peak group
        avg_peak_position = (np.around(np.mean(merged_peaks["all_peaks_in_this_group"][-1]))).astype(int)
        merged_peaks["avg_peak_position"].append(avg_peak_position)
        # compute extraction regions                    
        merged_peaks["extraction_region_start"].append((avg_peak_position - (region_size / 2)).astype(int))
        merged_peaks["extraction_region_end"].append((avg_peak_position + (region_size / 2)).astype(int))
        # compute sequence
        merged_peaks["sequence"].append(fasta_utils.get_sequence(merged_peaks["chromosome"][-1], \
                                                               merged_peaks["extraction_region_start"][-1], \
                                                               merged_peaks["extraction_region_end"][-1], \
                                                               ".", fasta_extractor))
        # format peaks
        merged_peaks["all_peaks_in_this_group"][-1] = ",".join([str(j) for j in merged_peaks["all_peaks_in_this_group"][-1]])
        merged_peaks["all_peak_types_in_this_group"][-1] = ",".join([str(j) for j in merged_peaks["all_peak_types_in_this_group"][-1]])
        
    return pd.DataFrame(merged_peaks)


def generate_negative_sequences_and_build_all_x(merged_peaks, cache_dir, fasta_shuffle_letters_path):
    # first create FASTA file containing sequences of peak regions
    peaks_fasta_path = os.path.join(cache_dir, "peak_seqs.fasta")
    peaks_fasta = open(peaks_fasta_path, "w+")
    for i in range(merged_peaks.shape[0]):
        row = merged_peaks.iloc[i]
        peaks_fasta.write(">{}:{}-{}\n".format(row["chromosome"], \
                                             row["extraction_region_start"], \
                                             row["extraction_region_end"]))
        peaks_fasta.write(row["sequence"] + "\n")
    peaks_fasta.close()
    
    # run fasta_shuffle_letters to generate negatives
    output_path = os.path.join(cache_dir, "neg_seqs.fasta")
    cmd = "{} -kmer 2 -seed 97 -copies 1 -line 1000000 {} {}".format(fasta_shuffle_letters_path, peaks_fasta_path, output_path)
    os.system(cmd)
    
    # parse output file
    assert os.path.exists(output_path)
    all_seqs = []
    output = open(output_path, "r").readlines()
    for line in output:
        line = line.strip()
        if len(line) == 0 or line.startswith(">"):
            continue
        all_seqs.append(line)
    assert len(all_seqs) == merged_peaks.shape[0]
    
    # build all_x
    all_x = {}
    all_x["chromosome"] = []
    all_x["sequence"] = []
    all_x["all_peak_types_in_this_group"] = []
    all_x["class"] = []
    
    for i in range(merged_peaks.shape[0]):
        row = merged_peaks.iloc[i]
        
        # +ve
        all_x["chromosome"].append(row["chromosome"])
        all_x["sequence"].append(row["sequence"])
        all_x["all_peak_types_in_this_group"].append(row["all_peak_types_in_this_group"])
        all_x["class"].append("+ve")
        
        # -ve
        all_x["chromosome"].append(row["chromosome"])
        all_x["sequence"].append(all_seqs[i])
        all_x["all_peak_types_in_this_group"].append(row["all_peak_types_in_this_group"])
        all_x["class"].append("-ve")
    
    all_x = pd.DataFrame(all_x)
    
    return all_x


# class to create batches for ENCODE TF-ChIP-Seq data
class ENCODETFChIPSeqDataset(Dataset):
    def __init__(self, x, split_name, num_outputs, output_names, shrink_set=False):
        super().__init__()

        self.num_outputs = num_outputs
        self.output_names = output_names
        self.x = x        
        self.num_windows = 1

        if shrink_set:
            self.x = self.x.iloc[:10]
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        row = self.x.iloc[idx]
        seq = fasta_utils.one_hot_encode(row["sequence"]).astype(np.float32)
        targets = np.zeros(self.num_outputs)
        targets[:] = -1
        mask = np.zeros(self.num_outputs, dtype=bool)
        for peak in row["all_peak_types_in_this_group"].split(","):
            peak_ind = self.output_names.index(peak)
            if row["class"] == "+ve":
                targets[peak_ind] = 1
            else:
                targets[peak_ind] = 0
            mask[peak_ind] = True
        
        return seq, targets, mask
    
# class used to read, process and build train, val, test sets using the ENCODE TF-ChIP-Seq datasets
class ENCODETFChIPSeqDataLoader(pl.LightningDataModule):
    def update_metrics(self, y_hat, y, loss, split):
        self.all_metrics[split]["{}_avg_epoch_loss".format(self.name)].update(loss)

        for met in ["Accuracy", "Precision", "Recall", "F1"]:
            self.all_metrics[split]["{}_{}".format(self.name, met)].update(y_hat, y)
    
    def compute_metrics(self, split):
        metrics_dict = self.all_metrics[split].compute()
        self.all_metrics[split].reset()
        return metrics_dict

    def __init__(self, \
                 batch_size, \
                 cache_dir, \
                 common_cache_dir, \
                 datasets_save_dir, \
                 fasta_shuffle_letters_path = "fasta_shuffle_letters", \
                 n_cpus=8, \
                 train_chromosomes = ['1', '3', '5', '6', '7', '8', '11', '12', '14', '15', '16', '18', '19', '22'], \
                 test_chromosomes = ['2', '9', '10', '13', '20', '21'], \
                 val_chromosomes = ['4', '17'], \
                 region_size = 600, \
                 max_peak_offset_for_grouping = 100, \
                 qValueThreshold = 0.05, \
                 min_num_peaks = 1000, \
                 use_cache = True, \
                 shrink_test_set = False):
        super().__init__()
        
        np.random.seed(97)
        torch.manual_seed(97)

        print("Creating ENCODETFChIPSeq DataLoader object")

        self.name = "ENCODETFChIPSeq"

        self.task = "classification"
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.with_mask = True
        
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        self.train_chromosomes = ["chr"+i for i in train_chromosomes]
        self.test_chromosomes = ["chr"+i for i in test_chromosomes]
        self.val_chromosomes = ["chr"+i for i in val_chromosomes]
        
        self.region_size = region_size
        self.max_peak_offset_for_grouping = max_peak_offset_for_grouping
        
        self.promoter_windows_relative_to_TSS = [] # dummy

        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.common_cache_dir = common_cache_dir
        if not os.path.exists(self.common_cache_dir):
            os.mkdir(self.common_cache_dir)
                
        self.datasets_save_dir = datasets_save_dir
        assert os.path.exists(self.datasets_save_dir), "ENCODE TF-ChIP-Seq datasets not found. Please download them first."
        
        self.fasta_file = os.path.join(self.common_cache_dir, "hg38.fa")
        if not os.path.exists(self.fasta_file):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O {}".format(self.fasta_file + ".gz"))
            os.system("gunzip {}".format(self.fasta_file + ".gz"))
            assert os.path.exists(self.fasta_file)            
        self.fasta_extractor = fasta_utils.FastaStringExtractor(self.fasta_file)
        
        # read metadata
        self.metadata = pd.read_csv(os.path.join(self.datasets_save_dir, "metadata.tsv"), sep="\t")
        # remove hg19, keep only hg38
        self.metadata = self.metadata[(self.metadata["File assembly"] == "GRCh38")].reset_index(drop=True)
        # remove duplicates from previous ENCODE analysis
        duplicated_tracks = []
        for grp in self.metadata.groupby("Experiment accession"):
            if grp[1].shape[0] > 1:
                for i in range(grp[1].shape[0]):
                    row = grp[1].iloc[i]
                    if row["File analysis title"] == "ENCODE3 GRCh38":
                        duplicated_tracks.append(row.name)
        self.metadata = self.metadata.drop(duplicated_tracks).reset_index(drop=True)
        assert len(set(self.metadata["Experiment accession"])) == self.metadata.shape[0]
        
        self.metadata["name"] = self.metadata['Biosample term name'] + "_" + self.metadata['Experiment target'] + "_" + self.metadata["File accession"]
        
        print("We have peaks from {} ENCODE datasets".format(self.metadata.shape[0]))
#         print("Distribution of datasets across cells:")
#         # count plot to show the number of samples from each cell
#         fig, ax = plt.subplots(figsize=(10, 10))
#         sns.countplot(data=self.metadata, y = 'Biosample term name')
#         plt.xscale("log")
#         plt.show()

        self.qValueThreshold = qValueThreshold
        self.min_num_peaks = min_num_peaks

        self.all_x_cache_path = os.path.join(self.cache_dir, "all_x.tsv")
        if os.path.exists(self.all_x_cache_path):
            print("Using cached all_x")
            self.all_x = pd.read_csv(self.all_x_cache_path, sep="\t")
            self.all_peaks_cache_path = os.path.join(self.cache_dir, "all_peaks.tsv")
            self.all_peaks = pd.read_csv(self.all_peaks_cache_path, sep="\t")
        else:            
            assert os.path.exists(fasta_shuffle_letters_path), "fasta_shuffle_letters_path not found. Please install fasta_shuffle_letters using the MEME suite and provide its path as an input to this dataloader."

            self.merged_peaks_cache_path = os.path.join(self.cache_dir, "merged_peaks.tsv")
            self.all_peaks_cache_path = os.path.join(self.cache_dir, "all_peaks.tsv")
            if use_cache and os.path.exists(self.merged_peaks_cache_path):
                print("Using cached merged_peaks")
                self.merged_peaks = pd.read_csv(self.merged_peaks_cache_path, sep="\t")
            else:
                if use_cache and os.path.exists(self.all_peaks_cache_path):
                    print("Using cached all_peaks")
                    self.all_peaks = pd.read_csv(self.all_peaks_cache_path, sep="\t")
                else:                
                    self.all_peaks = []

                    print("Reading datasets...")
                    for i in tqdm(range(self.metadata.shape[0])):
                        row = self.metadata.iloc[i]
                        bed_file = os.path.join(self.datasets_save_dir, "{}.bed.gz".format(row["File accession"]))
                        if not os.path.exists(bed_file):
                            print("File doesn't exist! - {}".format(bed_file))
                            continue
                        try:
                            bed = pd.read_csv(bed_file, sep="\t", header=None, names=["chromosome", "start", "end", \
                                                                                  "name", "score", "strand", \
                                                                                  "signalValue", "pValue", "qValue", \
                                                                                  "peak"])
                            bed = bed[bed["qValue"] >= (-np.log10(self.qValueThreshold))].reset_index(drop=True)
                            if bed.shape[0] < self.min_num_peaks:
                                print("Not enough significant peaks for {}".format(bed_file))
                                continue
                            bed["name"] = row['Biosample term name'] + "_" + row['Experiment target'] + "_" + row["File accession"]
                            self.all_peaks.append(bed)
                        except:
                            print("File corrupted - {}".format(bed_file))

                    self.all_peaks = pd.concat(self.all_peaks)
                    self.all_peaks["peak_position"] = self.all_peaks["start"] + self.all_peaks["peak"]
                    self.all_peaks["peak_position_with_chr"] = self.all_peaks["chromosome"] + ":" + self.all_peaks["peak_position"].astype(str)
                    print("Total number of peaks = {}".format(self.all_peaks.shape[0]))
                    print("Total number of unique peak locations = {}".format(len(set(self.all_peaks["peak_position_with_chr"]))))

                    # remove peaks on chromosomes not used in train, val or test sets
                    self.all_peaks["is_train"] = np.isin(self.all_peaks["chromosome"], self.train_chromosomes)
                    self.all_peaks["is_test"] = np.isin(self.all_peaks["chromosome"], self.test_chromosomes)
                    self.all_peaks["is_val"] = np.isin(self.all_peaks["chromosome"], self.val_chromosomes)
                    self.all_peaks["filter_out_due_to_being_on_bad_chromosome"] = np.logical_not(np.logical_or(np.logical_or(self.all_peaks["is_train"], \
                                                                                                                             self.all_peaks["is_test"]),
                                                                                                               self.all_peaks["is_val"]))
                    print("{} peaks are being filtered out because they are on chromosomes not used in train, val or test sets"\
                          .format(self.all_peaks["filter_out_due_to_being_on_bad_chromosome"].sum()))
                    self.all_peaks = self.all_peaks[np.logical_not(self.all_peaks["filter_out_due_to_being_on_bad_chromosome"])].reset_index(drop=True)
                    print("Left with {} total peaks".format(self.all_peaks.shape[0]))
                    print("Left with {} total unique peak locations".format(len(set(self.all_peaks["peak_position_with_chr"]))))

                    print("Number of train peaks = {}".format(self.all_peaks["is_train"].sum()))
                    print("Number of test peaks = {}".format(self.all_peaks["is_test"].sum()))
                    print("Number of val peaks = {}".format(self.all_peaks["is_val"].sum()))

                    # create all_peaks cache
                    self.all_peaks.to_csv(self.all_peaks_cache_path, sep="\t", index=False)
                    print("Cached all_peaks file!")

                # merge nearby peaks
                print("Creating merged_peaks")
                self.all_peaks = self.all_peaks.sort_values(by="peak_position_with_chr").reset_index(drop=True)

                all_chromosomes = self.train_chromosomes + self.test_chromosomes + self.val_chromosomes
                self.merged_peaks = joblib.Parallel(n_jobs=-1)(joblib.delayed(merge_peaks)\
                                                        (self.all_peaks[self.all_peaks["chromosome"] == chrom], \
                                                         self.fasta_file, \
                                                         self.region_size, self.max_peak_offset_for_grouping) \
                                                        for chrom in tqdm(all_chromosomes))

                self.merged_peaks = pd.concat(self.merged_peaks)
                print("Total number of peaks after merging nearby peaks = {}".format(self.merged_peaks.shape[0]))
                self.merged_peaks.to_csv(self.merged_peaks_cache_path, sep="\t", index=False)
                print("Cached merged_peaks file!")
                
            # now generate negative class by shuffling sequences while maintaining dinucleotide freq - based on DeepBind
            # also get all_x
            print("Generating negative sequences and creating all_x")
            self.all_x = generate_negative_sequences_and_build_all_x(self.merged_peaks, self.cache_dir, fasta_shuffle_letters_path)
            self.all_x.to_csv(self.all_x_cache_path, sep="\t", index=False)
            print("Cached all_x file!")
        
        self.output_names = list(set(self.all_peaks["name"]))
        self.num_outputs = len(self.output_names)
        self.use_1hot_for_classification = False
        print("We have sig peaks from {} ENCODE datasets".format(self.num_outputs))

        # specify metrics to track for this dataloader
        self.metrics = torchmetrics.MetricCollection({"{}_Accuracy".format(self.name): torchmetrics.Accuracy(task="multilabel", num_labels=self.num_outputs, ignore_index=-1), \
                                                      "{}_Precision".format(self.name): torchmetrics.Precision(task="multilabel", num_labels=self.num_outputs, ignore_index=-1), \
                                                      "{}_Recall".format(self.name): torchmetrics.Recall(task="multilabel", num_labels=self.num_outputs, ignore_index=-1), \
                                                      "{}_F1".format(self.name): torchmetrics.F1Score(task="multilabel", num_labels=self.num_outputs, ignore_index=-1), \
                                                      "{}_avg_epoch_loss".format(self.name): torchmetrics.MeanMetric()})

        self.all_metrics = {}
        for split in ["train", "val", "test"]:
            self.all_metrics[split] = self.metrics.clone(prefix=split + "_")
            
        self.all_x["is_train"] = np.isin(self.all_x["chromosome"], self.train_chromosomes)
        self.all_x["is_test"] = np.isin(self.all_x["chromosome"], self.test_chromosomes)
        self.all_x["is_val"] = np.isin(self.all_x["chromosome"], self.val_chromosomes)

        self.train_set = self.all_x[self.all_x["is_train"]].reset_index(drop=True)
        self.test_set = self.all_x[self.all_x["is_test"]].reset_index(drop=True)
        self.val_set = self.all_x[self.all_x["is_val"]].reset_index(drop=True)
        
        print("Creating train dataset")
        self.train_dataset = ENCODETFChIPSeqDataset(self.train_set, "train", self.num_outputs, self.output_names)
        
        print("Creating test dataset")
        self.test_dataset = ENCODETFChIPSeqDataset(self.test_set, "test", self.num_outputs, self.output_names, shrink_set=shrink_test_set)
        
        print("Creating val dataset")
        self.val_dataset = ENCODETFChIPSeqDataset(self.val_set, "val", self.num_outputs, self.output_names, shrink_set=shrink_test_set)
        
        total_num_examples = len(self.train_dataset) + len(self.test_dataset) + len(self.val_dataset)
        
        print("Train set has {} examples ({:.2f}% of dataset)".format(len(self.train_dataset), \
                                                                   100.0*len(self.train_dataset)/total_num_examples))
        print("Test set has {} examples ({:.2f}% of dataset)".format(len(self.test_dataset), \
                                                                  100.0*len(self.test_dataset)/total_num_examples))
        print("Validation set has {} examples ({:.2f}% of dataset)".format(len(self.val_dataset), \
                                                                        100.0*len(self.val_dataset)/total_num_examples))
        
        print("Completed Instantiation of ENCODETFChIPSeq DataLoader")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_cpus, pin_memory=True)