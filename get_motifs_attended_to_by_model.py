import numpy as np
import pandas as pd
import os
import pdb
import argparse
import json
from tqdm import tqdm
import scipy.stats as stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from promoter_modelling.dataloaders import FluorescenceData, LL100, CCLE, Roadmap, Sharpr_MPRA, SuRE, ENCODETFChIPSeq
from promoter_modelling import backbone_modules
from promoter_modelling import MTL_modules
from promoter_modelling.utils import fasta_utils
from promoter_modelling.utils import misc_utils
from promoter_modelling.utils import plotting

np.random.seed(97)
torch.manual_seed(97)

args = argparse.ArgumentParser()
args.add_argument("--config_path", type=str, default="./config.json", help="Path to config file")
args.add_argument("--modelling_strategy", type=str, required=True, help="Modelling strategy to use, either 'joint', 'pretrain+finetune', 'pretrain+linear_probing' or 'single_task'")

args.add_argument("--joint_tasks", type=str, default=None, help="Comma separated list of tasks to jointly train on")
args.add_argument("--pretrain_tasks", type=str, default=None, help="Comma separated list of tasks to pretrain on")
args.add_argument("--finetune_tasks", type=str, default=None, help="Comma separated list of tasks to finetune or perform linear probing on")
args.add_argument("--single_task", type=str, default=None, help="Task to train on")

args.add_argument("--top_n_motifs_to_print", type=int, default=50, help="Number of top motifs to print for each cell")

args = args.parse_args()

assert os.path.exists(args.config_path), "Config file does not exist"
# Load config file
with open(args.config_path, "r") as f:
    config = json.load(f)

# directories needed
root_dir = config["root_dir"]
motif_insertion_analysis_dir = os.path.join(root_dir, "motif_insertion_analysis")
root_data_dir = config["root_data_dir"]
common_cache_dir = os.path.join(root_data_dir, "common")

# download motif occurrences file if not already downloaded
path_to_motif_occurrences_file = os.path.join(common_cache_dir, "motif_occurrences.tsv")
if not os.path.exists(path_to_motif_occurrences_file):
    os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1AL431APt7AYG0fpgZlLCF9neUW1OTUYr' -O {}".format(path_to_motif_occurrences_file))
    assert os.path.exists(path_to_motif_occurrences_file), "Failed to download motif occurrences file"

# download t-test results file if not already downloaded
path_to_ttest_results_file = os.path.join(common_cache_dir, "ttest_results.csv")
if not os.path.exists(path_to_ttest_results_file):
    os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1y4CHmxMNpPzpSgrIht6rA_hcF3eoJm9z' -O {}".format(path_to_ttest_results_file))
    assert os.path.exists(path_to_ttest_results_file), "Failed to download t-test results file"

# download motifs meme file if not already downloaded
path_to_motifs_meme_file = os.path.join(common_cache_dir, "all_motifs.meme")
if not os.path.exists(path_to_motifs_meme_file):
    os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1v2H_bYnacPlWSWbqpT3XBnEQzAOLSnZu' -O {}".format(path_to_motifs_meme_file))
    assert os.path.exists(path_to_motifs_meme_file), "Failed to download motifs meme file"

# needed for name format
if args.modelling_strategy == "joint":
    assert args.joint_tasks is not None, "Must specify tasks to jointly train on"
    tasks = args.joint_tasks.split(",")
elif args.modelling_strategy == "pretrain+finetune" or args.modelling_strategy == "pretrain+linear_probing":
    assert args.pretrain_tasks is not None, "Must specify tasks to pretrain on"
    assert args.finetune_tasks is not None, "Must specify tasks to finetune or perform linear probing on"
    pretrain_tasks = args.pretrain_tasks.split(",")
    finetune_tasks = args.finetune_tasks.split(",")
    tasks = finetune_tasks
elif args.modelling_strategy == "single_task":
    assert args.single_task is not None, "Must specify task to train on"
    tasks = [args.single_task]
else:
    raise ValueError("Invalid modelling strategy")

# model name format
name_format = ""
if "pretrain" in args.modelling_strategy:
    if "finetune" in args.modelling_strategy:
        name_format = "finetune_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
    if "linear_probing" in args.modelling_strategy:
        name_format = "linear_probing_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
elif "joint" in args.modelling_strategy:
    name_format = "joint_train_on_{}".format("+".join(tasks))
elif "single" in args.modelling_strategy:
    name_format = "individual_training_on_{}".format("+".join(tasks))

# load motif insertion analysis results
motif_insertion_analysis_results = pd.read_csv(os.path.join(motif_insertion_analysis_dir, "{}.tsv".format(name_format)), sep="\t")

# load t-test results
ttest_results = pd.read_csv(path_to_ttest_results_file)

# get names of all motifs
all_motifs = []
for col in motif_insertion_analysis_results.columns:
    if col.startswith("JURKAT_motif_insertion_prediction_for_"):
        all_motifs.append(col[len("JURKAT_motif_insertion_prediction_for_"):])
print("Number of motifs which were inserted = {}".format(len(all_motifs)))

# summarize insertion predictions for each motif
if not os.path.exists(os.path.join(motif_insertion_analysis_dir, "{}_summary.tsv".format(name_format))):
    print("Summarizing insertion predictions for each motif")
    
    # load motif occurrences
    motif_occurrences = pd.read_csv(path_to_motif_occurrences_file, sep="\t")

    # merge motif insertion analysis results with motif occurrences
    motif_insertion_analysis_results = motif_insertion_analysis_results.merge(motif_occurrences, on="sequence", how="inner")
    
    summary_df = {}
    summary_df["motif"] = []
    summary_df["num_seqs_without_motif"] = []
    for cell in ["JURKAT", "K562", "THP1"]:
        summary_df["{}_avg_pred".format(cell)] = []

    for motif in tqdm(all_motifs):
        summary_df["motif"].append(motif)
        
        # get sequences without motif
        subset_without_motif = motif_insertion_analysis_results[motif_insertion_analysis_results[motif] == 0].reset_index(drop=True)
        summary_df["num_seqs_without_motif"].append(subset_without_motif.shape[0])
        
        if summary_df["num_seqs_without_motif"] == 0:
            print("{} is present in every sequence, ignoring motif".format(motif))
        
        # subtract the predictions for the sequences without the motif from the predictions for the sequences with the motif
        # then take the average of the difference over all sequences to get the average effect of the motif
        for cell in ["JURKAT", "K562", "THP1"]:
            summary_df["{}_avg_pred".format(cell)].append((motif_insertion_analysis_results["{}_motif_insertion_prediction_for_{}".format(cell, motif)] \
                                                        - motif_insertion_analysis_results["baseline_{}_prediction".format(cell)]).mean())
            
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(motif_insertion_analysis_dir, "{}_summary.tsv".format(name_format)), sep="\t", index=False)
else:
    print("Loading summary from file")
    summary_df = pd.read_csv(os.path.join(motif_insertion_analysis_dir, "{}_summary.tsv".format(name_format)), sep="\t")

# drop HOMER motifs
summary_df = summary_df[~summary_df["motif"].str.contains("HOMER")].reset_index(drop=True)

# get top up/down regulated motifs for each cell type
up_regulated_motifs = {}
down_regulated_motifs = {}

# create subplots for each cell type
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, cell in enumerate(["JURKAT", "K562", "THP1"]):
    print(cell)
    first_letter = cell[0]

    cell_sorted_summary = summary_df.sort_values(by="{}_avg_pred".format(cell), ascending=False)
    print("Top {} up-regulated motifs: {}".format(args.top_n_motifs_to_print, cell_sorted_summary["motif"].values[:args.top_n_motifs_to_print]))
    print("Top {} down-regulated motifs: {}".format(args.top_n_motifs_to_print, cell_sorted_summary["motif"].values[-args.top_n_motifs_to_print:]))

    up_regulated_motifs[cell] = cell_sorted_summary["motif"].values[:args.top_n_motifs_to_print]
    down_regulated_motifs[cell] = cell_sorted_summary["motif"].values[-args.top_n_motifs_to_print:]

    # print the spearman correlation between the average insertion prediction and the dFE for motifs that are significantly up/down-regulated in the cell type    
    un = summary_df.merge(ttest_results[ttest_results["{}_q_val".format(first_letter)] < 0.05], \
                          left_on="motif", right_on="Motif", how="inner")
    srho = stats.spearmanr(un["{}_avg_pred".format(cell)], un["{}_dFE".format(first_letter)])
    print("{} SpearmanR between prediction and sig dFE = {}, num sig dFE motifs = {}".format(cell, \
                                                                                             srho, \
                                                                                             un.shape[0]))
    print()

    # plot the average insertion prediction vs. the dFE for motifs that are significantly up/down-regulated in the cell type
    sns.scatterplot(x="{}_avg_pred".format(cell), y="{}_dFE".format(first_letter), data=un, ax=axes[i])

    # label axes
    axes[i].set_xlabel("Motif Influence")
    axes[i].set_ylabel("Mean expression of sequences with motif - \nMean expression of sequences without motif")

    axes[i].set_title(cell + r" (num motifs = {}, $\rho$ = {:.4f})".format(un.shape[0], srho.correlation))

# set suptitle and save figure
fig.suptitle("Motif influence vs. difference in mean expression with and without the motif")
fig.savefig(os.path.join(motif_insertion_analysis_dir, "{}_motif_influence_vs_dFE.png".format(name_format)), bbox_inches="tight")

# create subplots for each cell type - all motifs, including those that are not significantly up/down-regulated
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, cell in enumerate(["JURKAT", "K562", "THP1"]):
    print(cell)
    first_letter = cell[0]

    cell_sorted_summary = summary_df.sort_values(by="{}_avg_pred".format(cell), ascending=False)

    up_regulated_motifs[cell] = cell_sorted_summary["motif"].values[:args.top_n_motifs_to_print]
    down_regulated_motifs[cell] = cell_sorted_summary["motif"].values[-args.top_n_motifs_to_print:]

    # print the spearman correlation between the average insertion prediction and the dFE for all motifs
    un = summary_df.merge(ttest_results, \
                          left_on="motif", right_on="Motif", how="inner")
    # label each motif as significantly up/down-regulated or not
    un["Significant"] = un["{}_q_val".format(first_letter)] < 0.05

    srho = stats.spearmanr(un["{}_avg_pred".format(cell)], un["{}_dFE".format(first_letter)])
    print("{} SpearmanR between prediction and dFE = {}, num motifs = {}".format(cell, \
                                                                                 srho, \
                                                                                 un.shape[0]))
    print()

    # plot the average insertion prediction vs. the dFE for motifs that are significantly up/down-regulated in the cell type
    sns.scatterplot(x="{}_avg_pred".format(cell), y="{}_dFE".format(first_letter), data=un, ax=axes[i], hue="Significant")

    # label axes
    axes[i].set_xlabel("Motif Influence")
    axes[i].set_ylabel("Mean expression of sequences with motif - \nMean expression of sequences without motif")

    axes[i].set_title(cell + r" (num motifs = {}, $\rho$ = {:.4f})".format(un.shape[0], srho.correlation))

# set suptitle and save figure
fig.suptitle("Motif influence vs. difference in mean expression with and without the motif")
fig.savefig(os.path.join(motif_insertion_analysis_dir, "{}_motif_influence_vs_dFE_include_unsig.png".format(name_format)), bbox_inches="tight")


# get motifs that are common to all three cell types
common_upreg_motifs = set(up_regulated_motifs["JURKAT"]).intersection(set(up_regulated_motifs["K562"])).intersection(set(up_regulated_motifs["THP1"]))
common_downreg_motifs = set(down_regulated_motifs["JURKAT"]).intersection(set(down_regulated_motifs["K562"])).intersection(set(down_regulated_motifs["THP1"]))

print("Upregulated motifs common to all three cell types: {}".format(common_upreg_motifs))
print("Downregulated motifs common to all three cell types: {}".format(common_downreg_motifs))

# plot motifs' PWMs
all_motifs_info = misc_utils.parse_meme_file(path_to_motifs_meme_file)
print("Read all {} motifs from file".format(len(all_motifs_info)))

# create dictionary of motif name to motif info
motif_dict = {}
for m in all_motifs_info:
    motif_dict[m.motif_name] = m

# first plot the motifs that are common to all three cell types
num = len(common_upreg_motifs)
fig, axes = plt.subplots(num, 1, figsize=(10, 2*num))
if num == 1:
    axes = [axes]
# compute the strength of common motifs as the average of the strength of the motif in each cell type
common_upreg_motifs = list(common_upreg_motifs)
common_upreg_motifs_strength = []
for motif in common_upreg_motifs:
    strength = 0
    for cell in ["JURKAT", "K562", "THP1"]:
        strength += summary_df[summary_df["motif"] == motif]["{}_avg_pred".format(cell)].values[0]
    common_upreg_motifs_strength.append(strength / 3)
# sort the motifs by their strength
common_upreg_motifs = [x for _, x in sorted(zip(common_upreg_motifs_strength, common_upreg_motifs), reverse=True)]

for i, motif in enumerate(common_upreg_motifs):
    plotting.seqlogo(motif_dict[motif].pwm, ax=axes[i])
    axes[i].axis('off')
    motif = motif.split("[")[0]
    axes[i].set_title(motif, fontsize=30)
plt.tight_layout()
plt.savefig(os.path.join(motif_insertion_analysis_dir, "{}_common_upreg_motifs.png".format(name_format)), dpi=300)

num = len(common_downreg_motifs)
fig, axes = plt.subplots(num, 1, figsize=(10, 2*num))
if num == 1:
    axes = [axes]
# compute the strength of common motifs as the average of the strength of the motif in each cell type
common_downreg_motifs = list(common_downreg_motifs)
common_downreg_motifs_strength = []
for motif in common_downreg_motifs:
    strength = 0
    for cell in ["JURKAT", "K562", "THP1"]:
        strength += summary_df[summary_df["motif"] == motif]["{}_avg_pred".format(cell)].values[0]
    common_downreg_motifs_strength.append(strength / 3)
# sort the motifs by their strength
common_downreg_motifs = [x for _, x in sorted(zip(common_downreg_motifs_strength, common_downreg_motifs))]

for i, motif in enumerate(common_downreg_motifs):
    plotting.seqlogo(motif_dict[motif].pwm, ax=axes[i])
    axes[i].axis('off')
    motif = motif.split("[")[0]
    axes[i].set_title(motif, fontsize=30)
plt.tight_layout()
plt.savefig(os.path.join(motif_insertion_analysis_dir, "{}_common_downreg_motifs.png".format(name_format)), dpi=300)

# differentially expressed motifs

# first z-score the avg preds
for cell in ["JURKAT", "K562", "THP1"]:
    summary_df["{}_avg_pred_zscore".format(cell)] = stats.zscore(summary_df["{}_avg_pred".format(cell)])

# then compute the difference in the z-score of the motif in the cell type of interest vs the maximum z-score of the motif in the other cell types
for cell in ["JURKAT", "K562", "THP1"]:
    other_cells = set(["JURKAT", "K562", "THP1"]).difference(set([cell]))

    # keep only +ve motifs
    subset = summary_df[summary_df["{}_avg_pred".format(cell)] > 0].copy().reset_index(drop=True)

    # compute the difference in the z-score of the motif in the cell type of interest vs the maximum z-score of the motif in the other cell types
    subset["{}_max_zscore_others".format(cell)] = -np.inf
    for other_cell in other_cells:
        subset["{}_max_zscore_others".format(cell)] = np.maximum(subset["{}_max_zscore_others".format(cell)], subset["{}_avg_pred_zscore".format(other_cell)])

    subset["{}_diff_zscore".format(cell)] = subset["{}_avg_pred_zscore".format(cell)] - subset["{}_max_zscore_others".format(cell)]

    # sort motifs by the difference in the z-score of the motif in the cell type of interest vs the maximum z-score of the motif in the other cell types
    subset = subset.sort_values(by="{}_diff_zscore".format(cell), ascending=False).reset_index(drop=True)

    # print the top motifs
    print("Top upregulated motifs in {}".format(cell))
    print(subset.head(args.top_n_motifs_to_print)[["motif", "{}_avg_pred".format(cell), "{}_avg_pred_zscore".format(cell), "{}_max_zscore_others".format(cell), "{}_diff_zscore".format(cell)]])

    # plot them
    num = args.top_n_motifs_to_print
    fig, axes = plt.subplots(num, 1, figsize=(10, 2*num))
    if num == 1:
        axes = [axes]
    for i, motif in enumerate(subset.head(num)["motif"]):
        plotting.seqlogo(motif_dict[motif].pwm, ax=axes[i])
        axes[i].axis('off')
        motif = motif.split("[")[0]
        axes[i].set_title(motif, fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(motif_insertion_analysis_dir, "{}_{}_upreg_motifs.png".format(name_format, cell)), dpi=300)

    # keep only -ve motifs
    subset = summary_df[summary_df["{}_avg_pred".format(cell)] < 0].copy().reset_index(drop=True)

    # compute the difference in the z-score of the motif in the cell type of interest vs the minimum z-score of the motif in the other cell types
    subset["{}_min_zscore_others".format(cell)] = np.inf
    for other_cell in other_cells:
        subset["{}_min_zscore_others".format(cell)] = np.minimum(subset["{}_min_zscore_others".format(cell)], subset["{}_avg_pred_zscore".format(other_cell)])
    
    subset["{}_diff_zscore".format(cell)] = subset["{}_avg_pred_zscore".format(cell)] - subset["{}_min_zscore_others".format(cell)]

    # sort motifs by the difference in the z-score of the motif in the cell type of interest vs the minimum z-score of the motif in the other cell types
    subset = subset.sort_values(by="{}_diff_zscore".format(cell), ascending=True).reset_index(drop=True)

    # print the top motifs
    print("Top downregulated motifs in {}".format(cell))
    print(subset.head(args.top_n_motifs_to_print)[["motif", "{}_avg_pred".format(cell), "{}_avg_pred_zscore".format(cell), "{}_min_zscore_others".format(cell), "{}_diff_zscore".format(cell)]])

    # plot them
    num = args.top_n_motifs_to_print
    fig, axes = plt.subplots(num, 1, figsize=(10, 2*num))
    if num == 1:
        axes = [axes]
    for i, motif in enumerate(subset.head(num)["motif"]):
        plotting.seqlogo(motif_dict[motif].pwm, ax=axes[i])
        axes[i].axis('off')
        motif = motif.split("[")[0]
        axes[i].set_title(motif, fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(motif_insertion_analysis_dir, "{}_{}_downreg_motifs.png".format(name_format, cell)), dpi=300)