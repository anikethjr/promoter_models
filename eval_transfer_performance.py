import numpy as np
import pandas as pd
import os
import pdb
import argparse
import wandb
import h5py
import json
from tqdm import tqdm
import scipy.stats as stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from promoter_modelling.dataloaders import FluorescenceData, LL100, CCLE, Roadmap, Sharpr_MPRA, SuRE, ENCODETFChIPSeq, lentiMPRA, STARRSeq
from promoter_modelling import backbone_modules
from promoter_modelling import MTL_modules

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision('medium')

def get_predictions(args, config, finetune=False):
    # define directories
    root_dir = config["root_dir"]

    # should already exist
    root_data_dir = config["root_data_dir"]
    common_cache_dir = os.path.join(root_data_dir, "common")
    model_save_dir = os.path.join(root_dir, "saved_models")
    
    # create data loaders
    if args.modelling_strategy == "joint":
        assert args.joint_tasks is not None, "Must specify tasks to jointly train on"
        tasks = args.joint_tasks.split(",")
    elif args.modelling_strategy == "pretrain":
        assert args.pretrain_tasks is not None, "Must specify tasks to pretrain on"
        tasks = args.pretrain_tasks.split(",")
    elif args.modelling_strategy == "pretrain+finetune" or args.modelling_strategy == "pretrain+linear_probing":
        assert args.pretrain_tasks is not None, "Must specify tasks to pretrain on"
        assert args.finetune_tasks is not None, "Must specify tasks to finetune or perform linear probing on"
        pretrain_tasks = args.pretrain_tasks.split(",")
        finetune_tasks = args.finetune_tasks.split(",")

        if finetune:
            tasks = finetune_tasks
        else:
            tasks = pretrain_tasks
    elif args.modelling_strategy == "single_task":
        assert args.single_task is not None, "Must specify task to train on"
        tasks = [args.single_task]
    else:
        raise ValueError("Invalid modelling strategy")

    dataloaders = {}
    print("Instantiating dataloaders...")
    for task in tasks:
        if task == "all_tasks" or task == "RNASeq": # special task names
            dataloaders[task] = []
            tasks_set = None
            if "pretrain" in args.modelling_strategy:
                if task == "all_tasks":
                    tasks_set = ["LL100", "CCLE", "Roadmap", "SuRE_classification", "Sharpr_MPRA", "ENCODETFChIPSeq"]
                elif task == "RNASeq":
                    tasks_set = ["LL100", "CCLE", "Roadmap"]
            elif args.modelling_strategy == "joint":
                if task == "all_tasks":
                    tasks_set = ["LL100", "CCLE", "Roadmap", "SuRE_classification", "Sharpr_MPRA", "ENCODETFChIPSeq", "FluorescenceData"]
                elif task == "RNASeq":
                    tasks_set = ["LL100", "CCLE", "Roadmap"]

            for t in tasks_set:
                if t == "LL100":
                    dataloaders[task].append(LL100.LL100DataLoader(batch_size=args.batch_size, \
                                                                    cache_dir=os.path.join(root_data_dir, "LL-100"), \
                                                                    common_cache_dir=common_cache_dir, \
                                                                    promoter_windows_relative_to_TSS=[[0, 250]]))
                elif t == "CCLE":
                    dataloaders[task].append(CCLE.CCLEDataLoader(batch_size=args.batch_size, \
                                                                    cache_dir=os.path.join(root_data_dir, "CCLE"), \
                                                                    common_cache_dir=common_cache_dir, \
                                                                    promoter_windows_relative_to_TSS=[[0, 250]]))
                elif t == "Roadmap":
                    dataloaders[task].append(Roadmap.RoadmapDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "Roadmap"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                    promoter_windows_relative_to_TSS=[[0, 250]]))
                elif t == "Sharpr_MPRA":
                    dataloaders[task].append(Sharpr_MPRA.SharprMPRADataLoader(batch_size=args.batch_size, \
                                                                                data_dir=os.path.join(root_data_dir, "Sharpr_MPRA")))
                elif t == "SuRE_classification":
                    for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                        dataloaders[task].append(SuRE.SuREDataLoader(batch_size=args.batch_size, \
                                                                        genome_id=genome_id, \
                                                                        cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                        datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                        task="classification", \
                                                                        shrink_test_set=args.shrink_test_set))
                elif t == "SuRE_regression":
                    for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                        dataloaders[task].append(SuRE.SuREDataLoader(batch_size=args.batch_size, \
                                                                        genome_id=genome_id, \
                                                                        cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                        datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                        task="regression", \
                                                                        shrink_test_set=args.shrink_test_set))
                elif t == "ENCODETFChIPSeq":
                    dataloaders[task].append(ENCODETFChIPSeq.ENCODETFChIPSeqDataLoader(batch_size=args.batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq"), \
                                                                                        common_cache_dir=common_cache_dir, \
                                                                                        datasets_save_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq_data"), \
                                                                                        shrink_test_set=args.shrink_test_set, \
                                                                                        fasta_shuffle_letters_path=args.fasta_shuffle_letters_path))
                elif t == "lentiMPRA":
                    dataloaders[task].append(lentiMPRA.lentiMPRADataLoader(batch_size=args.batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "lentiMPRA"), \
                                                                            common_cache_dir=common_cache_dir))
                elif t == "STARRSeq":
                    dataloaders[task].append(STARRSeq.STARRSeqDataLoader(batch_size=args.batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "STARRSeq"), \
                                                                            common_cache_dir=common_cache_dir))
                elif t == "FluorescenceData":
                    dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData")))
                elif t == "FluorescenceData_DE":
                    dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                        predict_DE=True))
        elif task == "LL100":
            dataloaders[task] = LL100.LL100DataLoader(batch_size=args.batch_size, \
                                                        cache_dir=os.path.join(root_data_dir, "LL-100"), \
                                                        common_cache_dir=common_cache_dir, \
                                                        promoter_windows_relative_to_TSS=[[0, 250]])
        elif task == "CCLE":
            dataloaders[task] = CCLE.CCLEDataLoader(batch_size=args.batch_size, \
                                                    cache_dir=os.path.join(root_data_dir, "CCLE"), \
                                                    common_cache_dir=common_cache_dir, \
                                                    promoter_windows_relative_to_TSS=[[0, 250]])
        elif task == "Roadmap":
            dataloaders[task] = Roadmap.RoadmapDataLoader(batch_size=args.batch_size, \
                                                            cache_dir=os.path.join(root_data_dir, "Roadmap"), \
                                                            common_cache_dir=common_cache_dir, \
                                                            promoter_windows_relative_to_TSS=[[0, 250]])
        elif task == "Sharpr_MPRA":
            dataloaders[task] = Sharpr_MPRA.SharprMPRADataLoader(batch_size=args.batch_size, \
                                                                    data_dir=os.path.join(root_data_dir, "Sharpr_MPRA"))
        elif task == "SuRE_classification":
            dataloaders[task] = []
            for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                dataloaders[task].append(SuRE.SuREDataLoader(batch_size=args.batch_size, \
                                                                genome_id=genome_id, \
                                                                cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                common_cache_dir=common_cache_dir, \
                                                                datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                task="classification", \
                                                                shrink_test_set=args.shrink_test_set))
        elif task == "SuRE_regression":
            dataloaders[task] = []
            for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                dataloaders[task].append(SuRE.SuREDataLoader(batch_size=args.batch_size, \
                                                                genome_id=genome_id, \
                                                                cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                common_cache_dir=common_cache_dir, \
                                                                datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                task="regression", \
                                                                shrink_test_set=args.shrink_test_set))
        elif task == "ENCODETFChIPSeq":
            dataloaders[task] = ENCODETFChIPSeq.ENCODETFChIPSeqDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                        datasets_save_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq_data"), \
                                                                        shrink_test_set=args.shrink_test_set, \
                                                                        fasta_shuffle_letters_path=args.fasta_shuffle_letters_path)
        elif task == "lentiMPRA":
            dataloaders[task] = lentiMPRA.lentiMPRADataLoader(batch_size=args.batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "lentiMPRA"), \
                                                                common_cache_dir=common_cache_dir)
        elif task == "STARRSeq":
            dataloaders[task] = STARRSeq.STARRSeqDataLoader(batch_size=args.batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "STARRSeq"), \
                                                                common_cache_dir=common_cache_dir)
        elif task == "FluorescenceData":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"))
        elif task == "FluorescenceData_DE":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                        predict_DE=True)
        elif task == "FluorescenceData_JURKAT":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=0)
        elif task == "FluorescenceData_K562":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=1)
        elif task == "FluorescenceData_THP1":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=2)
    
    all_dataloaders = []
    for task in tasks:
        if dataloaders[task].__class__ == list:
            all_dataloaders.extend(dataloaders[task])
        else:
            all_dataloaders.append(dataloaders[task])
    print("Total number of dataloaders = {}".format(len(all_dataloaders)))   

    # setup training parameters
    if "pretrain" in args.modelling_strategy and not finetune:
        metric_to_monitor = args.pretrain_metric_to_monitor
        metric_direction_which_is_optimal = args.pretrain_metric_direction_which_is_optimal
        batch_size = args.pretrain_batch_size
    else:
        metric_to_monitor = args.metric_to_monitor
        metric_direction_which_is_optimal = args.metric_direction_which_is_optimal
        batch_size = args.batch_size

    # multiple models are trained only for finetuning/joint training/single task training
    num_models_to_train = args.num_random_seeds
    if "pretrain" in args.modelling_strategy and not finetune:
        num_models_to_train = 1

    # model name format
    name_format = ""
    if "pretrain" in args.modelling_strategy and finetune:
        if "finetune" in args.modelling_strategy:
            name_format = "finetune_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
        if "linear_probing" in args.modelling_strategy:
            name_format = "linear_probing_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
    elif "pretrain" in args.modelling_strategy and not finetune:
        name_format = "pretrain_on_{}".format("+".join(tasks))
    elif "joint" in args.modelling_strategy:
        name_format = "joint_train_on_{}".format("+".join(tasks))
    elif "single" in args.modelling_strategy:
        name_format = "individual_training_on_{}".format("+".join(tasks))

    # map to model classes
    model_class = backbone_modules.get_backbone_class(args.model_name)
    if args.model_name != "MTLucifer":
        name_format = f"{args.model_name}_" + name_format

    # get best trained model path
    best_model_path = ""
    minimize_metric = metric_direction_which_is_optimal == "min"
    if minimize_metric:
        best_metric = np.inf
    else:
        best_metric = -np.inf

    # find best model across seeds
    for seed in range(num_models_to_train):
        if num_models_to_train > 1:
            print("Random seed = {}".format(seed))
            # set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)

            name = name_format + "_seed_{}".format(seed)
        else:
            name = name_format

        cur_models_save_dir = os.path.join(model_save_dir, name, "default", "checkpoints")
        
        # find path to best existing model
        all_saved_models = os.listdir(cur_models_save_dir)
        for path in all_saved_models:
            val_metric = path.split("=")[-1][:-len(".ckpt")]
            if "-v" in val_metric:
                val_metric = float(val_metric[:-len("-v1")])
            else:
                val_metric = float(val_metric)

            if minimize_metric:
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_model_path = os.path.join(cur_models_save_dir, path)
            else:
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_model_path = os.path.join(cur_models_save_dir, path)

    print("Best existing model across seeds is: {}".format(best_model_path))

    # load it
    checkpoint = torch.load(best_model_path, map_location=device)

    new_state_dict = {}
    for key in checkpoint["state_dict"]:
        if key.startswith("model."):
            new_state_dict[key[len("model."):]] = checkpoint["state_dict"][key]

    # instantiate model
    mtlpredictor = MTL_modules.MTLPredictor(model_class=model_class, \
                                            all_dataloader_modules=all_dataloaders, \
                                            batch_size=batch_size, \
                                            use_preconstructed_dataloaders=True).to(device)

    mtlpredictor.model.load_state_dict(new_state_dict, strict=False)        
    print("Loaded best trained model")

    # create dataloader of the task to be evaluated
    task_to_evaluate = args.task_to_evaluate
    evaluation_dataloaders = []
    if task_to_evaluate == "FluorescenceData":
        for dummy in range(len(all_dataloaders)):
            evaluation_dataloaders.append(FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "FluorescenceData")))
    elif task_to_evaluate == "FluorescenceData_JURKAT":
        for dummy in range(len(all_dataloaders)):
            evaluation_dataloaders.append(FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                return_specified_cells=0))
    elif task_to_evaluate == "FluorescenceData_K562":
        for dummy in range(len(all_dataloaders)):
            evaluation_dataloaders.append(FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                return_specified_cells=1))
    elif task_to_evaluate == "FluorescenceData_THP1":
        for dummy in range(len(all_dataloaders)):
            evaluation_dataloaders.append(FluorescenceData.FluorescenceDataLoader(batch_size=args.batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                return_specified_cells=2))
    else:
        raise ValueError("Invalid task to evaluate")
    
    final_evaluation_dataloader = MTL_modules.MTLDataLoader(evaluation_dataloaders, return_full_dataset_for_predict=True)
        
    # get predictions
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    model_outputs = trainer.predict(mtlpredictor, final_evaluation_dataloader)
    
    dataloader_to_outputs = {}
    dataloader_to_y = {}
    dataloader_to_pred = {}
    dataloader_to_output_names = {}
    init_y = None
    for i, dl in enumerate(all_dataloaders):
        dataloader_to_output_names[dl.name] = dl.output_names
        dl = dl.name
        print("Evaluating predictions of head that was trained on {} and tested on {}".format(dl, task_to_evaluate))
        
        if len(all_dataloaders) == 1:
            dataloader_to_outputs[dl] = model_outputs
        else:
            dataloader_to_outputs[dl] = model_outputs[i]
                    
        dataloader_to_y[dl] = torch.vstack([dataloader_to_outputs[dl][j]["y"] for j in range(len(dataloader_to_outputs[dl]))]).detach().cpu()
        dataloader_to_pred[dl] = torch.vstack([dataloader_to_outputs[dl][j]["pred"] for j in range(len(dataloader_to_outputs[dl]))]).detach().cpu()

        if init_y is None:
            init_y = dataloader_to_y[dl].detach().cpu()
        assert torch.all(init_y == dataloader_to_y[dl]), "y values are not the same across heads"

        print("y shape = {}".format(dataloader_to_y[dl].shape))
        print("pred shape = {}".format(dataloader_to_pred[dl].shape))

    eval_task_output_names = evaluation_dataloaders[0].output_names

    return init_y, dataloader_to_pred, dataloader_to_output_names, eval_task_output_names, tasks

args = argparse.ArgumentParser()
args.add_argument("--config_path", type=str, default="./config.json", help="Path to config file")
args.add_argument("--model_name", type=str, default="MTLucifer", help="Name of model to use, must be one of {}".format(backbone_modules.get_all_backbone_names()))
args.add_argument("--modelling_strategy", type=str, required=True, help="Modelling strategy to use, either 'joint', 'pretrain+finetune', 'pretrain+linear_probing' or 'single_task'")

args.add_argument("--joint_tasks", type=str, default=None, help="Comma separated list of tasks to jointly train on")
args.add_argument("--pretrain_tasks", type=str, default=None, help="Comma separated list of tasks to pretrain on")
args.add_argument("--finetune_tasks", type=str, default=None, help="Comma separated list of tasks to finetune or perform linear probing on")
args.add_argument("--single_task", type=str, default=None, help="Task to train on")

args.add_argument("--shrink_test_set", action="store_true", help="Shrink large test sets (SuRE and ENCODETFChIPSeq) to 10 examples to make evaluation faster")

args.add_argument("--batch_size", type=int, default=96, help="Batch size")
args.add_argument("--pretrain_batch_size", type=int, default=96, help="Pretrain batch size")

args.add_argument("--num_random_seeds", type=int, default=1, help="Number of random seeds to train with")

args.add_argument("--metric_to_monitor", type=str, default="val_Fluorescence_mean_SpearmanR", help="Name of metric to monitor for early stopping")
args.add_argument("--metric_direction_which_is_optimal", type=str, default="max", help="Should metric be maximised (specify 'max') or minimised (specify 'min')?")
args.add_argument("--pretrain_metric_to_monitor", type=str, default="overall_val_loss", help="Name of pretrain metric to monitor for early stopping")
args.add_argument("--pretrain_metric_direction_which_is_optimal", type=str, default="min", help="Should pretrain metric be maximised (specify 'max') or minimised (specify 'min')?")

args.add_argument("--task_to_evaluate", type=str, default="FluorescenceData", help="Task to evaluate on")
args.add_argument("--output_inds_to_compare", type=str, required=True, help="List of comma separated indices of outputs to compare to actual values from the evaluation task - the length of the list must be the same as the number of dataloaders used to train the model and the number of values in each comma separated string must be the same as the number of values per example in the evaluation task, use -1 if that dimension shouldn't be compared (e.g. 33,-1,55 88,12,21)")

args = args.parse_args()

assert os.path.exists(args.config_path), "Config file does not exist"
# Load config file
with open(args.config_path, "r") as f:
    config = json.load(f)

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# get predictions
if "finetune" in args.modelling_strategy:
    y, all_heads_preds, output_names, eval_task_output_names, training_tasks = get_predictions(args, config, finetune=True)
else:
    y, all_heads_preds, output_names, eval_task_output_names, training_tasks = get_predictions(args, config, finetune=False)

# if SuRE is one of the tasks the model was trained on, we average the predictions across the 4 samples
# then, compute the regression output for each cell as the weighted sum of the class probabilities and bin numbers
check_SuRE = False
if "SuRE_classification" in training_tasks:
    check_SuRE = True
if check_SuRE:
    print("Averaging SuRE predictions across 4 samples after softmax")
    SuRE_preds = []
    keys_to_remove = []
    for dl in all_heads_preds.keys():
        if "SuRE" in dl:
            SuRE_preds.append(F.softmax(all_heads_preds[dl], dim=1))
            keys_to_remove.append(dl)

    # remove SuRE predictions from all_heads_preds
    for key in keys_to_remove:
        del all_heads_preds[key]
        del output_names[key]

    SuRE_preds = torch.mean(torch.stack(SuRE_preds), dim=0)

    # compute regression output
    print("Computing regression output from SuRE predictions")
    K562_SuRE_preds = torch.sum(SuRE_preds[:, :5] * torch.arange(0, 5), dim=1)
    HepG2_SuRE_preds = torch.sum(SuRE_preds[:, 5:] * torch.arange(0, 5), dim=1)
    final_SuRE_preds = torch.vstack([K562_SuRE_preds, HepG2_SuRE_preds]).T

    all_heads_preds["SuRE_classification"] = final_SuRE_preds
    output_names["SuRE_classification"] = ["K562_SuRE_classification", "HepG2_SuRE_classification"]

    print("Final SuRE preds shape = {}".format(final_SuRE_preds.shape))

# if RNASeq is one of the tasks the model was trained on, expand training_tasks to have all their names in order
if "RNASeq" in training_tasks:
    training_tasks = ["LL100", "CCLE", "Roadmap"]

# compare predictions to actual values
print("Comparing predictions to actual values")
assert (args.output_inds_to_compare is not None), "Must specify output inds to compare"
args.output_inds_to_compare = args.output_inds_to_compare.strip().split(" ")

assert (len(args.output_inds_to_compare) == len(all_heads_preds)), "Number of elements in output inds to compare list must be the same as the number of training tasks/heads"

for i, task in enumerate(training_tasks):
    print("Comparing predictions from head trained on {} to actual values from {}".format(task, args.task_to_evaluate))
    output_inds_for_dl = args.output_inds_to_compare[i].split(",")
    assert len(output_inds_for_dl) == y.shape[1], "Number of output inds to compare must be the same as the number of values per example in the evaluation task"

    # convert output inds to ints
    output_inds_for_dl = [int(x) for x in output_inds_for_dl]

    for target_ind, target_output_name in enumerate(eval_task_output_names):
        correspongding_output_ind = output_inds_for_dl[target_ind]
        if correspongding_output_ind == -1:
            print("Skipping comparison of {}".format(target_output_name))
            continue

        # get source output name
        source_output_name = output_names[task][correspongding_output_ind]

        # get predictions and actual values
        preds = all_heads_preds[task][:, correspongding_output_ind]
        actuals = y[:, target_ind]

        # compute correlation
        pearsonr = stats.pearsonr(preds, actuals)[0]
        spearmanr = stats.spearmanr(preds, actuals).correlation

        print("PearsonR between {} and {} = {} ≈ {}".format(source_output_name, target_output_name, pearsonr, round(pearsonr, 4)))
        print("SpearmanR between {} and {} = {} ≈ {}".format(source_output_name, target_output_name, spearmanr, round(spearmanr, 4)))       

    print()
