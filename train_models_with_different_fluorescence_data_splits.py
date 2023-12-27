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
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score

import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from promoter_modelling.dataloaders import FluorescenceData, FluorescenceData_classification, FluorescenceData_with_motifs, FluorescenceData_DNABERT, \
                                           LL100, CCLE, Roadmap, Sharpr_MPRA, SuRE, ENCODETFChIPSeq, STARRSeq, Malinois_MPRA, lentiMPRA
from promoter_modelling import backbone_modules
from promoter_modelling import MTL_modules

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision('medium')

def train_model(args, config, finetune=False):
    # create directories
    # for modelling
    root_dir = config["root_dir"]
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    model_save_dir = os.path.join(root_dir, "saved_models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)
    summaries_save_dir = os.path.join(root_dir, "summaries")
    if not os.path.exists(summaries_save_dir):
        os.makedirs(summaries_save_dir, exist_ok=True)

    # for data
    root_data_dir = config["root_data_dir"]
    if not os.path.exists(root_data_dir):
        os.makedirs(root_data_dir, exist_ok=True)
    common_cache_dir = os.path.join(root_data_dir, "common")
    if not os.path.exists(common_cache_dir):
        os.makedirs(common_cache_dir, exist_ok=True)

    # setup task(s)
    if args.modelling_strategy == "joint":
        assert args.joint_tasks is not None, "Must specify tasks to jointly train on"
        tasks = args.joint_tasks.split(",")
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

    if args.model_name.startswith("MotifBased"):
        assert len(tasks) == 1, "Motif-based models can only be trained on a single task"
        assert tasks[0] == "FluorescenceData" or tasks[0] == "FluorescenceData_DE", "Motif-based models can only be trained on FluorescenceData or FluorescenceData_DE"

    # load pretrained model state dict if necessary
    if "pretrain" in args.modelling_strategy and finetune:
        print("Loading pre-trained model state dict")

        pretrained_model_name = "pretrain_on_{}".format("+".join(pretrain_tasks))
        # map to model classes
        model_class = backbone_modules.get_backbone_class(args.model_name)
        if args.model_name != "MTLucifer":
            pretrained_model_name = f"{args.model_name}_" + pretrained_model_name

        pretrain_metric_direction_which_is_optimal = args.pretrain_metric_direction_which_is_optimal
        pretrained_model_save_dir = os.path.join(model_save_dir, pretrained_model_name, "default", "checkpoints")

        # find path to best existing model
        all_saved_models = os.listdir(pretrained_model_save_dir)
        best_model_path = ""
        minimize_metric = pretrain_metric_direction_which_is_optimal == "min"
        if minimize_metric:
            best_metric = np.inf
        else:
            best_metric = -np.inf
        for path in all_saved_models:
            val_metric = path.split("=")[-1][:-len(".ckpt")]
            if "-v" in val_metric:
                val_metric = float(val_metric[:-len("-v1")])
            else:
                val_metric = float(val_metric)
                
            if minimize_metric:
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_model_path = path
            else:
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_model_path = path
                    
        print("Best pre-trained model is: {}".format(os.path.join(pretrained_model_save_dir, best_model_path)))

        # load it
        pretrained_checkpoint = torch.load(os.path.join(pretrained_model_save_dir, best_model_path), map_location=device)

    # setup training parameters
    if "pretrain" in args.modelling_strategy and not finetune:
        print("Pre-training model")
        metric_to_monitor = args.pretrain_metric_to_monitor
        metric_direction_which_is_optimal = args.pretrain_metric_direction_which_is_optimal
        lr = args.pretrain_lr
        weight_decay = args.pretrain_weight_decay
        batch_size = args.pretrain_batch_size
        max_epochs = args.pretrain_max_epochs
        train_mode = args.pretrain_train_mode
    else:
        print("Training model from scratch")
        metric_to_monitor = args.metric_to_monitor
        metric_direction_which_is_optimal = args.metric_direction_which_is_optimal
        lr = args.lr
        weight_decay = args.weight_decay
        batch_size = args.batch_size
        max_epochs = args.max_epochs
        train_mode = args.train_mode

    print("Learning rate = {}, weight decay = {}, batch size = {}, max epochs = {}, train mode = {}".format(lr, weight_decay, batch_size, max_epochs, train_mode))

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
    
    # instantiate dataloaders
    dataloaders = {}
    print("Instantiating dataloaders...")
    for task in tasks:
        if task == "all_tasks" or task == "RNASeq": # special task names
            dataloaders[task] = []
            tasks_set = None
            if args.modelling_strategy == "pretrain+finetune" or args.modelling_strategy == "pretrain+linear_probing":
                if task == "all_tasks":
                    tasks_set = ["LL100", "CCLE", "Roadmap", "SuRE_classification", "Sharpr_MPRA", "lentiMPRA", "ENCODETFChIPSeq"]
                elif task == "RNASeq":
                    tasks_set = ["LL100", "CCLE", "Roadmap"]
            elif args.modelling_strategy == "joint":
                if task == "all_tasks":
                    tasks_set = ["LL100", "CCLE", "Roadmap", "SuRE_classification", "Sharpr_MPRA", "lentiMPRA", "ENCODETFChIPSeq", "FluorescenceData"]
                elif task == "RNASeq":
                    tasks_set = ["LL100", "CCLE", "Roadmap"]

            for t in tasks_set:
                if t == "LL100":
                    dataloaders[task].append(LL100.LL100DataLoader(batch_size=batch_size, \
                                                                    cache_dir=os.path.join(root_data_dir, "LL-100"), \
                                                                    common_cache_dir=common_cache_dir))
                elif t == "CCLE":
                    dataloaders[task].append(CCLE.CCLEDataLoader(batch_size=batch_size, \
                                                                    cache_dir=os.path.join(root_data_dir, "CCLE"), \
                                                                    common_cache_dir=common_cache_dir))
                elif t == "Roadmap":
                    dataloaders[task].append(Roadmap.RoadmapDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "Roadmap"), \
                                                                        common_cache_dir=common_cache_dir))
                elif t == "Sharpr_MPRA":
                    dataloaders[task].append(Sharpr_MPRA.SharprMPRADataLoader(batch_size=batch_size, \
                                                                                data_dir=os.path.join(root_data_dir, "Sharpr_MPRA")))
                elif t == "lentiMPRA":
                    dataloaders[task].append(lentiMPRA.lentiMPRADataLoader(batch_size=batch_size, \
                                                                            data_dir=os.path.join(root_data_dir, "lentiMPRA", \
                                                                            common_cache_dir=common_cache_dir)))
                elif t == "STARRSeq":
                    dataloaders[task].append(STARRSeq.STARRSeqDataLoader(batch_size=batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "STARRSeq"), \
                                                                            common_cache_dir=common_cache_dir))
                elif t == "SuRE_classification":
                    for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                        dataloaders[task].append(SuRE.SuREDataLoader(batch_size=batch_size, \
                                                                        genome_id=genome_id, \
                                                                        cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                        datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                        task="classification", \
                                                                        shrink_test_set=args.shrink_test_set))
                elif t == "SuRE_regression":
                    for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                        dataloaders[task].append(SuRE.SuREDataLoader(batch_size=batch_size, \
                                                                        genome_id=genome_id, \
                                                                        cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                        datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                        task="regression", \
                                                                        shrink_test_set=args.shrink_test_set))
                elif t == "ENCODETFChIPSeq":
                    dataloaders[task].append(ENCODETFChIPSeq.ENCODETFChIPSeqDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq"), \
                                                                                        common_cache_dir=common_cache_dir, \
                                                                                        datasets_save_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq_data"), \
                                                                                        shrink_test_set=args.shrink_test_set, \
                                                                                        fasta_shuffle_letters_path=args.fasta_shuffle_letters_path))
                elif t == "FluorescenceData":
                    if args.model_name.startswith("MotifBased"):
                        dataloaders[task].append(FluorescenceData_with_motifs.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                                     cache_dir=os.path.join(root_data_dir, "FluorescenceData_with_motifs")))
                    elif "DNABERT" in args.model_name:
                        dataloaders[task].append(FluorescenceData_DNABERT.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                                     cache_dir=os.path.join(root_data_dir, "FluorescenceData_DNABERT")))
                    else:
                        dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData")))
                elif t == "FluorescenceData_DE":
                    if args.model_name.startswith("MotifBased"):
                        dataloaders[task].append(FluorescenceData_with_motifs.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                                     cache_dir=os.path.join(root_data_dir, "FluorescenceData_with_motifs_DE"), \
                                                                                                     predict_DE=True))
                    elif "DNABERT" in args.model_name:
                        dataloaders[task].append(FluorescenceData_DNABERT.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                                     cache_dir=os.path.join(root_data_dir, "FluorescenceData_DNABERT_DE"), \
                                                                                                     predict_DE=True))
                    else:
                        dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                                        predict_DE=True))
                elif t == "FluorescenceData_classification":
                    dataloaders[task].append(FluorescenceData_classification.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_classification")))
        elif task == "LL100":
            dataloaders[task] = LL100.LL100DataLoader(batch_size=batch_size, \
                                                        cache_dir=os.path.join(root_data_dir, "LL-100"), \
                                                        common_cache_dir=common_cache_dir)
        elif task == "CCLE":
            dataloaders[task] = CCLE.CCLEDataLoader(batch_size=batch_size, \
                                                    cache_dir=os.path.join(root_data_dir, "CCLE"), \
                                                    common_cache_dir=common_cache_dir)
        elif task == "Roadmap":
            dataloaders[task] = Roadmap.RoadmapDataLoader(batch_size=batch_size, \
                                                            cache_dir=os.path.join(root_data_dir, "Roadmap"), \
                                                            common_cache_dir=common_cache_dir)
        elif task == "STARRSeq":
            dataloaders[task] = STARRSeq.STARRSeqDataLoader(batch_size=batch_size, \
                                                                cache_dir=os.path.join(root_data_dir, "STARRSeq"), \
                                                                common_cache_dir=common_cache_dir)
        elif task == "Sharpr_MPRA":
            dataloaders[task] = Sharpr_MPRA.SharprMPRADataLoader(batch_size=batch_size, \
                                                                    data_dir=os.path.join(root_data_dir, "Sharpr_MPRA"))
        elif task == "lentiMPRA":
            dataloaders[task] = lentiMPRA.lentiMPRADataLoader(batch_size=batch_size, \
                                                                data_dir=os.path.join(root_data_dir, "lentiMPRA"), \
                                                                common_cache_dir=common_cache_dir)
        elif task == "SuRE_classification":
            dataloaders[task] = []
            for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                dataloaders[task].append(SuRE.SuREDataLoader(batch_size=batch_size, \
                                                                genome_id=genome_id, \
                                                                cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                common_cache_dir=common_cache_dir, \
                                                                datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                task="classification", \
                                                                shrink_test_set=args.shrink_test_set))
        elif task == "SuRE_regression":
            dataloaders[task] = []
            for genome_id in ["SuRE42_HG02601", "SuRE43_GM18983", "SuRE44_HG01241", "SuRE45_HG03464"]:
                dataloaders[task].append(SuRE.SuREDataLoader(batch_size=batch_size, \
                                                                genome_id=genome_id, \
                                                                cache_dir=os.path.join(root_data_dir, "SuRE"), \
                                                                common_cache_dir=common_cache_dir, \
                                                                datasets_save_dir=os.path.join(root_data_dir, "SuRE_data"), \
                                                                task="regression", \
                                                                shrink_test_set=args.shrink_test_set))
        elif task == "ENCODETFChIPSeq":
            dataloaders[task] = ENCODETFChIPSeq.ENCODETFChIPSeqDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq"), \
                                                                        common_cache_dir=common_cache_dir, \
                                                                        datasets_save_dir=os.path.join(root_data_dir, "ENCODETFChIPSeq_data"), \
                                                                        shrink_test_set=args.shrink_test_set, \
                                                                        fasta_shuffle_letters_path=args.fasta_shuffle_letters_path)
        elif task == "FluorescenceData":
            if args.model_name.startswith("MotifBased"):
                dataloaders[task] = FluorescenceData_with_motifs.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_with_motifs"))
            elif "DNABERT" in args.model_name:
                dataloaders[task] = FluorescenceData_DNABERT.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DNABERT"))
            else:
                dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "FluorescenceData"))
        elif task == "FluorescenceData_DE":
            if args.model_name.startswith("MotifBased"):
                dataloaders[task] = FluorescenceData_with_motifs.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_with_motifs_DE"), \
                                                                                        predict_DE=True)
            elif "DNABERT" in args.model_name:
                dataloaders[task] = FluorescenceData_DNABERT.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DNABERT_DE"), \
                                                                                        predict_DE=True)
            else:
                dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                            predict_DE=True)
        elif task == "FluorescenceData_classification":
            dataloaders[task] = FluorescenceData_classification.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_classification"))
        elif task == "FluorescenceData_JURKAT":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=0)
        elif task == "FluorescenceData_K562":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=1)
        elif task == "FluorescenceData_THP1":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=2)
    
    all_dataloaders = []
    for task in tasks:
        if dataloaders[task].__class__ == list:
            all_dataloaders.extend(dataloaders[task])
        else:
            all_dataloaders.append(dataloaders[task])
    print("Total number of dataloaders = {}".format(len(all_dataloaders)))   

    # train models
    all_seeds_r2 = {}
    all_seeds_pearsonr = {}
    all_seeds_srho = {}

    all_seeds_accuracy = {}
    all_seeds_precision = {}
    all_seeds_recall = {}
    all_seeds_f1 = {}

    percentile_thres = 90
    all_seeds_highly_expressed_accuracy = {}
    all_seeds_highly_expressed_precision = {}
    all_seeds_highly_expressed_recall = {}
    all_seeds_highly_expressed_f1 = {}
    all_seeds_lowly_expressed_accuracy = {}

    for seed in range(num_models_to_train):
        if num_models_to_train > 1:
            print("Random seed = {}".format(seed))
            # set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)

            for i in range(len(all_dataloaders)):
                if all_dataloaders[i].name.startswith("Fluorescence"):
                    if args.model_name.startswith("MotifBased"):
                        if all_dataloaders[i].predict_DE:
                            all_dataloaders[i] = FluorescenceData_with_motifs.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                 cache_dir=os.path.join(root_data_dir, "FluorescenceData_with_motifs_DE"), \
                                                                                 seed=seed, \
                                                                                 return_specified_cells=all_dataloaders[i].return_specified_cells, \
                                                                                 predict_DE=True)
                        else:
                            all_dataloaders[i] = FluorescenceData_with_motifs.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_with_motifs"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells)
                    elif "DNABERT" in args.model_name:
                        if all_dataloaders[i].predict_DE:
                            all_dataloaders[i] = FluorescenceData_DNABERT.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_DNABERT_DE"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells, \
                                                                                    predict_DE=True)
                        else:
                            all_dataloaders[i] = FluorescenceData_DNABERT.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_DNABERT"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells)
                    elif "classification" in all_dataloaders[i].name:
                        if all_dataloaders[i].predict_DE:
                            all_dataloaders[i] = FluorescenceData_classification.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_classification"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells, \
                                                                                    predict_DE=True)
                        else:
                            all_dataloaders[i] = FluorescenceData_classification.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_classification"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells)
                    else:
                        if all_dataloaders[i].predict_DE:
                            all_dataloaders[i] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells, \
                                                                                    predict_DE=True)
                        else:
                            all_dataloaders[i] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                                    seed=seed, \
                                                                                    return_specified_cells=all_dataloaders[i].return_specified_cells)
            
            name = name_format + "_dl_seed_{}".format(seed)
        else:
            name = name_format

        if args.model_name.startswith("MotifBased"):
            mtlpredictor = MTL_modules.MTLPredictor(model_class=model_class,\
                                                all_dataloader_modules=all_dataloaders, \
                                                batch_size=batch_size, \
                                                max_epochs=args.max_epochs, \
                                                lr=lr, \
                                                weight_decay=weight_decay, \
                                                with_motifs=True, \
                                                use_preconstructed_dataloaders=True, \
                                                train_mode=train_mode)
        else:                                                
            mtlpredictor = MTL_modules.MTLPredictor(model_class=model_class,\
                                                all_dataloader_modules=all_dataloaders, \
                                                batch_size=batch_size, \
                                                max_epochs=args.max_epochs, \
                                                lr=lr, \
                                                weight_decay=weight_decay, \
                                                use_preconstructed_dataloaders=True, \
                                                train_mode=train_mode)
        
        cur_models_save_dir = os.path.join(model_save_dir, name, "default", "checkpoints")

        # first check if there's an existing joint model
        check = False
        if args.use_existing_models:
            if os.path.exists(cur_models_save_dir):
                done_file = os.path.join(model_save_dir, name, "default", "done.txt")
                if os.path.exists(done_file):
                    check = True
        if check: # found existing model and using it
            print("Using existing models and evaluating them")
            
            # find path to best existing model
            all_saved_models = os.listdir(cur_models_save_dir)
            best_model_path = "" 
            minimize_metric = metric_direction_which_is_optimal == "min"
            if minimize_metric:
                best_metric = np.inf
            else:
                best_metric = -np.inf
            for path in all_saved_models:
                val_metric = path.split("=")[-1][:-len(".ckpt")]
                if "-v" in val_metric:
                    val_metric = float(val_metric[:-len("-v1")])
                else:
                    val_metric = float(val_metric)
                    
                if minimize_metric:
                    if val_metric < best_metric:
                        best_metric = val_metric
                        best_model_path = path
                else:
                    if val_metric > best_metric:
                        best_metric = val_metric
                        best_model_path = path
                        
            print("Best existing model is: {}".format(os.path.join(cur_models_save_dir, best_model_path)))

            # load it
            checkpoint = torch.load(os.path.join(cur_models_save_dir, best_model_path), map_location=device)

            new_state_dict = {}
            for key in checkpoint["state_dict"]:
                if key.startswith("model."):
                    new_state_dict[key[len("model."):]] = checkpoint["state_dict"][key]

            mtlpredictor.model.load_state_dict(new_state_dict, strict=False)        
            print("Loaded existing model")
            
            # get test set predictions
            trainer = L.Trainer(accelerator="gpu", devices=1)
            best_model_test_outputs = trainer.predict(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader())

        else:
            print("Training model")

            if "pretrain" in args.modelling_strategy and finetune:
                new_state_dict = {}
                for key in pretrained_checkpoint["state_dict"]:
                    if key.startswith("model."):
                        new_state_dict[key[len("model."):]] = pretrained_checkpoint["state_dict"][key]

                mtlpredictor.model.load_state_dict(new_state_dict, strict=False)        
                print("Loaded pretrained model")
            
            # freeze backbone for linear probing
            if "linear_probing" in args.modelling_strategy and finetune:
                print("Freezing backbone for linear probing")
                # freeze backbone
                for param_name, param in mtlpredictor.model.named_parameters():
                    if param_name.startswith("Backbone.promoter_"):
                        param.requires_grad = False

                for param_name, param in mtlpredictor.model.named_parameters():
                    if param_name.startswith("Backbone.promoter_"):
                        assert param.requires_grad == False

            wandb_logger = WandbLogger(name=name, \
                                    project='promoter_modelling_pytorch', log_model=False)

            checkpoint_filename = "best-{epoch:02d}-{" + "{}".format(metric_to_monitor) + ":.5f}"
            checkpoint_callback = ModelCheckpoint(monitor=metric_to_monitor, \
                                                dirpath=os.path.join(model_save_dir, name, "default", "checkpoints"), \
                                                filename=checkpoint_filename, \
                                                save_top_k=args.save_top_k, mode=metric_direction_which_is_optimal)

            patience = args.patience
            early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, \
                                                patience=patience, verbose=True, mode=metric_direction_which_is_optimal)

            trainer = L.Trainer(logger=wandb_logger, \
                                callbacks=[early_stop_callback, checkpoint_callback], \
                                deterministic=True, accelerator="gpu", devices=1, \
                                log_every_n_steps=10, default_root_dir=model_save_dir, \
                                max_epochs=max_epochs, \
                                limit_test_batches=0, reload_dataloaders_every_n_epochs=2, enable_progress_bar = True, \
                                gradient_clip_val=1.0, num_sanity_val_steps=32, precision="16-mixed")

            trainer.fit(mtlpredictor, mtlpredictor.get_mtldataloader())

            # create done file
            done_file = os.path.join(model_save_dir, name, "default", "done.txt")
            with open(done_file, "w+") as f:
                f.write("done")

            wandb.finish()

            # get test set predictions
            best_model_test_outputs = trainer.predict(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader(), ckpt_path="best")

        # get metrics
        dataloader_to_outputs = {}
        dataloader_to_y = {}
        dataloader_to_pred = {}

        for i, dl in enumerate(all_dataloaders):
            dl = dl.name
            print(dl)
            
            if len(all_dataloaders) == 1:
                dataloader_to_outputs[dl] = best_model_test_outputs
            else:
                dataloader_to_outputs[dl] = best_model_test_outputs[i]
                        
            dataloader_to_y[dl] = torch.vstack([dataloader_to_outputs[dl][j]["y"] for j in range(len(dataloader_to_outputs[dl]))])
            dataloader_to_pred[dl] = torch.vstack([dataloader_to_outputs[dl][j]["pred"] for j in range(len(dataloader_to_outputs[dl]))])

            print("y shape = {}".format(dataloader_to_y[dl].shape))
            print("pred shape = {}".format(dataloader_to_pred[dl].shape))

            if "Fluorescence" in dl and "classification" in dl:
                print()
                for j, output in enumerate(all_dataloaders[i].output_names):
                    cur_y = dataloader_to_y[dl][:, j]
                    cur_pred = dataloader_to_pred[dl][:, j]

                    # apply sigmoid and round
                    cur_pred = torch.sigmoid(cur_pred)
                    cur_pred = torch.round(cur_pred)

                    # get overall metrics
                    acc = accuracy_score(cur_y, cur_pred)
                    f1 = f1_score(cur_y, cur_pred)
                    precision = precision_score(cur_y, cur_pred)
                    recall = recall_score(cur_y, cur_pred)

                    # get metrics for highly expressed promoters
                    test_set = all_dataloaders[i].test_set
                    highly_expressed_promoters = test_set[test_set["numerical_{}".format(output)] >= np.percentile(test_set["numerical_{}".format(output)], percentile_thres)]
                    highly_expressed_promoters_indices = highly_expressed_promoters.index.values
                    print("Number of highly expressed promoters = {}".format(len(highly_expressed_promoters_indices)))
                    highly_expressed_promoters_y = cur_y[highly_expressed_promoters_indices]
                    highly_expressed_promoters_pred = cur_pred[highly_expressed_promoters_indices]

                    acc_highly_expressed = accuracy_score(highly_expressed_promoters_y, highly_expressed_promoters_pred)
                    f1_highly_expressed = f1_score(highly_expressed_promoters_y, highly_expressed_promoters_pred)
                    precision_highly_expressed = precision_score(highly_expressed_promoters_y, highly_expressed_promoters_pred)
                    recall_highly_expressed = recall_score(highly_expressed_promoters_y, highly_expressed_promoters_pred)

                    # get metrics for lowly expressed promoters
                    lowly_expressed_promoters = test_set[test_set["numerical_{}".format(output)] <= np.percentile(test_set["numerical_{}".format(output)], 100 - percentile_thres)]
                    lowly_expressed_promoters_indices = lowly_expressed_promoters.index.values
                    print("Number of lowly expressed promoters = {}".format(len(lowly_expressed_promoters_indices)))
                    lowly_expressed_promoters_y = cur_y[lowly_expressed_promoters_indices]
                    lowly_expressed_promoters_pred = cur_pred[lowly_expressed_promoters_indices]

                    acc_lowly_expressed = accuracy_score(lowly_expressed_promoters_y, lowly_expressed_promoters_pred)

                    print()
                    print("{} Accuracy = {} ≈ {}".format(output, acc, np.around(acc, 4)))
                    print("{} F1 = {} ≈ {}".format(output, f1, np.around(f1, 4)))
                    print("{} Precision = {} ≈ {}".format(output, precision, np.around(precision, 4)))
                    print("{} Recall = {} ≈ {}".format(output, recall, np.around(recall, 4)))
                    print()

                    print("{} Accuracy (highly expressed) = {} ≈ {}".format(output, acc_highly_expressed, np.around(acc_highly_expressed, 4)))
                    print("{} F1 (highly expressed) = {} ≈ {}".format(output, f1_highly_expressed, np.around(f1_highly_expressed, 4)))
                    print("{} Precision (highly expressed) = {} ≈ {}".format(output, precision_highly_expressed, np.around(precision_highly_expressed, 4)))
                    print("{} Recall (highly expressed) = {} ≈ {}".format(output, recall_highly_expressed, np.around(recall_highly_expressed, 4)))
                    print()

                    print("{} Accuracy (lowly expressed) = {} ≈ {}".format(output, acc_lowly_expressed, np.around(acc_lowly_expressed, 4)))
                    print()

                    if output not in all_seeds_accuracy:
                        all_seeds_accuracy[output] = []
                        all_seeds_precision[output] = []
                        all_seeds_recall[output] = []
                        all_seeds_f1[output] = []

                        all_seeds_highly_expressed_accuracy[output] = []
                        all_seeds_highly_expressed_precision[output] = []
                        all_seeds_highly_expressed_recall[output] = []
                        all_seeds_highly_expressed_f1[output] = []

                        all_seeds_lowly_expressed_accuracy[output] = []

                    all_seeds_accuracy[output].append(acc)
                    all_seeds_precision[output].append(precision)
                    all_seeds_recall[output].append(recall)
                    all_seeds_f1[output].append(f1)

                    all_seeds_highly_expressed_accuracy[output].append(acc_highly_expressed)
                    all_seeds_highly_expressed_precision[output].append(precision_highly_expressed)
                    all_seeds_highly_expressed_recall[output].append(recall_highly_expressed)
                    all_seeds_highly_expressed_f1[output].append(f1_highly_expressed)

                    all_seeds_lowly_expressed_accuracy[output].append(acc_lowly_expressed)
            elif "Fluorescence" in dl:
                print()
                for j, output in enumerate(all_dataloaders[i].output_names):
                    cur_y = dataloader_to_y[dl][:, j]
                    cur_pred = dataloader_to_pred[dl][:, j]

                    r2 = r2_score(cur_y, cur_pred)
                    pearsonr = stats.pearsonr(cur_y, cur_pred)[0]
                    srho = stats.spearmanr(cur_y, cur_pred).correlation

                    print("{} R2 = {} ≈ {}".format(output, r2, np.around(r2, 4)))
                    print("{} PearsonR = {} ≈ {}".format(output, pearsonr, np.around(pearsonr, 4)))
                    print("{} Spearman rho = {} ≈ {}".format(output, srho, np.around(srho, 4)))
                    print()
                    
                    if output not in all_seeds_r2:
                        all_seeds_r2[output] = []
                        all_seeds_pearsonr[output] = []
                        all_seeds_srho[output] = []
                        
                    all_seeds_r2[output].append(r2)
                    all_seeds_pearsonr[output].append(pearsonr)
                    all_seeds_srho[output].append(srho)
    
    print()
    if len(all_seeds_r2) > 0 or len(all_seeds_accuracy) > 0:
        print("FINAL RESULTS ON FLUORESCENCE DATA")
        summary = vars(args)
        for output in all_seeds_accuracy:
            acc = np.average(all_seeds_accuracy[output])
            precision = np.average(all_seeds_precision[output])
            recall = np.average(all_seeds_recall[output])
            f1 = np.average(all_seeds_f1[output])

            acc_std = np.std(all_seeds_accuracy[output])
            precision_std = np.std(all_seeds_precision[output])
            recall_std = np.std(all_seeds_recall[output])
            f1_std = np.std(all_seeds_f1[output])

            print("{} avg Accuracy = {} +- {} ≈ {} +- {}".format(output, acc, acc_std, np.around(acc, 4), np.around(acc_std, 4)))
            print("{} avg Precision = {} +- {} ≈ {} +- {}".format(output, precision, precision_std, np.around(precision, 4), np.around(precision_std, 4)))
            print("{} avg Recall = {} +- {} ≈ {} +- {}".format(output, recall, recall_std, np.around(recall, 4), np.around(recall_std, 4)))
            print("{} avg F1 = {} +- {} ≈ {} +- {}".format(output, f1, f1_std, np.around(f1, 4), np.around(f1_std, 4)))
            print()

            summary[output + "_all_Accuracy"] = all_seeds_accuracy[output]
            summary[output + "_all_Precision"] = all_seeds_precision[output]
            summary[output + "_all_Recall"] = all_seeds_recall[output]
            summary[output + "_all_F1"] = all_seeds_f1[output]

            summary[output + "_avg_Accuracy"] = acc
            summary[output + "_avg_Precision"] = precision
            summary[output + "_avg_Recall"] = recall
            summary[output + "_avg_F1"] = f1

            summary[output + "_std_Accuracy"] = acc_std
            summary[output + "_std_Precision"] = precision_std
            summary[output + "_std_Recall"] = recall_std
            summary[output + "_std_F1"] = f1_std

            summary[output + "_avg_Accuracy_disp"] = "{} +- {}".format(np.around(acc, 4), np.around(acc_std, 4))
            summary[output + "_avg_Precision_disp"] = "{} +- {}".format(np.around(precision, 4), np.around(precision_std, 4))
            summary[output + "_avg_Recall_disp"] = "{} +- {}".format(np.around(recall, 4), np.around(recall_std, 4))
            summary[output + "_avg_F1_disp"] = "{} +- {}".format(np.around(f1, 4), np.around(f1_std, 4))


        for output in all_seeds_highly_expressed_accuracy:
            acc = np.average(all_seeds_highly_expressed_accuracy[output])
            precision = np.average(all_seeds_highly_expressed_precision[output])
            recall = np.average(all_seeds_highly_expressed_recall[output])
            f1 = np.average(all_seeds_highly_expressed_f1[output])

            acc_std = np.std(all_seeds_highly_expressed_accuracy[output])
            precision_std = np.std(all_seeds_highly_expressed_precision[output])
            recall_std = np.std(all_seeds_highly_expressed_recall[output])
            f1_std = np.std(all_seeds_highly_expressed_f1[output])

            print("{} avg Accuracy (highly expressed) = {} +- {} ≈ {} +- {}".format(output, acc, acc_std, np.around(acc, 4), np.around(acc_std, 4)))
            print("{} avg Precision (highly expressed) = {} +- {} ≈ {} +- {}".format(output, precision, precision_std, np.around(precision, 4), np.around(precision_std, 4)))
            print("{} avg Recall (highly expressed) = {} +- {} ≈ {} +- {}".format(output, recall, recall_std, np.around(recall, 4), np.around(recall_std, 4)))
            print("{} avg F1 (highly expressed) = {} +- {} ≈ {} +- {}".format(output, f1, f1_std, np.around(f1, 4), np.around(f1_std, 4)))
            print()

            summary[output + "_all_Accuracy_highly_expressed"] = all_seeds_highly_expressed_accuracy[output]
            summary[output + "_all_Precision_highly_expressed"] = all_seeds_highly_expressed_precision[output]
            summary[output + "_all_Recall_highly_expressed"] = all_seeds_highly_expressed_recall[output]
            summary[output + "_all_F1_highly_expressed"] = all_seeds_highly_expressed_f1[output]

            summary[output + "_avg_Accuracy_highly_expressed"] = acc
            summary[output + "_avg_Precision_highly_expressed"] = precision
            summary[output + "_avg_Recall_highly_expressed"] = recall
            summary[output + "_avg_F1_highly_expressed"] = f1

            summary[output + "_std_Accuracy_highly_expressed"] = acc_std
            summary[output + "_std_Precision_highly_expressed"] = precision_std
            summary[output + "_std_Recall_highly_expressed"] = recall_std
            summary[output + "_std_F1_highly_expressed"] = f1_std

            summary[output + "_avg_Accuracy_highly_expressed_disp"] = "{} +- {}".format(np.around(acc, 4), np.around(acc_std, 4))
            summary[output + "_avg_Precision_highly_expressed_disp"] = "{} +- {}".format(np.around(precision, 4), np.around(precision_std, 4))
            summary[output + "_avg_Recall_highly_expressed_disp"] = "{} +- {}".format(np.around(recall, 4), np.around(recall_std, 4))
            summary[output + "_avg_F1_highly_expressed_disp"] = "{} +- {}".format(np.around(f1, 4), np.around(f1_std, 4))

        for output in all_seeds_lowly_expressed_accuracy:
            acc = np.average(all_seeds_lowly_expressed_accuracy[output])

            acc_std = np.std(all_seeds_lowly_expressed_accuracy[output])

            print("{} avg Accuracy (lowly expressed) = {} +- {} ≈ {} +- {}".format(output, acc, acc_std, np.around(acc, 4), np.around(acc_std, 4)))
            print()

            summary[output + "_all_Accuracy_lowly_expressed"] = all_seeds_lowly_expressed_accuracy[output]

            summary[output + "_avg_Accuracy_lowly_expressed"] = acc

            summary[output + "_std_Accuracy_lowly_expressed"] = acc_std

            summary[output + "_avg_Accuracy_lowly_expressed_disp"] = "{} +- {}".format(np.around(acc, 4), np.around(acc_std, 4))

        for output in all_seeds_r2:
            r2 = np.average(all_seeds_r2[output])
            pearsonr = np.average(all_seeds_pearsonr[output])
            srho = np.average(all_seeds_srho[output])
            
            r2_std = np.std(all_seeds_r2[output])
            pearsonr_std = np.std(all_seeds_pearsonr[output])
            srho_std = np.std(all_seeds_srho[output])
            
            print("{} avg R2 = {} +- {} ≈ {} +- {}".format(output, r2, r2_std, np.around(r2, 4), np.around(r2_std, 4)))
            print("{} avg PearsonR = {} +- {} ≈ {} +- {}".format(output, pearsonr, pearsonr_std, np.around(pearsonr, 4), np.around(pearsonr_std, 4)))
            print("{} avg Spearman rho = {} +- {} ≈ {} +- {}".format(output, srho, srho_std, np.around(srho, 4), np.around(srho_std, 4)))
            print()
            
            summary[output + "_all_R2"] = all_seeds_r2[output]
            summary[output + "_all_PearsonR"] = all_seeds_pearsonr[output]
            summary[output + "_all_SpearmanR"] = all_seeds_srho[output]
            
            summary[output + "_avg_R2"] = r2
            summary[output + "_avg_PearsonR"] = pearsonr
            summary[output + "_avg_SpearmanR"] = srho
            
            summary[output + "_std_R2"] = r2_std
            summary[output + "_std_PearsonR"] = pearsonr_std
            summary[output + "_std_SpearmanR"] = srho_std
            
            summary[output + "_avg_R2_disp"] = "{} +- {}".format(np.around(r2, 4), np.around(r2_std, 4))
            summary[output + "_avg_PearsonR_disp"] = "{} +- {}".format(np.around(pearsonr, 4), np.around(pearsonr_std, 4))
            summary[output + "_avg_SpearmanR_disp"] = "{} +- {}".format(np.around(srho, 4), np.around(srho_std, 4))
        
        # save summary
        with open(os.path.join(summaries_save_dir, name_format + "_dlseed.json"), "w") as f:
            json.dump(summary, f, indent=4)

    print("Done!")

args = argparse.ArgumentParser()
args.add_argument("--config_path", type=str, default="./config.json", help="Path to config file")
args.add_argument("--model_name", type=str, default="MTLucifer", help="Name of model to use, must be one of {}".format(backbone_modules.get_all_backbone_names()))
args.add_argument("--modelling_strategy", type=str, required=True, help="Modelling strategy to use, either 'joint', 'pretrain+finetune', 'pretrain+linear_probing' or 'single_task'")

args.add_argument("--joint_tasks", type=str, default=None, help="Comma separated list of tasks to jointly train on")
args.add_argument("--pretrain_tasks", type=str, default=None, help="Comma separated list of tasks to pretrain on")
args.add_argument("--finetune_tasks", type=str, default=None, help="Comma separated list of tasks to finetune or perform linear probing on")
args.add_argument("--single_task", type=str, default=None, help="Task to train on")

args.add_argument("--shrink_test_set", action="store_true", help="Shrink large test sets (SuRE and ENCODETFChIPSeq) to 10 examples to make evaluation faster")

args.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
args.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
args.add_argument("--pretrain_lr", type=float, default=1e-5, help="pretrain learning rate")
args.add_argument("--pretrain_weight_decay", type=float, default=1e-4, help="pretrain weight decay")

args.add_argument("--batch_size", type=int, default=96, help="Batch size")
args.add_argument("--pretrain_batch_size", type=int, default=96, help="Pretrain batch size")

args.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs to joint-train, finetune or linear probe for")
args.add_argument("--pretrain_max_epochs", type=int, default=50, help="Maximum number of epochs to pretrain for")

args.add_argument("--train_mode", type=str, default="min_size", help="Specifies how multiple dataloaders are iterated over during training. Must be 'min_size' or 'max_size_cycle'")
args.add_argument("--pretrain_train_mode", type=str, default="min_size", help="Specifies how multiple dataloaders are iterated over during pretraining. Must be 'min_size' or 'max_size_cycle'")

args.add_argument("--num_random_seeds", type=int, default=1, help="Number of random seeds to train with")
args.add_argument("--use_existing_models", action="store_true", help="Use existing models if available")

args.add_argument("--wandb_project_name", type=str, default="promoter_modelling", help="Wandb project name")
args.add_argument("--metric_to_monitor", type=str, default="val_Fluorescence_mean_SpearmanR", help="Name of metric to monitor for early stopping")
args.add_argument("--metric_direction_which_is_optimal", type=str, default="max", help="Should metric be maximised (specify 'max') or minimised (specify 'min')?")
args.add_argument("--pretrain_metric_to_monitor", type=str, default="overall_val_loss", help="Name of pretrain metric to monitor for early stopping")
args.add_argument("--pretrain_metric_direction_which_is_optimal", type=str, default="min", help="Should pretrain metric be maximised (specify 'max') or minimised (specify 'min')?")

args.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
args.add_argument("--save_top_k", type=int, default=1, help="Number of top models to save")

args.add_argument("--fasta_shuffle_letters_path", type=str, default="fasta_shuffle_letters", help="Full path to the fasta_shuffle_letters executable")

args = args.parse_args()

assert os.path.exists(args.config_path), "Config file does not exist"
# Load config file
with open(args.config_path, "r") as f:
    config = json.load(f)

# setup wandb
root_dir = config["root_dir"]
wandb_logs_save_dir = os.path.join(root_dir, "wandb_logs")
if not os.path.exists(wandb_logs_save_dir):
    os.makedirs(wandb_logs_save_dir, exist_ok=True)
wandb_cache_dir = os.path.join(root_dir, "wandb_cache")
if not os.path.exists(wandb_cache_dir):
    os.makedirs(wandb_cache_dir, exist_ok=True)
os.environ["WANDB_DIR"] = wandb_logs_save_dir
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# train models
if "pretrain" in args.modelling_strategy:
    train_model(args, config, finetune=False)
    train_model(args, config, finetune=True)
else:
    train_model(args, config, finetune=False)

print("ALL DONE!")