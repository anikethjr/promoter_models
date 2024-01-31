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

import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from promoter_modelling.dataloaders import FluorescenceData, FluorescenceData_classification, FluorescenceData_with_motifs, FluorescenceData_DNABERT, \
                                           LL100, CCLE, Roadmap, Sharpr_MPRA, SuRE, ENCODETFChIPSeq, STARRSeq, Malinois_MPRA, Malinois_MPRA_DNABERT, Malinois_MPRA_with_motifs, lentiMPRA
from promoter_modelling import backbone_modules
from promoter_modelling import MTL_modules
from promoter_modelling.utils import fasta_utils
from promoter_modelling.utils import misc_utils

np.random.seed(97)
torch.manual_seed(97)

def perform_analysis(args, config, finetune=False):
    # create directories
    root_dir = config["root_dir"]
    motif_insertion_analysis_dir = os.path.join(root_dir, "motif_insertion_analysis")
    if not os.path.exists(motif_insertion_analysis_dir):
        os.mkdir(motif_insertion_analysis_dir)
    # should already exist
    root_data_dir = config["root_data_dir"]
    common_cache_dir = os.path.join(root_data_dir, "common")
    model_save_dir = os.path.join(root_dir, "saved_models")
    
    # create data loaders
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

    # instantiate dataloaders
    dataloaders = {}
    print("Instantiating dataloaders...")
    for task in tasks:
        if task == "all_tasks" or task == "RNASeq": # special task names
            dataloaders[task] = []
            tasks_set = None
            if args.modelling_strategy.startswith("pretrain"):
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
                                                                            cache_dir=os.path.join(root_data_dir, "lentiMPRA", \
                                                                            common_cache_dir=common_cache_dir, 
                                                                            shrink_test_set=args.shrink_test_set)))
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
                    elif (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
                        dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                                        use_construct=True))
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
                    elif (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
                        dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                                        use_construct=True, \
                                                                                        predict_DE=True))
                    else:
                        dataloaders[task].append(FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                                        predict_DE=True))
                elif t == "FluorescenceData_classification":
                    dataloaders[task].append(FluorescenceData_classification.FluorescenceDataLoader(batch_size=batch_size, \
                                                                                                    cache_dir=os.path.join(root_data_dir, "FluorescenceData_classification")))
                elif t == "Malinois_MPRA":
                    if args.model_name.startswith("MotifBased"):
                        dataloaders[task].append(Malinois_MPRA_with_motifs.MalinoisMPRADataLoader(batch_size=batch_size, \
                                                                                            cache_dir=os.path.join(root_data_dir, "Malinois_MPRA"), \
                                                                                            common_cache_dir=common_cache_dir))
                    elif "DNABERT" in args.model_name:
                        dataloaders[task].append(Malinois_MPRA_DNABERT.MalinoisMPRADataLoader(batch_size=batch_size, \
                                                                                            cache_dir=os.path.join(root_data_dir, "Malinois_MPRA"), \
                                                                                            common_cache_dir=common_cache_dir))
                    else:
                        dataloaders[task].append(Malinois_MPRA.MalinoisMPRADataLoader(batch_size=batch_size, \
                                                                                        cache_dir=os.path.join(root_data_dir, "Malinois_MPRA"), \
                                                                                        common_cache_dir=common_cache_dir))
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
                                                                cache_dir=os.path.join(root_data_dir, "lentiMPRA"), \
                                                                common_cache_dir=common_cache_dir, 
                                                                shrink_test_set=args.shrink_test_set)
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
            elif (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
                dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                            use_construct=True)
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
            elif (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
                dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                            cache_dir=os.path.join(root_data_dir, "FluorescenceData_DE"), \
                                                                            use_construct=True, \
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
                                                                        return_specified_cells=[0])
        elif task == "FluorescenceData_K562":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=[1])
        elif task == "FluorescenceData_THP1":
            dataloaders[task] = FluorescenceData.FluorescenceDataLoader(batch_size=batch_size, \
                                                                        cache_dir=os.path.join(root_data_dir, "FluorescenceData"), \
                                                                        return_specified_cells=[2])
        elif task == "Malinois_MPRA":
            if args.model_name.startswith("MotifBased"):
                dataloaders[task] = Malinois_MPRA_with_motifs.MalinoisMPRADataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "Malinois_MPRA"), \
                                                                                    common_cache_dir=common_cache_dir)
            elif "DNABERT" in args.model_name:
                dataloaders[task] = Malinois_MPRA_DNABERT.MalinoisMPRADataLoader(batch_size=batch_size, \
                                                                                    cache_dir=os.path.join(root_data_dir, "Malinois_MPRA"), \
                                                                                    common_cache_dir=common_cache_dir)
            else:
                dataloaders[task] = Malinois_MPRA.MalinoisMPRADataLoader(batch_size=batch_size, \
                                                                                cache_dir=os.path.join(root_data_dir, "Malinois_MPRA"), \
                                                                                common_cache_dir=common_cache_dir)
    
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
        lr = args.pretrain_lr
        weight_decay = args.pretrain_weight_decay
        batch_size = args.pretrain_batch_size
        max_epochs = args.pretrain_max_epochs
        train_mode = args.pretrain_train_mode
    else:
        metric_to_monitor = args.metric_to_monitor
        metric_direction_which_is_optimal = args.metric_direction_which_is_optimal
        lr = args.lr
        weight_decay = args.weight_decay
        batch_size = args.batch_size
        max_epochs = args.max_epochs
        train_mode = args.train_mode

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
        if "simple_regression" in args.modelling_strategy:
            name_format = "simple_regression_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
    elif "pretrain" in args.modelling_strategy and not finetune:
        name_format = "pretrain_on_{}".format("+".join(tasks))
    elif "joint" in args.modelling_strategy:
        name_format = "joint_train_on_{}".format("+".join(tasks))
    elif "single" in args.modelling_strategy:
        if "simple_regression" in args.modelling_strategy:
            name_format = "simple_regression_on_{}".format("+".join(tasks))
        else:
            name_format = "individual_training_on_{}".format("+".join(tasks))

    # map to model classes
    model_class = backbone_modules.get_backbone_class(args.model_name)
    if args.model_name != "MTLucifer":
        name_format = f"{args.model_name}_" + name_format

    # add optional name suffix to model name - only when not pretraining
    if args.optional_name_suffix is not None:
        if "pretrain" in args.modelling_strategy:
            if finetune:
                name_format += "_" + args.optional_name_suffix
        else:
            name_format += "_" + args.optional_name_suffix

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
    mtlpredictor = MTL_modules.MTLPredictor(model_class=backbone_modules.MTLucifer, \
                                            all_dataloader_modules=all_dataloaders, \
                                            batch_size=batch_size, \
                                            use_preconstructed_dataloaders=True).to(device)

    mtlpredictor.model.load_state_dict(new_state_dict, strict=False)        
    print("Loaded best trained model")

    # load meme file
    meme_file = args.meme_file
    
    # parse meme file
    motifs = misc_utils.parse_meme_file(meme_file)
    
    print("Num motifs = {}".format(len(motifs)))

    assert all_dataloaders[-1].name == "Fluorescence", "Last task must be FluorescenceData - the anaylsis is performed using the fluorescence data. You need models trained using the train_models.py script with the last task as FluorescenceData."

    all_fluorescence_measurements = all_dataloaders[-1].merged

    # ascertain the importances of each motif
    if not os.path.exists(os.path.join(motif_insertion_analysis_dir, "{}.tsv".format(name_format))):
        np.random.seed(97)

        for i, motif in tqdm(enumerate(motifs)):
            print(motif.motif_name)

            all_motif_inserted_sequences_raw = [misc_utils.insert_motif_into_sequence(seq, motif, args.num_motif_insertions_per_sequence) \
                                            for seq in all_fluorescence_measurements["sequence"]]

            # get the fluorescence predictions for each sequence with the motif inserted
            all_motif_inserted_sequences = [fasta_utils.one_hot_encode(seq).astype(np.float32) for seq in all_motif_inserted_sequences_raw]

            preds = []

            for j in range(0, len(all_motif_inserted_sequences), batch_size):
                batch = np.array(all_motif_inserted_sequences[j:min(j+batch_size, len(all_motif_inserted_sequences))])
                batch = torch.tensor(batch).to(device)
                with torch.no_grad():
                    pred = mtlpredictor.model(batch)[0][-1]
                preds.append(pred.cpu().numpy())

            preds = np.concatenate(preds, axis=0)

            all_fluorescence_measurements["motif_inserted_sequence_for_{}".format(motif.motif_name)] = all_motif_inserted_sequences_raw
            for output_name, pred in zip(all_dataloaders[-1].output_names, preds.T):
                all_fluorescence_measurements["{}_motif_insertion_prediction_for_{}".format(output_name, motif.motif_name)] = pred
                
            # get baseline predictions
            baseline_predictions = []
            all_baseline_sequences = [fasta_utils.one_hot_encode(seq).astype(np.float32) for seq in all_fluorescence_measurements["sequence"]]
            for j in tqdm(range(0, all_fluorescence_measurements.shape[0], batch_size)):
                batch = np.array(all_baseline_sequences[j:min(j+batch_size, len(all_baseline_sequences))])
                batch = torch.tensor(batch).to(device)
                with torch.no_grad():
                    pred = mtlpredictor.model(batch)[0][-1]
                baseline_predictions.append(pred.cpu().numpy())

            baseline_predictions = np.concatenate(baseline_predictions, axis=0)

            for output_name, bpred in zip(all_dataloaders[-1].output_names, baseline_predictions.T):
                all_fluorescence_measurements["baseline_{}_prediction".format(output_name)] = bpred
                
            if i % 25 == 0:
                all_fluorescence_measurements = all_fluorescence_measurements.copy()

        # save the results
        all_fluorescence_measurements.to_csv(os.path.join(motif_insertion_analysis_dir, "{}.tsv".format(name_format)), sep="\t", index=False)
    
    else:
        print("Motif insertion results already exist")

args = argparse.ArgumentParser()
args.add_argument("--config_path", type=str, default="./config.json", help="Path to config file")
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

args.add_argument("--meme_file", type=str, required=True, help="Path to meme file containing motifs to analyse")
args.add_argument("--num_motif_insertions_per_sequence", type=int, default=10, help="Number of motif insertions to perform per sequence")

args = args.parse_args()

assert os.path.exists(args.config_path), "Config file does not exist"
# Load config file
with open(args.config_path, "r") as f:
    config = json.load(f)

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# train models
if "pretrain" in args.modelling_strategy:
    perform_analysis(args, config, finetune=True)
else:
    perform_analysis(args, config, finetune=False)