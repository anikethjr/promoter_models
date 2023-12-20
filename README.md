# Pretraining strategies for effective promoter-driven gene expression prediction

The code base for our work on building accurate models of promoter-driven gene expression by leveraging existing genomic data. We explain how to reproduce our results in detail and also point to our best models and preprocessed data. Please cite this work if you use our code or data:

```bibtex
@article {reddyherschletal2023,
	author = {Reddy, Aniketh Janardhan and Herschl, Michael H and Kolli, Sathvik and Lu, Amy X and Geng, Xinyang and Kumar, Aviral and Hsu, Patrick D and Levine, Sergey and Ioannidis, Nilah M},
	title = {Pretraining strategies for effective promoter-driven gene expression prediction},
	elocation-id = {2023.02.24.529941},
	year = {2023},
	doi = {10.1101/2023.02.24.529941},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/02/27/2023.02.24.529941},
	eprint = {https://www.biorxiv.org/content/early/2023/02/27/2023.02.24.529941.full.pdf},
	journal = {bioRxiv}
}
```

We give the complete list of commands needed to reproduce our main modelling results using this code base. 

By default, the modelling results and data are stored in the same directory as the code (./results and ./data respectively). Please change the config.json file if they are to be stored elsewhere. 

Our dataloaders are designed to download the necessary raw data files from the internet or a Google Drive. However, the raw ENCODE TF-binding data files need to be manually downloaded due to their size from here - https://drive.google.com/file/d/1VLVjRf3nFLSYEamTKm_zN5cObdS1BJc3/view?usp=share_link. Then, extract them to a subdirectory of the data directory called "ENCODETFChIPSeq_data".

Now, downloading the raw data files and preprocessing them is a time-consuming task, especially for the larger datasets. Thus, we provide a folder containing most of the preprocessed data here - https://drive.google.com/file/d/1Cs0KamYJy-qq3HaPsUMNoe6j8IB1QCJV/view?usp=share_link. Simply extract the contents of this file to the data directory.

We also provide the best model checkpoints for the pretrained and jointly trained models to enable quick replication of our results. Download them from here - https://drive.google.com/file/d/1zbDoHd4k9FXeVQ2BxdrDw0WfS6mt74wJ/view?usp=share_link. Then, extract them to a subdirectory of the results directory called "saved_models".

## Commands to be run to reproduce results

### Install the package and dependencies:

Run the following commands to create a mamba/conda environment and install PyTorch. Further dependencies will be installed automatically when the package is installed.
```
mamba create -n promoter_modelling pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba activate promoter_modelling
```

Also run the following commands for reproducible results:
```
conda env config vars set CUBLAS_WORKSPACE_CONFIG=:4096:8
mamba deactivate
mamba activate promoter_modelling
```

Run the following command to install the package:
```
python -m pip install -e .
```

### Commands to reproduce architecture validation results:
To get results using a motif-based FCN model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name MotifBasedFCN --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a larger motif-based FCN model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name MotifBasedFCNLarge --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a pure CNN model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name PureCNN --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a larger pure CNN model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name PureCNNLarge --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results by finetuning a pretrained DNABERT model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name DNABERT --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a LegNet model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name LegNet --modelling_strategy single_task --single_task FluorescenceData --lr 0.005 --weight_decay 0.01 --batch_size 1024 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a larger LegNet model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name LegNetLarge --modelling_strategy single_task --single_task FluorescenceData --lr 0.005 --weight_decay 0.01 --batch_size 192 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using our CNN+Transformer model that we use in all further analyses:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

### Evaluating all training strategies:
Training models on just the fluorescence data:
```
python train_models.py --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Pretaining on various tasks and then performing linear probing on the fluorescence data:
```
python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks RNASeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks ENCODETFChIPSeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 10 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 20 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 10 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 8 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Pretaining on various tasks and then performing fine-tuning on the fluorescence data:
```
python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks RNASeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks ENCODETFChIPSeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 10 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Joint training on various tasks along with the fluorescence data:
```
python train_models.py --modelling_strategy joint --joint_tasks RNASeq,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models

python train_models.py --modelling_strategy joint --joint_tasks ENCODETFChIPSeq,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models

python train_models.py --modelling_strategy joint --joint_tasks Sharpr_MPRA,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models

python train_models.py --modelling_strategy joint --joint_tasks SuRE_classification,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 12 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models

python train_models.py --modelling_strategy joint --joint_tasks SuRE_classification,Sharpr_MPRA,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 12 --train_mode "min_size" --max_epochs 50 --num_random_seeds 1 --use_existing_models
```

### Training using multiple splits of the fluorescence dataset:

Training models on just the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Pretaining on various tasks and then performing linear probing on the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks RNASeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks ENCODETFChIPSeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 64 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Pretaining on various tasks and then performing fine-tuning on the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks RNASeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks ENCODETFChIPSeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 64 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

### Running the motif insertion analysis using the best model:
#### Performing the motif insertion analysis using the models pretrained on all MPRA data and then fine-tuned on the fluorescence data:
Command to perform the analysis:
```
python perform_motif_insertion_analysis.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --num_random_seeds 5 --num_motif_insertions_per_sequence 10 --meme_file all_motifs.meme --batch_size 2048
```
Command to summarize the results:
```
python get_motifs_attended_to_by_model.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData
```

#### Performing the motif insertion analysis using the models trained from scratch on the fluorescence data:
Command to perform the analysis:
```
python perform_motif_insertion_analysis.py --modelling_strategy single_task --single_task FluorescenceData --num_random_seeds 5 --num_motif_insertions_per_sequence 10 --meme_file all_motifs.meme --batch_size 2048
```
Command to summarize the results:
```
python get_motifs_attended_to_by_model.py --modelling_strategy single_task --single_task FluorescenceData
```

### Evaluating performance on the fluorescence data using the purely pretrained models:
Command to look at goodness of K-562 fluorescence predictions after pretraining on MPRA data (other cell lines don't have MPRA data):
```
python eval_transfer_performance.py --modelling_strategy pretrain --pretrain_tasks SuRE_classification,Sharpr_MPRA --output_inds_to_compare "-1,0,-1 -1,2,-1"
```

Command to look at goodness of fluorescence predictions after pretraining on RNASeq data (Roadmap doesn't have data on JURKAT and THP1 cells):
```
python eval_transfer_performance.py --modelling_strategy pretrain --pretrain_tasks RNASeq --output_inds_to_compare "30,34,89 983,544,143 -1,54,-1"
```

### Training using multiple splits of the fluorescence dataset for the classification task:
Training models on just the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy single_task --single_task FluorescenceData_classification --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models --metric_to_monitor val_Fluorescence_classification_mean_Accuracy
```

Pretaining on various tasks and then performing fine-tuning on the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData_classification --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 8 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models --metric_to_monitor val_Fluorescence_classification_mean_Accuracy
```