# Strategies for effectively modelling promoter-driven gene expression using transfer learning

The code base for our work on building accurate models of promoter-driven gene expression by leveraging existing genomic data.

We give the complete list of commands needed to reproduce our main modelling results using this code base. 

By default, the modelling results and data are stored in the same directory as the code (./results and ./data respectively). Please change the config.json file if they are to be stored elsewhere. 

Our dataloaders are designed to download the necessary raw data files from the internet or a Google Drive. However, the raw ENCODE TF-binding data files need to be manually downloaded due to their size from here - https://drive.google.com/file/d/1VLVjRf3nFLSYEamTKm_zN5cObdS1BJc3/view?usp=share_link. Then, extract them to a subdirectory of the data directory called "ENCODETFChIPSeq_data".

## Commands to be run to reproduce results

### Install the package and dependencies:

Run the following commands to create a mamba/conda environment and install PyTorch. Further dependencies will be installed automatically when the package is installed.
```
mamba create -n promoter_models pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba activate promoter_models
```

Also run the following commands for reproducible results:
```
conda env config vars set CUBLAS_WORKSPACE_CONFIG=:4096:8
mamba deactivate
mamba activate promoter_models
```

To run the Malinois model, you will need to install the boda package before installing our package as follows:
```
git clone https://github.com/sjgosai/boda2.git
cd boda2/
pip install -e .
```

Run the following command to install our package:
```bash
pip install -e .
```

### Commands to reproduce architecture validation results:
#### Using the fluorescence data:

To get results using an MTLucifer model:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

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

To get results using a ResNet model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name ResNet --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode min_size --num_random_seeds 5 --use_existing_models
```

To get results using a LegNet model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name LegNet --modelling_strategy single_task --single_task FluorescenceData --lr 0.005 --weight_decay 0.01 --batch_size 1024 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a larger LegNet model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name LegNetLarge --modelling_strategy single_task --single_task FluorescenceData --lr 0.005 --weight_decay 0.01 --batch_size 192 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using an MPRAnn model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name MPRAnn --modelling_strategy single_task --single_task FluorescenceData --lr 0.001 --weight_decay 0.0 --batch_size 32 --max_epochs 50 --train_mode min_size --num_random_seeds 5 --use_existing_models
```

To get results using a Malinois model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name Malinois --modelling_strategy single_task --single_task FluorescenceData --lr 0.0032658700881052086 --weight_decay 0.0003438210249762151 --batch_size 512 --max_epochs 200 --num_random_seeds 5 --use_existing_models --patience 30
```

To get results using a randomly initialized DNABERT model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name DNABERTRandomInit --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

To get results using a randomly initialized Enformer model:
```
python train_models_with_different_fluorescence_data_splits.py --model_name EnformerRandomInit --modelling_strategy single_task --single_task FluorescenceData --lr 5e-4 --weight_decay 5e-4 --batch_size 96 --max_epochs 50 --train_mode min_size --num_random_seeds 5 --use_existing_models
```

#### Using the Malinois MPRA:
Similarly, to get results using the Malinois MPRA data, run the following commands:
```
python train_models.py --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode min_size --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name MotifBasedFCN --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name MotifBasedFCNLarge --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name PureCNN --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name PureCNNLarge --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name ResNet --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name LegNet --modelling_strategy single_task --single_task Malinois_MPRA --lr 0.05 --weight_decay 0.01 --batch_size 1024 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name LegNetLarge --modelling_strategy single_task --single_task Malinois_MPRA --lr 0.01 --weight_decay 0.01 --batch_size 192 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name MPRAnn --modelling_strategy single_task --single_task Malinois_MPRA --lr 0.001 --weight_decay 0.0 --batch_size 32 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name Malinois --modelling_strategy single_task --single_task Malinois_MPRA --lr 0.0032658700881052086 --weight_decay 0.0003438210249762151 --batch_size 1076 --max_epochs 200 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models --patience 30

python train_models.py --model_name DNABERTRandomInit --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name EnformerRandomInit --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode min_size --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models
```

### Evaluating all training strategies:
#### Using the fluorescence data:
Pretaining on various tasks using MTLucifer and then performing linear probing on the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks RNASeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks ENCODETFChIPSeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 32 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Pretaining on various tasks using MTLucifer and then performing fine-tuning on the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks RNASeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks ENCODETFChIPSeq --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 32 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 5 --use_existing_models
```

Joint training on various tasks along with the fluorescence data:
```
python train_models_with_different_fluorescence_data_splits.py --modelling_strategy joint --joint_tasks RNASeq,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 32 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy joint --joint_tasks ENCODETFChIPSeq,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 32 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy joint --joint_tasks Sharpr_MPRA,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy joint --joint_tasks SuRE_classification,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 12 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --modelling_strategy joint --joint_tasks SuRE_classification,Sharpr_MPRA,FluorescenceData --shrink_test_set --lr 1e-5 --weight_decay 1e-4 --batch_size 12 --train_mode "min_size" --max_epochs 50 --num_random_seeds 5 --use_existing_models
```

Fine-tuning DNABERT and Enformer models on the fluorescence data (the last command does linear probing using Lasso on Enformer outputs):
```
python train_models_with_different_fluorescence_data_splits.py --model_name DNABERT --modelling_strategy single_task --single_task FluorescenceData --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py  --model_name Enformer --modelling_strategy single_task --single_task FluorescenceData --lr 1e-3 --weight_decay 5e-4 --batch_size 96 --max_epochs 50 --train_mode min_size --num_random_seeds 5 --use_existing_models

python train_models_with_different_fluorescence_data_splits.py --model_name EnformerFullFrozenBase --modelling_strategy single_task_simple_regression --single_task FluorescenceData --use_existing_models --num_random_seeds 5
```


#### Using the Malinois MPRA:
Pretaining on various tasks using MTLucifer and then performing linear probing on the Malinois MPRA data:
```
python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks RNASeq --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks ENCODETFChIPSeq --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 64 --pretrain_max_epochs 10 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks Sharpr_MPRA --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+linear_probing --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-3 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"
```

Pretaining on various tasks using MTLucifer and then performing fine-tuning on the Malinois MPRA data:
```
python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks RNASeq --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks ENCODETFChIPSeq --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 64 --pretrain_max_epochs 10 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks Sharpr_MPRA --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 96 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy pretrain+finetune --pretrain_tasks SuRE_classification,Sharpr_MPRA,lentiMPRA --finetune_tasks Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode "min_size" --pretrain_lr 1e-5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 24 --pretrain_max_epochs 50 --pretrain_train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"
```

Joint training on various tasks along with the Malinois MPRA data:
```
python train_models.py --modelling_strategy joint --joint_tasks RNASeq,Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 32 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy joint --joint_tasks ENCODETFChIPSeq,Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 32 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy joint --joint_tasks Sharpr_MPRA,Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy joint --joint_tasks SuRE_classification,Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 8 --max_epochs 50 --train_mode "min_size" --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"

python train_models.py --modelling_strategy joint --joint_tasks SuRE_classification,Sharpr_MPRA,Malinois_MPRA --shrink_test_set --lr 1e-4 --weight_decay 1e-4 --batch_size 8 --train_mode "min_size" --max_epochs 50 --num_random_seeds 1 --use_existing_models --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR"
```

Fine-tuning DNABERT and Enformer models on the Malinois MPRA data (the last command does linear probing using Lasso on Enformer outputs):
```
python train_models.py --model_name DNABERT --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-5 --weight_decay 1e-4 --batch_size 64 --max_epochs 50 --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name Enformer --modelling_strategy single_task --single_task Malinois_MPRA --lr 1e-4 --weight_decay 1e-4 --batch_size 96 --max_epochs 50 --train_mode min_size --num_random_seeds 1 --metric_to_monitor "val_MalinoisMPRA_mean_SpearmanR" --use_existing_models

python train_models.py --model_name EnformerFullFrozenBase --modelling_strategy single_task_simple_regression --single_task Malinois_MPRA --use_existing_models --num_random_seeds 5
```
