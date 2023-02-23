import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

import torchmtl

np.random.seed(97)
torch.manual_seed(97)

# combines multiple dataloaders into one
class MTLDataLoader(pl.LightningDataModule):
    # change min_size to max_size_cycle to iterate through largest dataset fully during each training epoch
    # more details: https://github.com/Lightning-AI/lightning/blob/15ef52bc732d1f907de4de58683f131652c0d68c/src/pytorch_lightning/trainer/supporters.py
    def __init__(self, all_dataloaders, train_mode="min_size"):
        super().__init__()
        
        self.all_train_ds = CombinedLoader([i.train_dataloader() for i in all_dataloaders], train_mode)
        self.all_test_ds = [i.test_dataloader() for i in all_dataloaders]
        self.all_val_ds = [i.val_dataloader() for i in all_dataloaders]
        
    def train_dataloader(self):
        return self.all_train_ds

    def test_dataloader(self):
        return self.all_test_ds
    
    def val_dataloader(self):
        return self.all_val_ds

    def predict_dataloader(self):
        return self.all_test_ds

'''
Main class used to train models
model_class: class of model to be trained
all_dataloader_modules: list of dataloader modules to be used for training - can either be a list of dataloader modules or a list of dataloaders
batch_size: batch size to be used for training
'''

# maybe make loss_fn and task_type attributes of individual dataloaders?
class MTLPredictor(pl.LightningModule):
    def __init__(self, \
                 model_class, \
                 all_dataloader_modules, \
                 batch_size, \
                 n_cpus=8, \
                 lr=1e-5, \
                 weight_decay=1e-4, \
                 with_motifs=False, \
                 use_preconstructed_dataloaders=False, \
                 train_mode="min_size"):
        super().__init__()
        
        # create MTLDataLoader from individual DataLoaders
        print("Creating MTLDataLoader from individual DataLoaders")
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        self.all_dataloader_modules = all_dataloader_modules
        self.all_dataloaders = []
        if use_preconstructed_dataloaders:
            self.all_dataloaders = self.all_dataloader_modules
        else:
            for module in self.all_dataloader_modules:
                self.all_dataloaders.append(module(batch_size=self.batch_size, n_cpus=self.n_cpus))
        
        self.num_tasks = len(self.all_dataloader_modules)
        self.train_mode = train_mode
        self.mtldataloader = MTLDataLoader(self.all_dataloaders, self.train_mode)
                
        # do inputs include motif occurrences?
        self.with_motifs = with_motifs        
        if self.with_motifs:
            self.num_motifs = self.all_dataloaders[0].num_motifs
            self.backbone_model = model_class(num_motifs=self.num_motifs)
        else:
            self.backbone_model = model_class()
            
        # create MTL model
        model_config = [
                            {
                                'name': "Backbone",
                                'layers': self.backbone_model,
                                # No anchor_layer means this layer receives input directly
                            }
                        ]
        
        if self.num_tasks == 1: # we don't want the loss scaling when there's only one loss term
            print("Single dataloader model")
            task = self.all_dataloaders[0]
            model_config.append({
                                    'name': task.name,
                                    'layers': nn.Linear(self.backbone_model.embed_dims, task.num_outputs),
                                    'loss': task.loss_fn,
                                    'loss_weight': torch.tensor(0.0),
                                    'anchor_layer': 'Backbone'
                                })
        else:
            for i, task in enumerate(self.all_dataloaders):
                model_config.append({
                                        'name': task.name,
                                        'layers': nn.Linear(self.backbone_model.embed_dims, task.num_outputs),
                                        'loss': task.loss_fn,
                                        # 'loss_weight': torch.tensor(0.0),
                                        'loss_weight': 'auto',
                                        'loss_init_val': 0.0,
                                        'anchor_layer': 'Backbone'
                                    })
            
        self.model = torchmtl.MTLModel(model_config, output_tasks=[task.name for task in self.all_dataloaders])
        
        # optimizer hyperparams
        self.lr = lr
        self.weight_decay = weight_decay

    def get_mtldataloader(self):
        return self.mtldataloader
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        for i, dl_batch in enumerate(batch):
            loss = 0.0

            if self.with_motifs:
                X, motifs, y = dl_batch
                y_hat, l_funcs, l_weights = self([X, motifs])
            elif self.all_dataloader_modules[i].with_mask:
                X, y, mask = dl_batch
                y_hat, l_funcs, l_weights = self(X)
            else:
                X, y = dl_batch
                y_hat, l_funcs, l_weights = self(X)

            # derived from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
            # implements https://arxiv.org/abs/1705.07115
            std = torch.exp(l_weights[i])**(1/2)
            is_regression = int(self.all_dataloader_modules[i].task == "regression")
            coeff = 1 / ((is_regression+1)*(std**2))
                        
            if self.all_dataloader_modules[i].with_mask:
                loss += coeff * l_funcs[i](y_hat[i][mask], y[mask]) + torch.log(std)
            else:
                if self.all_dataloader_modules[i].task == "classification" and self.all_dataloaders[i].use_1hot_for_classification:
                    s = 0
                    for j, output in enumerate(self.all_dataloaders[i].output_names):
                        num_outputs_for_this = self.all_dataloaders[i].num_classes_per_output[j]
                        loss += coeff * l_funcs[i](y_hat[i][:, s:s+num_outputs_for_this], y[:, j]) + torch.log(std)
                        s += num_outputs_for_this
                else:
                    loss += coeff * l_funcs[i](y_hat[i], y) + torch.log(std)
            
            self.log("{}_train_loss".format(self.all_dataloaders[i].name), loss, on_step=True, on_epoch=True, logger=True)

            total_loss += loss
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, logger=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.with_motifs:
            X, motifs, y = batch
        elif self.all_dataloader_modules[dataloader_idx].with_mask:
            X, y, mask = batch
        else:
            X, y = batch
        
        # get predictions
        if len(self.all_dataloaders[dataloader_idx].promoter_windows_relative_to_TSS) > 1: # average predictions over all windows
            pred = None
            l_funcs = None
            l_weights = None
            for i in range(X.shape[1]):
                if self.with_motifs:
                    y_hat, l_funcs, l_weights = self([X[:, i], motifs[:, i]])
                else:
                    y_hat, l_funcs, l_weights = self(X[:, i])
                if pred is None:
                    pred = y_hat[dataloader_idx]
                else:
                    pred += y_hat[dataloader_idx]
            pred /= X.shape[1]

            # derived from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
            # implements https://arxiv.org/abs/1705.07115
            std = torch.exp(l_weights[dataloader_idx])**(1/2)
            is_regression = int(self.all_dataloader_modules[dataloader_idx].task == "regression")
            coeff = 1 / ((is_regression+1)*(std**2))

            if self.all_dataloader_modules[dataloader_idx].with_mask:
                loss = coeff * l_funcs[dataloader_idx](pred[mask], y[mask]) + torch.log(std)
            else:
                if self.all_dataloader_modules[dataloader_idx].task == "classification" and self.all_dataloaders[dataloader_idx].use_1hot_for_classification:
                    s = 0
                    loss = 0
                    for j, output in enumerate(self.all_dataloaders[dataloader_idx].output_names):
                        num_outputs_for_this = self.all_dataloaders[dataloader_idx].num_classes_per_output[j]
                        loss += coeff * l_funcs[dataloader_idx](y_hat[dataloader_idx][:, s:s+num_outputs_for_this], y[:, j]) + torch.log(std)
                        s += num_outputs_for_this
                else:
                    loss = coeff * l_funcs[dataloader_idx](pred, y) + torch.log(std)
        else: # only one window/sequence
            l_funcs = None
            l_weights = None
            if self.with_motifs:
                y_hat, l_funcs, l_weights = self([X.squeeze(1), motifs.squeeze(1)])
            else:
                y_hat, l_funcs, l_weights = self(X.squeeze(1))            
            pred = y_hat[dataloader_idx]

            # derived from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
            # implements https://arxiv.org/abs/1705.07115
            std = torch.exp(l_weights[dataloader_idx])**(1/2)
            is_regression = int(self.all_dataloader_modules[dataloader_idx].task == "regression")
            coeff = 1 / ((is_regression+1)*(std**2))
            
            if self.all_dataloader_modules[dataloader_idx].with_mask:
                loss = coeff * l_funcs[dataloader_idx](pred[mask], y[mask]) + torch.log(std)
            else:
                if self.all_dataloader_modules[dataloader_idx].task == "classification" and self.all_dataloaders[dataloader_idx].use_1hot_for_classification:
                    s = 0
                    loss = 0
                    for j, output in enumerate(self.all_dataloaders[dataloader_idx].output_names):
                        num_outputs_for_this = self.all_dataloaders[dataloader_idx].num_classes_per_output[j]
                        loss += coeff * l_funcs[dataloader_idx](pred[:, s:s+num_outputs_for_this], y[:, j]) + torch.log(std)
                        s += num_outputs_for_this
                else:
                    loss = coeff * l_funcs[dataloader_idx](pred, y) + torch.log(std)

        # update metrics
        self.all_dataloaders[dataloader_idx].update_metrics(pred.cpu().detach(), y.cpu().detach(), loss.cpu().detach(), "val")

        self.log("{}_val_loss".format(self.all_dataloaders[dataloader_idx].name), loss)
    
    def validation_epoch_end(self, val_step_outputs):
        overall_loss = 0
        for i, dl in enumerate(self.all_dataloaders):
            dl_metrics = dl.compute_metrics("val")
            self.log_dict(dl_metrics, on_step=False, on_epoch=True, logger=True) 
            overall_loss += dl_metrics["val_{}_avg_epoch_loss".format(dl.name)]

        self.log("overall_val_loss", overall_loss)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.with_motifs:
            X, motifs, y = batch
        elif self.all_dataloader_modules[dataloader_idx].with_mask:
            X, y, mask = batch
        else:
            X, y = batch
        
        # get predictions
        if len(self.all_dataloaders[dataloader_idx].promoter_windows_relative_to_TSS) > 1: # average predictions over all windows
            pred = None
            l_funcs = None
            l_weights = None
            for i in range(X.shape[1]):
                if self.with_motifs:
                    y_hat, l_funcs, l_weights = self([X[:, i], motifs[:, i]])
                else:
                    y_hat, l_funcs, l_weights = self(X[:, i])
                if pred is None:
                    pred = y_hat[dataloader_idx]
                else:
                    pred += y_hat[dataloader_idx]
            pred /= X.shape[1]

            # derived from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
            # implements https://arxiv.org/abs/1705.07115
            std = torch.exp(l_weights[dataloader_idx])**(1/2)
            is_regression = int(self.all_dataloader_modules[dataloader_idx].task == "regression")
            coeff = 1 / ((is_regression+1)*(std**2))

            if self.all_dataloader_modules[dataloader_idx].with_mask:
                loss = coeff * l_funcs[dataloader_idx](pred[mask], y[mask]) + torch.log(std)
            else:
                if self.all_dataloader_modules[dataloader_idx].task == "classification" and self.all_dataloaders[dataloader_idx].use_1hot_for_classification:
                    s = 0
                    loss = 0
                    for j, output in enumerate(self.all_dataloaders[dataloader_idx].output_names):
                        num_outputs_for_this = self.all_dataloaders[dataloader_idx].num_classes_per_output[j]
                        loss += coeff * l_funcs[dataloader_idx](y_hat[dataloader_idx][:, s:s+num_outputs_for_this], y[:, j]) + torch.log(std)
                        s += num_outputs_for_this
                else:
                    loss = coeff * l_funcs[dataloader_idx](pred, y) + torch.log(std)
        else: # only one window/sequence
            l_funcs = None
            l_weights = None
            if self.with_motifs:
                y_hat, l_funcs, l_weights = self([X.squeeze(1), motifs.squeeze(1)])
            else:
                y_hat, l_funcs, l_weights = self(X.squeeze(1))            
            pred = y_hat[dataloader_idx]

            # derived from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
            # implements https://arxiv.org/abs/1705.07115
            std = torch.exp(l_weights[dataloader_idx])**(1/2)
            is_regression = int(self.all_dataloader_modules[dataloader_idx].task == "regression")
            coeff = 1 / ((is_regression+1)*(std**2))
            
            if self.all_dataloader_modules[dataloader_idx].with_mask:
                loss = coeff * l_funcs[dataloader_idx](pred[mask], y[mask]) + torch.log(std)
            else:
                if self.all_dataloader_modules[dataloader_idx].task == "classification" and self.all_dataloaders[dataloader_idx].use_1hot_for_classification:
                    s = 0
                    loss = 0
                    for j, output in enumerate(self.all_dataloaders[dataloader_idx].output_names):
                        num_outputs_for_this = self.all_dataloaders[dataloader_idx].num_classes_per_output[j]
                        loss += coeff * l_funcs[dataloader_idx](pred[:, s:s+num_outputs_for_this], y[:, j]) + torch.log(std)
                        s += num_outputs_for_this
                else:
                    loss = coeff * l_funcs[dataloader_idx](pred, y) + torch.log(std)
        
        # update metrics
        self.all_dataloaders[dataloader_idx].update_metrics(pred.cpu().detach(), y.cpu().detach(), loss.cpu().detach(), "test")

        self.log("{}_test_loss".format(self.all_dataloaders[dataloader_idx].name), loss)
        
    def test_epoch_end(self, test_step_outputs):        
        overall_loss = 0
        for i, dl in enumerate(self.all_dataloaders):
            dl_metrics = dl.compute_metrics("test")
            self.log_dict(dl_metrics, on_step=False, on_epoch=True, logger=True) 
            overall_loss += dl_metrics["test_{}_avg_epoch_loss".format(dl.name)]

        self.log("overall_test_loss", overall_loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.with_motifs:
            X, motifs, y = batch
        elif self.all_dataloader_modules[dataloader_idx].with_mask:
            X, y, mask = batch
        else:
            X, y = batch
        
        # get predictions
        if len(self.all_dataloaders[dataloader_idx].promoter_windows_relative_to_TSS) > 1: # average predictions over all windows
            pred = None
            l_funcs = None
            l_weights = None
            for i in range(X.shape[1]):
                if self.with_motifs:
                    y_hat, l_funcs, l_weights = self([X[:, i], motifs[:, i]])
                else:
                    y_hat, l_funcs, l_weights = self(X[:, i])
                if pred is None:
                    pred = y_hat[dataloader_idx]
                else:
                    pred += y_hat[dataloader_idx]
            pred /= X.shape[1]
        else: # only one window/sequence
            l_funcs = None
            l_weights = None
            if self.with_motifs:
                y_hat, l_funcs, l_weights = self([X.squeeze(1), motifs.squeeze(1)])
            else:
                y_hat, l_funcs, l_weights = self(X.squeeze(1))            
            pred = y_hat[dataloader_idx]
        
        return {"y": y, "pred": pred}
    
    def predict_epoch_end(self, predict_step_outputs):        
        predict_y = []
        predict_preds = []
        
        for i in range(len(predict_step_outputs)):
            predict_y.append([])
            predict_preds.append([])
        
        if len(predict_step_outputs) == 1:
            for d in predict_step_outputs:
                y = d["y"]
                pred = d["pred"]
                predict_y[0].append(y)
                predict_preds[0].append(pred)        
        else:
            for i in range(len(predict_step_outputs)):
                for d in predict_step_outputs[i]:
                    y = d["y"]
                    pred = d["pred"]
                    predict_y[i].append(y)
                    predict_preds[i].append(pred)
            
        for i in range(len(predict_step_outputs)):
            predict_y[i] = np.vstack(predict_y[i])
            predict_preds[i] = np.vstack(predict_preds[i])
        
        return predict_y, predict_preds
    
    def configure_optimizers(self):
        print("Using AdamW lr = {} weight_decay = {}".format(self.lr, self.weight_decay))
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer