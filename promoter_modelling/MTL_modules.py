import numpy as np
import pdb
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader

import torchmtl

from promoter_modelling.backbone_modules import *

np.random.seed(97)
torch.manual_seed(97)

# combines multiple dataloaders into one
class MTLDataLoader(pl.LightningDataModule):
    # change min_size to max_size_cycle to iterate through largest dataset fully during each training epoch
    # more details: https://github.com/Lightning-AI/lightning/blob/15ef52bc732d1f907de4de58683f131652c0d68c/src/pytorch_lightning/trainer/supporters.py
    def __init__(self, all_dataloaders, train_mode="min_size", return_full_dataset_for_predict=False):
        super().__init__()
        
        self.all_train_ds = CombinedLoader([i.train_dataloader() for i in all_dataloaders], train_mode)
        self.all_test_ds = [i.test_dataloader() for i in all_dataloaders]
        self.all_val_ds = [i.val_dataloader() for i in all_dataloaders]

        if return_full_dataset_for_predict:
            self.all_predict_ds = [i.full_dataloader() for i in all_dataloaders]
        else:
            self.all_predict_ds = [i.test_dataloader() for i in all_dataloaders]
        
    def train_dataloader(self):
        return self.all_train_ds

    def test_dataloader(self):
        return self.all_test_ds
    
    def val_dataloader(self):
        return self.all_val_ds

    def predict_dataloader(self):
        return self.all_predict_ds

# final prediction module that uses 3 linear layers per output and then stacks them
class MTLFinalPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.ln = nn.LayerNorm(self.input_size)
        self.all_layers = nn.ModuleList()
        for i in range(self.output_size):
            self.all_layers.append(nn.Sequential(nn.Linear(self.input_size, self.hidden_size), \
                                                 nn.GELU(), \
                                                 nn.Linear(self.hidden_size, self.hidden_size), \
                                                 nn.GELU(), \
                                                 nn.Linear(self.hidden_size, 1)))
    
    def forward(self, x):
        all_outputs = []
        x = self.ln(x)
        for i in range(self.output_size):
            all_outputs.append(self.all_layers[i](x))
        return torch.cat(all_outputs, dim=1)

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
                 max_epochs=None, \
                 n_cpus=0, \
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
            self.backbone = model_class(num_motifs=self.num_motifs)
        else:
            self.backbone = model_class()
            
        # create MTL model
        model_config = [
                            {
                                'name': "Backbone",
                                'layers': self.backbone,
                                # No anchor_layer means this layer receives input directly
                            }
                        ]
        
        if self.num_tasks == 1: # we don't want the loss scaling when there's only one loss term
            print("Single dataloader model")
            task = self.all_dataloaders[0]
            if model_class == MPRAnn: # MPRAnn has a single linear layer at the end
                model_config.append({
                                        'name': task.name,
                                        'layers': nn.Linear(self.backbone.embed_dims, task.num_outputs),
                                        'loss': task.loss_fn,
                                        'loss_weight': torch.tensor(1.0),
                                        'anchor_layer': 'Backbone'
                                    })
            elif task.name.startswith("FluorescenceData"): # only when predicting fluorescence data use MTLFinalPredictor
                model_config.append({
                                        'name': task.name,
                                        'layers': MTLFinalPredictor(self.backbone.embed_dims, task.num_outputs),
                                        'loss': task.loss_fn,
                                        'loss_weight': torch.tensor(0.0),
                                        'anchor_layer': 'Backbone'
                                    })
            else:
                model_config.append({
                                        'name': task.name,
                                        'layers': nn.Linear(self.backbone.embed_dims, task.num_outputs),
                                        'loss': task.loss_fn,
                                        'loss_weight': torch.tensor(1.0),
                                        'anchor_layer': 'Backbone'
                                    })
        else:
            for i, task in enumerate(self.all_dataloaders):
                model_config.append({
                                        'name': task.name,
                                        'layers': nn.Linear(self.backbone.embed_dims, task.num_outputs),
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
        self.max_epochs = max_epochs

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
            
            self.log("{}_train_loss".format(self.all_dataloaders[i].name), loss.cpu().detach().float(), on_step=True, on_epoch=True, logger=True)

            total_loss += loss
        
        self.log("train_loss", total_loss.cpu().detach().float(), on_step=True, on_epoch=True, logger=True)

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
        self.all_dataloaders[dataloader_idx].update_metrics(pred.cpu().detach().float(), y.cpu().detach().float(), loss.cpu().detach().float(), "val")

        self.log("{}_val_loss".format(self.all_dataloaders[dataloader_idx].name), loss.cpu().detach().float())
    
    def on_validation_epoch_end(self):
        overall_loss = 0
        for i, dl in enumerate(self.all_dataloaders):
            dl_metrics = dl.compute_metrics("val")
            self.log_dict(dl_metrics, on_step=False, on_epoch=True, logger=True) 
            overall_loss += dl_metrics["val_{}_avg_epoch_loss".format(dl.name)]

        self.log("overall_val_loss", overall_loss)

        gc.collect()
        torch.cuda.empty_cache()
    
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
        self.all_dataloaders[dataloader_idx].update_metrics(pred.cpu().detach().float(), y.cpu().detach().float(), loss.cpu().detach().float(), "test")

        self.log("{}_test_loss".format(self.all_dataloaders[dataloader_idx].name), loss.cpu().detach().float())
        
    def test_epoch_end(self, test_step_outputs):        
        overall_loss = 0
        for i, dl in enumerate(self.all_dataloaders):
            dl_metrics = dl.compute_metrics("test")
            self.log_dict(dl_metrics, on_step=False, on_epoch=True, logger=True) 
            overall_loss += dl_metrics["test_{}_avg_epoch_loss".format(dl.name)]

        self.log("overall_test_loss", overall_loss)

        gc.collect()
        torch.cuda.empty_cache()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.with_motifs:
            X, motifs, y = batch
        elif self.all_dataloader_modules[dataloader_idx].with_mask:
            if len(batch) < 3: # happens when evaluating on a dataloader other than the training dataloader
                X, y = batch
            else:
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
        
        return {"y": y.cpu().detach().float(), "pred": pred.cpu().detach().float()}
    
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

        gc.collect()
        torch.cuda.empty_cache()
        
        return predict_y, predict_preds
    
    def configure_optimizers(self):
        # if backbone is of type backbone_modules.LegNet, use AdamW + OneCycleLR
        if isinstance(self.backbone, LegNet) or isinstance(self.backbone, LegNetLarge):
            div_factor = 25
            print("Using AdamW + OneCycleLR min_lr = {} max_lr = {} weight_decay = {}".format(self.lr / div_factor, self.lr, self.weight_decay))
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr / div_factor, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=self.lr,
                                                div_factor=div_factor,
                                                steps_per_epoch=len(self.all_dataloaders[-1].train_dataloader()), 
                                                epochs=self.max_epochs, 
                                                pct_start=0.3,
                                                three_phase="store_true")
            return [optimizer], [scheduler]
        elif isinstance(self.backbone, MPRAnn):
            print("Using Adam lr = {}".format(self.lr))
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
            return optimizer

        print("Using AdamW lr = {} weight_decay = {}".format(self.lr, self.weight_decay))
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer