import logging
import pickle
import random
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

from src.dataset import EarlyFixingDataset, WellObjDataset
from src.model import decode_fixing
from src.net import ObjSurrogate, InstanceGCN, Fixer
from src.utils import timeit


class Trainer(ABC):
    """Generic trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, epochs=5, lr= 0.1,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, mixed_precision=True,
                 device=None, wandb_project=None, wandb_group=None,
                 logger=None, checkpoint_every=50, random_seed=42,
                 max_loss=None) -> None:
        self._is_initalized = False

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self._dtype = next(self.net.parameters()).dtype

        self.mixed_precision = mixed_precision

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.checkpoint_every = checkpoint_every

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

        self._log_to_wandb = False if wandb_project is None else True
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group

        self.max_loss = max_loss

    @classmethod
    def load_trainer(cls, net: nn.Module, run_id: str, wandb_project=None,
                     logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and creates the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project=wandb_project,
            entity="brunompac",
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            device=wandb.config['device'],
            logger=logger,
            wandb_project=wandb_project,
            random_seed=wandb.config['random_seed'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        # load optimizer
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])

        # load scheduler
        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self._loss_func = eval(f"nn.{self.loss_func}(pos_weight=0.33)")
        # self._loss_func = eval(f"nn.{self.loss_func}()")

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.prepare_data()

        self._is_initalized = True

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )

        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        if self._log_to_wandb:
            self._add_to_wandb_config({
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "mixed_precision": self.mixed_precision,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            })

            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self._loss_func = eval(f"nn.{self.loss_func}()")

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def _add_to_wandb_config(self, d: dict):
        if not hasattr(self, '_wandb_config'):
            self._wandb_config = dict()

        for k, v in d.items():
            self._wandb_config[k] = v

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net, log=None)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    @abstractmethod
    def prepare_data(self):
        """Must populate `self.data` and `self.val_data`.
        """
        # TODO: maybe I should refactor this so that the Dataset is an input to
        # the Trainer?

    @staticmethod
    def _add_data_to_log(data: dict, prefix: str, data_to_log=dict()):
        for k, v in data.items():
            if k != 'all':
                data_to_log[prefix+k] = v
        
        return data_to_log

    def _run_epoch(self):
        # train
        train_time, (train_losses, train_times) = timeit(self.train_pass)()

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_losses['all']}")

        # validation
        val_time, (val_losses, val_times) = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"Validation loss = {val_losses['all']}")

        data_to_log = {
            "train_loss": train_losses['all'],
            "val_loss": val_losses['all'],
            "train_time": train_time,
            "val_time": val_time,
        }
        self._add_data_to_log(train_losses, 'train_loss_', data_to_log)
        self._add_data_to_log(val_losses, 'val_loss_', data_to_log)
        self._add_data_to_log(train_times, 'train_time_', data_to_log)
        self._add_data_to_log(val_times, 'val_time_', data_to_log)

        val_score = val_losses['all']  # defines best model

        return data_to_log, val_score

    def run(self) -> nn.Module:
        if not self._is_initalized:
            self.setup_training()

        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            if val_score < self.best_val:
                if self._log_to_wandb:
                    self.l.info(f"Saving best model")
                    self.save_model(name='model_best')

                self.best_val = val_score

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            if self.max_loss is not None:
                if val_score > self.max_loss:
                    break

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('Training finished!')

        return self.net

    def train_pass(self):
        train_loss = 0
        train_size = 0

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in self.data:
                X = X.to(self.device)
                y = y.to(self.device)

                self._optim.zero_grad()

                with self.autocast_if_mp():
                    forward_time_, y_hat = timeit(self.net)(X)
                    forward_time += forward_time_

                    loss_time_, loss = self.get_loss_and_metrics(y_hat, y)
                    loss_time += loss_time_

                if self.mixed_precision:
                    backward_time_, _  = timeit(self._scaler.scale(loss).backward)()
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    backward_time_, _  = timeit(loss.backward)()
                    self._optim.step()
                backward_time += backward_time_

                train_loss += loss.item() * len(y)
                train_size += len(y)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        losses = self.aggregate_loss_and_metrics(train_loss, train_size)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times
    
    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat, y)

        if validation:
            # here you can compute performance metrics
            return loss_time, loss, None
        else:
            return loss_time, loss
    
    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        return losses

    def validation_pass(self):
        val_loss = 0
        val_size = 0
        val_metrics = list()

        forward_time = 0
        loss_time = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self.val_data:
                X = X.to(self.device)
                y = y.to(self.device)

                with self.autocast_if_mp():
                    forward_time_, y_hat = timeit(self.net)(X)
                    forward_time += forward_time_

                    loss_time_, loss, metrics = self.get_loss_and_metrics(y_hat, y, validation=True)
                    loss_time += loss_time_

                    val_metrics.append(metrics)

                val_loss += loss.item() * len(y)  # scales to data size
                val_size += len(y)

        losses = self.aggregate_loss_and_metrics(val_loss, val_size, val_metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
        }

        return losses, times

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath

class ObjectiveSurrogateTrainer(Trainer):
    def __init__(self, net: ObjSurrogate, ef_objs_fpath, epochs=5, lr=0.1, batch_size=2**4, optimizer: str = 'Adam', optimizer_params: dict = None, loss_func: str = 'MSELoss', lr_scheduler: str = None, lr_scheduler_params: dict = None, mixed_precision=True, device=None, wandb_project=None, wandb_group=None, logger=None, checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        self.ef_objs_fpath = Path(ef_objs_fpath)
        self.batch_size = int(batch_size)

        self._add_to_wandb_config({
            'batch_size': self.batch_size,
        })

        super().__init__(net, epochs, lr, optimizer, optimizer_params, loss_func, lr_scheduler, lr_scheduler_params, mixed_precision, device, wandb_project, wandb_group, logger, checkpoint_every, random_seed, max_loss)

    def prepare_data(self):
        with open(self.ef_objs_fpath, 'rb') as f:
            ef_objs = pickle.load(f)

        # ds = WellObjDataset(ef_objs)

        # indices = np.arange(len(ds))
        # np.random.shuffle(indices)

        # train_i = int(0.8 * len(ds))
        # train_sampler = SubsetRandomSampler(indices[:train_i])
        # test_sampler = SubsetRandomSampler(indices[train_i:])

        # self.data = DataLoader(ds, 2**6, sampler=train_sampler)
        # self.val_data = DataLoader(ds, 2**6, sampler=test_sampler)

        wells = list(ef_objs.keys())

        np.random.seed(42)
        test_is = np.random.choice(len(wells), int(0.2 * len(wells)), replace=False)
        test_wells = [wells[i] for i in test_is]
        ef_objs_test = {w: ef_objs[w] for w in test_wells}

        ef_objs_train = ef_objs.copy()
        for w in test_wells:
            ef_objs_train.pop(w, None)

        ds_train = WellObjDataset(ef_objs_train)
        ds_test = WellObjDataset(ef_objs_test)

        self.data = DataLoader(ds_train, 2**10, shuffle=True)
        self.val_data = DataLoader(ds_test, 2**6)

    def train_pass(self):
        train_loss = 0
        train_size = 0

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train()
        with torch.set_grad_enabled(True):
            # for (q_liq_fun, bsw, gor, z_c, z_gl, q_gl_max), y in self.data:
            for (bsw, gor, z_c, z_gl, q_gl_max), y in self.data:
                # q_liq_fun = q_liq_fun.to(self.device)
                bsw = bsw.to(self.device)
                gor = gor.to(self.device)
                z_c = z_c.to(self.device)
                z_gl = z_gl.to(self.device)
                q_gl_max = q_gl_max.to(self.device)

                y = y.to(self.device).double()

                self._optim.zero_grad()

                with self.autocast_if_mp():
                    # forward_time_, y_hat = timeit(self.net)(q_liq_fun, bsw, gor, z_c, z_gl, q_gl_max)
                    forward_time_, y_hat = timeit(self.net)(bsw, gor, z_c, z_gl, q_gl_max)
                    forward_time += forward_time_

                    loss_time_, loss = self.get_loss_and_metrics(y_hat, y.unsqueeze(-1))
                    loss_time += loss_time_

                if self.mixed_precision:
                    backward_time_, _  = timeit(self._scaler.scale(loss).backward)()
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    backward_time_, _  = timeit(loss.backward)()
                    self._optim.step()
                backward_time += backward_time_

                train_loss += loss.item() * len(y)
                train_size += len(y)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        losses = self.aggregate_loss_and_metrics(train_loss, train_size)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times

    def validation_pass(self):
        val_loss = 0
        val_size = 0
        val_metrics = list()

        forward_time = 0
        loss_time = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            # for (q_liq_fun, bsw, gor, z_c, z_gl, q_gl_max), y in self.val_data:
            for (bsw, gor, z_c, z_gl, q_gl_max), y in self.val_data:
                # q_liq_fun = q_liq_fun.to(self.device)
                bsw = bsw.to(self.device)
                gor = gor.to(self.device)
                z_c = z_c.to(self.device)
                z_gl = z_gl.to(self.device)
                q_gl_max = q_gl_max.to(self.device)

                y = y.to(self.device).double()

                with self.autocast_if_mp():
                    # forward_time_, y_hat = timeit(self.net)(q_liq_fun, bsw, gor, z_c, z_gl, q_gl_max)
                    forward_time_, y_hat = timeit(self.net)(bsw, gor, z_c, z_gl, q_gl_max)
                    forward_time += forward_time_

                    loss_time_, loss, metrics = self.get_loss_and_metrics(y_hat, y.unsqueeze(-1), validation=True)
                    loss_time += loss_time_

                    val_metrics.append(metrics)

                val_loss += loss.item() * len(y)  # scales to data size
                val_size += len(y)

        losses = self.aggregate_loss_and_metrics(val_loss, val_size, val_metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
        }

        return losses, times

class EarlyFixingTrainer(Trainer):
    def __init__(self, net: Fixer, surrogate: ObjSurrogate, ef_objs_fpath, epochs=5, lr=0.1, batch_size=2**4, optimizer: str = 'Adam', optimizer_params: dict = None, loss_func: str = 'MSELoss', lr_scheduler: str = None, lr_scheduler_params: dict = None, mixed_precision=True, device=None, wandb_project=None, wandb_group=None, logger=None, checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        self.ef_objs_fpath = Path(ef_objs_fpath)
        self.batch_size = int(batch_size)

        self.surrogate = surrogate
        self.surrogate.eval()

        self._add_to_wandb_config({
            'batch_size': self.batch_size,
        })

        super().__init__(net, epochs, lr, optimizer, optimizer_params, loss_func, lr_scheduler, lr_scheduler_params, mixed_precision, device, wandb_project, wandb_group, logger, checkpoint_every, random_seed, max_loss)

        # self.net.to(self.device)
        self.surrogate.to(self.device)

    def prepare_data(self):
        with open(self.ef_objs_fpath, 'rb') as f:
            ef_objs = pickle.load(f)

        self.ef_objs = ef_objs

        # ds = WellObjDataset(ef_objs)

        # indices = np.arange(len(ds))
        # np.random.shuffle(indices)

        # train_i = int(0.8 * len(ds))
        # train_sampler = SubsetRandomSampler(indices[:train_i])
        # test_sampler = SubsetRandomSampler(indices[train_i:])

        # self.data = DataLoader(ds, 2**6, sampler=train_sampler)
        # self.val_data = DataLoader(ds, 2**6, sampler=test_sampler)

        wells = list(ef_objs.keys())

        np.random.seed(42)
        test_is = np.random.choice(len(wells), int(0.2 * len(wells)), replace=False)
        test_wells = [wells[i] for i in test_is]
        ef_objs_test = {w: ef_objs[w] for w in test_wells}

        ef_objs_train = ef_objs.copy()
        for w in test_wells:
            ef_objs_train.pop(w, None)

        ds_train = EarlyFixingDataset(ef_objs_train)
        ds_test = EarlyFixingDataset(ef_objs_test)

        self.test_wells = ds_test.wells

        self.data = DataLoader(ds_train, 2**6, shuffle=True)
        self.val_data = DataLoader(ds_test, 2**6)

    def train_pass(self):
        train_loss = 0
        train_size = 0

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train()
        with torch.set_grad_enabled(True):
            # for (q_liq_fun, bsw, gor, q_gl_max), (z_c, z_gl, obj, well_i) in self.data:
            for (bsw, gor, q_gl_max), (z_c, z_gl, obj, well_i) in self.data:
                # q_liq_fun = q_liq_fun.to(self.device)
                bsw = bsw.to(self.device)
                gor = gor.to(self.device)
                q_gl_max = q_gl_max.to(self.device)

                z_c = z_c.to(self.device)
                z_gl = z_gl.to(self.device)
                obj = obj.to(self.device)
                well_i = well_i.to(self.device)

                with self.autocast_if_mp():
                    # forward_time_, y_hat = timeit(self.net)(q_liq_fun, bsw, gor, q_gl_max)
                    forward_time_, y_hat = timeit(self.net)(bsw, gor, q_gl_max)
                    forward_time += forward_time_

                    # loss_time_, loss = self.get_loss_and_metrics(y_hat, (q_liq_fun, bsw, gor, q_gl_max, z_c, z_gl, obj, well_i))
                    loss_time_, loss = self.get_loss_and_metrics(y_hat, (bsw, gor, q_gl_max, z_c, z_gl, obj, well_i))
                    loss_time += loss_time_

                if self.mixed_precision:
                    backward_time_, _  = timeit(self._scaler.scale(loss).backward)()
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    backward_time_, _  = timeit(loss.backward)()
                    self._optim.step()
                backward_time += backward_time_

                train_loss += loss.item() * len(obj)
                train_size += len(obj)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        losses = self.aggregate_loss_and_metrics(train_loss, train_size)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times

    def validation_pass(self):
        val_loss = 0
        val_size = 0
        val_metrics = list()

        forward_time = 0
        loss_time = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            # for (q_liq_fun, bsw, gor, q_gl_max), (z_c, z_gl, obj, well_i) in self.val_data:
            for (bsw, gor, q_gl_max), (z_c, z_gl, obj, well_i) in self.val_data:
                # q_liq_fun = q_liq_fun.to(self.device)
                bsw = bsw.to(self.device)
                gor = gor.to(self.device)
                q_gl_max = q_gl_max.to(self.device)

                z_c = z_c.to(self.device)
                z_gl = z_gl.to(self.device)
                obj = obj.to(self.device)
                well_i = well_i.to(self.device)

                with self.autocast_if_mp():
                    # forward_time_, y_hat = timeit(self.net)(q_liq_fun, bsw, gor, q_gl_max)
                    forward_time_, y_hat = timeit(self.net)(bsw, gor, q_gl_max)
                    forward_time += forward_time_

                    # loss_time_, loss, metrics = self.get_loss_and_metrics(y_hat, (q_liq_fun, bsw, gor, q_gl_max, z_c, z_gl, obj, well_i), validation=True)
                    loss_time_, loss, metrics = self.get_loss_and_metrics(y_hat, (bsw, gor, q_gl_max, z_c, z_gl, obj, well_i), validation=True)
                    loss_time += loss_time_

                    val_metrics.append(metrics)

                val_loss += loss.item() * len(obj)  # scales to data size
                val_size += len(obj)

        losses = self.aggregate_loss_and_metrics(val_loss, val_size, val_metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
        }

        return losses, times

    def get_loss_and_metrics(self, y_hat, problem_data, validation=False):
        # q_liq_fun, bsw, gor, q_gl_max, z_c, z_gl, obj, well_ix = problem_data
        bsw, gor, q_gl_max, z_c, z_gl, obj, well_ix = problem_data

        start = time()
        y_pred = torch.softmax(y_hat, -1)
        z_c_hat = y_pred[:,0,:]
        z_gl_hat = y_pred[:,1,:]

        # surr_obj = self.surrogate(q_liq_fun, bsw, gor, z_c_hat, z_gl_hat, q_gl_max)
        surr_obj = self.surrogate(bsw, gor, z_c_hat, z_gl_hat, q_gl_max)
        loss = - surr_obj
        loss = loss.mean()

        loss_time = time() - start

        if validation:
            with torch.no_grad():
                z_c_argmax = z_c_hat.argmax(-1, keepdim=True)
                z_c_hat_int = torch.zeros_like(z_c_hat).scatter_(-1, z_c_argmax, 1)

                z_gl_argmax = z_gl_hat.argmax(-1, keepdim=True)
                z_gl_hat_int = torch.zeros_like(z_gl_hat).scatter_(-1, z_gl_argmax, 1)

                pairs = list()
                for well_i, zc, zgl in zip(well_ix, z_c_hat_int, z_gl_hat_int):
                    pairs.append(decode_fixing(zc.cpu().numpy(), zgl.cpu().numpy(), self.test_wells[well_i]))

                real_obj = list()
                for (c_pair, gl_pair), qglm, well_i in zip(pairs, q_gl_max, well_ix):
                    well = self.test_wells[well_i]
                    real_obj.append(self.ef_objs[well][(*c_pair, *gl_pair, qglm.item())])
                real_obj = torch.Tensor(real_obj).to(obj)

                surr_obj_gap = obj - surr_obj.squeeze(1)
                rel_surr_obj_gap = surr_obj_gap / obj

                real_obj_gap = obj - real_obj
                rel_real_obj_gap = real_obj_gap / obj

                z_c_dist = (z_c.argmax(-1) - z_c_hat.argmax(-1)).abs()
                z_gl_dist = (z_gl.argmax(-1) - z_gl_hat.argmax(-1)).abs()

            return loss_time, loss, (real_obj_gap, rel_real_obj_gap, surr_obj_gap, rel_surr_obj_gap, z_c_dist, z_gl_dist)
        else:
            return loss_time, loss

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
        }

        if metrics is not None:
            real_obj_gap = torch.hstack([m[0] for m in metrics])
            rel_real_obj_gap = torch.hstack([m[1] for m in metrics])

            surr_obj_gap = torch.hstack([m[2] for m in metrics])
            rel_surr_obj_gap = torch.hstack([m[3] for m in metrics])

            z_c_dist = torch.hstack([m[4] for m in metrics]).double()
            z_gl_dist = torch.hstack([m[5] for m in metrics]).double()

            losses['mean_surr_obj_gap'] = surr_obj_gap.mean()
            losses['surr_obj_gap'] = surr_obj_gap.detach().cpu().numpy()
            losses['rel_surr_obj_gap'] = rel_surr_obj_gap.detach().cpu().numpy()

            losses['mean_real_obj_gap'] = real_obj_gap.mean()
            losses['real_obj_gap'] = real_obj_gap.detach().cpu().numpy()
            losses['rel_real_obj_gap'] = rel_real_obj_gap.detach().cpu().numpy()

            losses['z_c_dist'] = z_c_dist.mean()
            losses['z_gl_dist'] = z_gl_dist.mean()

        return losses

class FeasibilityTrainer(ObjectiveSurrogateTrainer):
    def __init__(self, net: ObjSurrogate, ef_objs_fpath, epochs=5, lr=0.1, batch_size=2 ** 4, optimizer: str = 'Adam', optimizer_params: dict = None, loss_func: str = 'BCEWithLogitsLoss', lr_scheduler: str = None, lr_scheduler_params: dict = None, mixed_precision=True, device=None, wandb_project=None, wandb_group=None, logger=None, checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        super().__init__(net, ef_objs_fpath, epochs, lr, batch_size, optimizer, optimizer_params, loss_func, lr_scheduler, lr_scheduler_params, mixed_precision, device, wandb_project, wandb_group, logger, checkpoint_every, random_seed, max_loss)

    def get_loss_and_metrics(self, y_hat, y, validation=False):
        y = (y > 0).to(y)

        loss_time, loss =  timeit(self._loss_func)(y_hat, y)

        if validation:
            # here you can compute performance metrics
            y_pred = (torch.sigmoid(y_hat) > 0.5).to(y)
            hits = (y_pred == y).sum()
            return loss_time, loss, hits
        else:
            return loss_time, loss

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
        }

        if metrics is not None:
            losses['accuracy'] = sum(metrics) / size

        return losses
