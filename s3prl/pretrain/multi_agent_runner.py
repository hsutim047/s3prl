# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_pretrain.py ]
#   Synopsis     [ scripts for running the pre-training of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import glob
import random
import csv
import importlib
from tqdm import tqdm
import yaml
from collections import defaultdict
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
#-------------#
from optimizers import get_optimizer, get_grouped_parameters
from schedulers import get_scheduler

from distiller.dataset import OnlineWaveDataset

##########
# RUNNER #
##########
class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, tensorboard logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.logger = SummaryWriter(args.expdir)                                                 

        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        # self.upstream = self._get_upstream()

        # self.upstreams = [self._get_upstream() for _ in range(config['runner']['multi_agent']['num_agents'])]

        self.upstream = self._get_upstream()

        self.upstream_config = yaml.load(
            open(self.args.upstream_config, "r"), Loader=yaml.FullLoader
        )


    def _get_upstream(self):
        init_upstream = self.init_ckpt.get('Upstream_Config')
        if init_upstream:
            self.args.upstream_config = init_upstream
        module_path = f'pretrain.{self.args.upstream}.pretrain_expert'
        Upstream = getattr(importlib.import_module(module_path), 'UpstreamPretrainExpert')
        upstream = Upstream(self.config['pretrain_expert']['datarc'], 
                            self.args.upstream_config,
                            self.args.device,
                            self.args.multi_gpu).to(self.args.device)

        assert hasattr(upstream, 'device')
        assert hasattr(upstream, 'forward')
        assert hasattr(upstream, 'load_model')
        assert hasattr(upstream, 'add_state_to_save')
        assert hasattr(upstream, 'on_before_zero_grad')
        assert hasattr(upstream, 'get_train_dataloader')

        if self.init_ckpt != {}:
            print('[Runner] - Loading upstream weights from the previous experiment')
            upstream.load_model(self.init_ckpt)
        if hasattr(upstream, 'loss_to_device'):
            print('[Runner] - Loss to device')
            upstream.loss_to_device()
        return upstream

    def _get_optimizer(self, model_params, model_name=None):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )

        if self.init_ckpt != {}:
            init_optimizer = self.init_ckpt.get(f'Optimizer-{model_name}')
            assert init_optimizer
            print(f'[Runner][Optimizer-{model_name}] - Loading optimizer weights from the previous experiment')
            optimizer.load_state_dict(init_optimizer)
        return optimizer

    def _get_scheduler(self, optimizer, model_name=None):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )

        if self.init_ckpt != {}:
            init_scheduler = self.init_ckpt.get(f'Scheduler-{model_name}')
            assert init_scheduler
            print(f'[Runner][Scheduler-{model_name}] - Loading scheduler weights from the previous experiment')
            scheduler.load_state_dict(init_scheduler)
        return scheduler

    def get_all_wavs(self):
        # read all wavs path from metadata
        all_wavs_list = []
        config = self.config['pretrain_expert']['datarc']
        for dataset in config['sets']:
            metadata_path = os.path.join(config['metadata_prefix'], dataset + '.csv')
            with open(metadata_path, 'r') as f:
                rows = csv.reader(f, delimiter=',')
                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    all_wavs_list.append(row[1])
        return all_wavs_list
    
    def get_wavs_length(self):
        # create [path - length] mapping
        wavs_name_to_length = {}
        config = self.config['pretrain_expert']['datarc']
        for dataset in config['sets']:
            metadata_path = os.path.join(config['metadata_prefix'], dataset + '.csv')
            with open(metadata_path, 'r') as f:
                rows = csv.reader(f, delimiter=',')
                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    wavs_name_to_length[row[1]] = [row[2]]
        return wavs_name_to_length

    @staticmethod
    def sample_hours(all_wavs_list, wavs_name_to_length, round_hours):
        # sample hours waveform from all wavs list
        total_wavs_length = 0
        hard_wavs_list = []
        index_array = [i for i in range(len(all_wavs_list))]
        # sample rate == 16000
        round_length = round_hours * 3600 * 16000 
        while total_wavs_length < round_length:
            idx = random.sample(index_array, 1)
            # append element to hard wavs list and add total wavs length
            hard_wavs_list.append(all_wavs_list[idx[0]])
            total_wavs_length += wavs_name_to_length[all_wavs_list[idx[0]]]
            # swap element and pop
            all_wavs_list[idx[0]], all_wavs_list[-1] = all_wavs_list[-1], all_wavs_list[idx[0]]
            all_wavs_list.pop()
            index_array.pop()
        return all_wavs_list, hard_wavs_list

    def select_hard_wavs(self, all_wavs_list, sample_queries, wavs_name_to_length, criteria):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.config['pretrain_expert']['datarc']["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            wavs_list=all_wavs_list,
            wavs_name_to_len=wavs_name_to_length,
            **self.config['pretrain_expert']['datarc'],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=False, # eval
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
        )
        for model in self.upstream.models:
            model.eval()

        agent_preds = []
        scores = []

        for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
            preds = []
            for model in self.upstream.models:
                _, pred, _ = model(data, return_pred=True) # B x N x T x D
                preds.append(pred)
            # agent_preds.append(torch.cat(preds, dim=0))
            scores += eval(f'calculate_{criteria}_score')(preds)

        # scores = eval(f'calculate_{criteria}_score')(agent_preds)

        hard_index_list = torch.topk(scores, sample_queries).indices.tolist().sort(reverse=True)

        hard_wavs_list = [all_wavs_list.pop(idx) for idx in hard_index_list]

        # set model train mode
        for model in self.upstream.models:
            model.train()
            
        return all_wavs_list, hard_wavs_list
            
    def train(self):
        # set model train mode
        for upstream in self.upstreams:
            upstream.train()
            
        # prepare data
        gradient_accumulate_steps = self.config['runner']['gradient_accumulate_steps']
        train_batch_size = self.config['pretrain_expert']['datarc']['train_batch_size']
        print('[Runner] - Accumulated batch size:', train_batch_size * gradient_accumulate_steps)
        # dataloader = self.upstream.get_train_dataloader()

        # prepare multi agent
        num_agents = len(self.upstream.models)
        agents_name = self.upstream_config['multi_agent']['agents_name']
        rounds = self.config['runner']['multi_agent']['rounds']
        round_hours = self.config['runner']['multi_agent']['round_hours']
        round_sample_queries = self.config['runner']['multi_agent']['round_sample_queries']

        all_wavs_list = self.get_all_wavs() # TODO*
        wavs_name_to_length = self.get_wavs_length() # TODO*
        all_wavs_list, hard_wavs_list = self.sample_hours(all_wavs_list, wavs_name_to_length, round_hours) # TODO*
        # dataloaders = [
        #     upstream.get_train_dataloader(hard_wavs_list, round_sample_queries, wavs_name_to_length)
        #     for upstream in self.upstreams
        # ] # TODO*

        dataloaders = self.upstream.get_train_dataloader(hard_wavs_list, round_sample_queries, wavs_name_to_length)

        # TODO set epoch
        n_epochs = self.config['runner']['n_epochs']
        if n_epochs > 0: 
            total_steps = int(n_epochs * len(dataloaders[0].dataset) / gradient_accumulate_steps)
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.config['runner']['total_steps']
            n_epochs = int(total_steps * gradient_accumulate_steps / len(dataloaders[0].dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')

        assert total_steps > self.config['runner']['log_step']
        assert total_steps > self.config['runner']['save_step']

        # set amp
        amp = self.config['runner'].get('fp16', False)
        if amp:
            print('[Runner] - Enabled fp16 training')
            scalers = [torch.cuda.amp.GradScaler()] * num_agents

        # set optimizer
        model_params_agents = [upstream.model for upstream in self.upstream.models]
        optimizers = [
            self._get_optimizer(model_params_agent, agent_name)
            for model_params_agent, agent_name in zip(model_params_agents, agents_name)
        ]

        # set scheduler
        schedulers = None
        if self.config.get('scheduler'):
            # scheduler = self._get_scheduler(optimizer)
            schedulers = [
                self._get_scheduler(optimizer, agent_name)
                for optimizer, agent_name in zip(optimizers, agents_name)
            ]

        round_pbar = tqdm(total=rounds, dynamic_ncols=True, desc='overall')
        init_round = self.init_ckpt.get('Round') # TODO* save ckpt
        if init_round:
            round_pbar.n = init_round
        
        while round_pbar.n < round_pbar.total:
            '''
            1. select data using DistilHuBERTs
            2. train those DistilHuBERTs by sampling data with round_sample_queries
            '''

            all_wavs_list, round_hard_wavs_list = \
                self.select_hard_wavs(all_wavs_list, round_sample_queries) # TODO*

            # dataloaders = [
            #     upstream.get_train_dataloader(round_hard_wavs_list, round_sample_queries, wavs_name_to_length)
            #     for upstream in self.upstreams
            # ] # combine past dataset and new hard wavs

            dataloaders = self.upstream.get_train_dataloader(round_hard_wavs_list, round_sample_queries, wavs_name_to_length)

            # set progress bar
            pbar = tqdm(total=total_steps, dynamic_ncols=True, desc='overall')
            
            init_step = self.init_ckpt.get('Step')
            if init_step:
                pbar.n = init_step

            all_losses = [0] * num_agents
            backward_steps = 0
            records = [defaultdict(list)] * num_agents

            while pbar.n < pbar.total:
                for data in tqdm(zip(dataloaders), dynamic_ncols=True, desc='train'):
                    # try/except block for forward/backward
                    try:
                        if pbar.n >= pbar.total:
                            break
                        global_step = pbar.n + 1
                        
                        with torch.cuda.amp.autocast(enabled=amp):
                            losses, records = self.upstream(
                                data,
                                records=records,
                                global_step=global_step,
                                log_step=self.config['runner']['log_step'],
                            )
                        
                        if gradient_accumulate_steps > 1:
                            losses = [loss / gradient_accumulate_steps for loss in losses]
                        # TODO multi gpu
                        if self.args.multi_gpu:
                            # loss = loss.sum()
                            losses = [loss.sum() for loss in losses]
                        if amp:
                            for scaler, loss in zip(scalers, losses):
                                scaler.scale(loss).backward()
                        else:
                            for loss in losses:
                                loss.backward()
                    except RuntimeError as e:
                        if 'CUDA out of memory' in e:
                            print(f'[Runner] - CUDA out of memory at step {global_step}')
                            torch.cuda.empty_cache()
                            for optimizer in optimizers:
                                optimizer.zero_grad()
                            continue
                        else:
                            raise
                    
                    # record losses
                    all_losses = [all_loss + loss.item() for all_loss, loss in zip(all_losses, losses)]
                    del losses

                    # whether to accumulate gradient
                    backward_steps += 1
                    if backward_steps % gradient_accumulate_steps != 0:
                        continue

                    # unscale
                    if amp:
                        for scaler, optimizer in zip(scalers, optimizers):
                            scaler.unscale_(optimizer)
                    
                    # gradient clipping
                    grad_norms = [
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['runner']['gradient_clipping'])
                        for model in self.upstream.models
                    ]
                    for i, grad_norm in enumerate(grad_norms):
                        if math.isnan(grad_norm):
                            print(f'[Runner][Model {i}] - Error : grad norm is NaN at global step {global_step}')
                    
                    # optimize
                    if amp:
                        for scaler, optimizer in zip(scalers, optimizers):
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        for grad_norm, optimizer in zip(grad_norms, scalers, optimizers):
                            if not math.isnan(grad_norm):
                                optimizer.step()
                    
                    # adjust learning rate
                    for scheduler in schedulers:
                        if scheduler:
                            scheduler.step()
                    
                    # logging
                    if global_step % self.config['runner']['log_step'] == 0 or pbar.n == pbar.total - 1:
                        # log loss
                        # self.logger.add_scalar(f'')
                        for i, all_loss in enumerate(all_losses):
                            self.logger.add_scalar(f'{agents_name[i]}/train-loss', all_loss, global_step=global_step)
                        # log lr
                        for i, optimizer in enumerate(optimizers):
                            if hasattr(optimizer, 'get_lr'):
                                self.logger.add_scalar(f'{agents_name[i]}/train-lr', optimizer.get_lr()[0], global_step=global_step)
                            else:
                                self.logger.add_scalar(f'{agents_name[i]}/train-lr', self.config['optimizer']['lr'], global_step=global_step)
                        # log norm
                        for i, grad_norm in enumerate(grad_norms):
                            self.logger.add_scalar(f'{agents_name[i]}/train-gradient-norm', grad_norm, global_step=global_step)
                        
                        # log customized contents
                        self.upstream.log_records(
                            records=records,
                            logger=self.logger,
                            prefix='train', # train/test
                            global_step=global_step,
                        )
                        records = [defaultdict(list)] * num_agents
                    
                    if global_step % self.config['runner']['save_step'] == 0 or pbar.n == pbar.total - 1:
                        def check_ckpt_num(directory):
                            max_keep = self.config['runner']['max_keep']
                            ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                            if len(ckpt_pths) >= max_keep:
                                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                                for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                    os.remove(ckpt_pth)
                        check_ckpt_num(self.args.expdir)
                        
                        all_states = {}
                        for agent_name, optimizer in zip(agents_name, optimizers):
                            all_states[f'Optimizer-{agent_name}'] = optimizer.state_dict()
                        
                        all_states.update(
                            {
                                'Round': round_pbar.n,
                                'Step': pbar.n,
                                'Args': self.args,
                                'Config': self.config,
                            }
                        )
                        all_states = self.upstream.add_state_to_save(all_states)

                        for agent_name, scheduler in zip(agents_name, schedulers):
                            all_states[f'Scheduler-{agent_name}'] = scheduler.state_dict()
                        
                        name = f'states-epoch-{n_epochs}.ckpt' if pbar.n == pbar.total - 1 and n_epochs > 0 \
                               else f'states-{global_step}.ckpt'
                        save_path = os.path.join(self.args.expdir, name)
                        tqdm.write(f'[Runner] - Save the checkpoint to: {save_path}')
                        torch.save(all_states, save_path)
                    
                    all_losses = [0] * num_agents
                    pbar.update(1)
            
            pbar.close()
