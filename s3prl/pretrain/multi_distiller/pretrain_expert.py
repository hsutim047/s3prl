"""
    Pre-train expert for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

from collections import defaultdict
from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pretrain.distiller.dataset import OnlineWaveDataset
from upstream.distiller.model import DistillerConfig, DistillerModel
import random


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False

        
class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu
        self.datasets = []
        self.dataloaders = []
        
        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        # get agents config
        self.agent_config = edict(self.upstream_config["multi_agent"])
        model_configs = [DistillerConfig(self.upstream_config[agent_name]) for agent_name in self.agent_config.agents_name]
        self.num_agents = self.agent_config.num_agents
        
        # init teacher
        teacher_config = edict(self.upstream_config["teacher"])
        teacher = torch.hub.load("s3prl/s3prl", teacher_config.model)
        if (
            teacher_config.model.find("hubert") >= 0
            or teacher_config.model.find("wav2vec2") >= 0
        ):
            teacher.model.encoder.layerdrop = 0
            print("[DistillerForPretrain] - Disabled teacher's encoder layerdrop")

        # check agents and teacher
        for model_config in model_configs:
            assert model_config.n_tasks <= teacher_config.n_layers, (
                model_config.n_tasks,
                teacher_config.n_layers,
            )
        print(
            "[DistillerForPretrain] - Using {} as teacher with {} layers".format(
                teacher_config.model, teacher_config.n_layers
            )
        )

        # freeze teacher
        self.teacher = teacher
        freeze_model(self.teacher)
        
        # prepare agents
        print(f"[UpstreamPretrainExpert] - Initializing {self.agent_config.num_agents} models...")
        self.models = [DistillerForPretrain(model_config, self.teacher) for model_config in model_configs]

        # set multi gpu
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self, wavs_list, wavs_name_to_length, idx):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            wavs_list=wavs_list,
            wavs_name_to_length=wavs_name_to_length,
            **self.datarc,
        )
        try:
            prev_dataset = self.datasets[idx]
            dataset = ConcatDataset([prev_dataset, dataset])
            self.datasets[idx] = dataset
        except IndexError:
            self.datasets.append(dataset)
                  
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        return dataloader

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            for model, agent_name in zip(self.models, self.agent_config.agents_name):
                model.module.distiller.load_state_dict(all_states[agent_name])
        else:
            for model, agent_name in zip(self.models, self.agent_config.agents_name):
                model.distiller.load_state_dict(all_states[agent_name])

    # Interface
    def add_state_to_save(self, all_states):
        for model, agent_name in zip(self.models, self.agent_config.agents_name):
            all_states[agent_name] = (
                model.float().distiller.state_dict()
                if not self.multi_gpu
                else model.float().module.distiller.state_dict()
            )
        all_states["Upstream_Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self, hard_wavs_list, round_sample_queries, wavs_name_to_length):
        self.dataloaders = []
        for idx in range(self.num_agents):
            wavs_list = random.sample(hard_wavs_list, round_sample_queries)
            dataloader = self._get_train_dataloader(wavs_list, wavs_name_to_length, idx)
            self.dataloaders.append(dataloader)            
        return self.dataloaders

    # Interface Not done ######################################################################
    def forward(self, data_list, records_list, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [[wave_input1, pad_mask1], [wave_input2, pad_mask2], ...]
            records:
                [defaultdict(list), ...], by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step
        Return:
            losses
        """
        losses = []
        for data, model, records in enumerate(zip(data_list, self.models, records_list)):
            wave_input, wave_orig, wave_len, pad_mask = data
            wave_input = wave_input.to(self.device)
            wave_len = wave_len.to(self.device)
            pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

            loss, other_res = model(
                wave_input,
                wave_orig,
                wave_len,
                pad_mask,
                return_other=global_step % log_step == 0,
            )
            losses.append(loss)

            if global_step % log_step == 0:
                for key, value in other_res.items():
                    if isinstance(value, torch.Tensor):
                        value = float(value.mean().cpu().item())
                    records[key] = value

        return losses, records_list

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records_list, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                [defaultdict(list), ...], contents already appended
            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents
            prefix:
                train/test
            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for agent_name, records in zip(self.agents_name, records_list):
            for key, values in records.items():
                if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                    logger.add_image(f'{agent_name}/{prefix}-{key}', values, global_step=global_step)
                elif isinstance(values, float):
                    logger.add_scalar(f'{agent_name}/{prefix}-{key}', values, global_step=global_step)


class DistillerForPretrain(nn.Module):
    """
    Distiller for pretraining
    """

    def __init__(self, config: DistillerConfig, teacher):
        super().__init__()
        self.config = config
        self.distiller = DistillerModel(config)
        self.teacher = teacher        

        if config.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(config.loss_type)

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")

        if config.init_teacher_conv_layers:
            print(
                "[DistillerForPretrain] - "
                "Initializing feature extractor from teacher"
            )
            self.distiller.feature_extractor.load_state_dict(
                self.teacher.model.feature_extractor.state_dict()
            )
            if self.distiller.post_extract_proj is not None:
                self.distiller.post_extract_proj.load_state_dict(
                    self.teacher.model.post_extract_proj.state_dict()
                )

        if config.init_teacher_encoder_layers:
            print("[DistillerForPretrain] - " "Initializing encoder from teacher")
            self.distiller.encoder.pos_conv.load_state_dict(
                self.teacher.model.encoder.pos_conv.state_dict()
            )
            for l in range(config.encoder_layers):
                self.distiller.encoder.layers[l].load_state_dict(
                    self.teacher.model.encoder.layers[l].state_dict()
                )

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
        return_pred: bool = False,
    ):
        """
        Forward function.
        Input:
            wave_input: FloatTensor (B x T_wave)
            wave_orig: List of FloatTensor
            wave_len: LongTensor (B)
            pad_mask: FloatTensor (B x T)
            return_other: Bool (returns other information for logging)
        """

        # Forward model
        feat, feat_final, pred, pad_mask = self.distiller(wave_input, pad_mask)

        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            with torch.cuda.amp.autocast(False):
                teacher_hiddens = self.teacher(wave_orig)
            if self.config.task_emb_type == "none":
                teacher_hiddens = teacher_hiddens["hidden_states"][self.config.n_tasks]
                teacher_hiddens = teacher_hiddens.unsqueeze(1)
            else:
                if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                    teacher_hiddens = [
                        teacher_hiddens["hidden_states"][i]
                        for i in self.distiller.pred_layer_id
                    ]
                else:
                    teacher_hiddens = teacher_hiddens["hidden_states"][1:]
                teacher_hiddens = torch.stack(teacher_hiddens, dim=1)  # B x N x T x D

        # Compute all objectives
        (
            total_loss,
            rec_loss,
            rec_layer_loss,
            feat_pen,
            sim_loss,
            sim_layer_loss,
        ) = self.compute_loss(feat, pred, teacher_hiddens, return_other)

        if return_other:
            with torch.no_grad():
                other_res = {
                    "rec_loss": rec_loss,
                    "feat_pen": feat_pen,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }
                teacher_norm = torch.abs(teacher_hiddens).mean((0, 2, 3))
                if self.config.task_emb_type == "none":
                    other_res[f"rec_l{self.config.n_tasks}"] = rec_layer_loss[0]
                    other_res[f"tar_norm_l{self.config.n_tasks}"] = teacher_norm[0]
                    if sim_layer_loss is not None:
                        other_res[f"sim_l{self.config.n_tasks}"] = sim_layer_loss[0]
                else:
                    for i in range(self.config.n_tasks):
                        layer_id = i + 1
                        if self.config.task_emb_type in [
                            "expand-last",
                            "hnet",
                            "self-hidden",
                        ]:
                            layer_id = self.distiller.pred_layer_id[i]
                        other_res[f"rec_l{layer_id}"] = rec_layer_loss[i]
                        other_res[f"tar_norm_l{layer_id}"] = teacher_norm[i]
                        if sim_layer_loss is not None:
                            other_res[f"sim_l{layer_id}"] = sim_layer_loss[i]
                    if self.config.task_emb_type not in [
                        "expand-last",
                        "hnet",
                        "self-hidden",
                    ]:
                        other_res[
                            "norm_task_emb"
                        ] = self.distiller.task_embedding.weight.pow(2).mean()
        else:
            other_res = None
        
        if return_pred:
            total_loss, pred, other_res

        return total_loss, other_res

    def compute_loss(self, feat, pred, target, return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
        """

        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss_func(pred, target)  # B x N x T x D

        if return_other:
            with torch.no_grad():
                rec_layer_loss = rec_loss.mean((0, 2, 3))
        else:
            rec_layer_loss = None

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1))
            # B x N x T
            if return_other:
                with torch.no_grad():
                    sim_layer_loss = sim_loss.mean((0, 2))
            else:
                sim_layer_loss = None
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # Feature loss
        feat_pen = feat.float().pow(2).mean()

        total_loss = (
            rec_loss
            + feat_pen * self.config.feat_pen_loss
            + sim_loss * self.cosine_loss
        )

        return total_loss, rec_loss, rec_layer_loss, feat_pen, sim_loss, sim_layer_loss
