import torch.optim as optim
import torch.nn as nn
import copy
from ..scheduler import MultiStepLRScheduler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Adds `src` to path
from seg_models.PointCloudSegmentation.openpoints.utils import registry
MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT): 
    Returns:
        Model: a constructed model specified by NAME.
    """
    return MODELS.build(cfg, **kwargs)


def get_num_layer_for_vit(var_name, num_max_layer):
    # remove module, and encoder.
    var_name = copy.deepcopy(var_name)
    var_name = var_name.replace('module.', '')
    var_name = var_name.replace('encoder.', '')

    if any(key in var_name for key in {"cls_token", "mask_token", "cls_pos", "pos_embed", "patch_embed"}):
        return 0
    elif "rel_pos_bias" in var_name:
        return num_max_layer - 1

    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, 
                         filter_by_modules_names=None, 
                         ):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or any(key in name for key in skip_list):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None
        
        if get_layer_scale is not None:
            scale = get_layer_scale(layer_id)
        else:
            scale = 1.0
        
        if filter_by_modules_names is not None:
            filter_exist = False 
            for module_name in filter_by_modules_names.keys():
                filter_exist = module_name in name
                if filter_exist:
                    break 
            if filter_exist:
                group_name = module_name + '_' + group_name
                this_weight_decay = filter_by_modules_names[module_name].get('weight_decay', this_weight_decay)
                scale = filter_by_modules_names[module_name].get('lr_scale', 1.0) * scale

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())



def build_optimizer_from_cfg(cfg, model,filter_bias_and_bn: bool = True,
        filter_by_modules_names=None,  **kwargs):
    layer_decay = cfg.layer_decay
    if 0. < layer_decay < 1.0:
        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        get_num_layer = assigner.get_layer_id
        get_layer_scale = assigner.get_scale
    else:
        get_num_layer, get_layer_scale = None, None


    assert isinstance(model, nn.Module)
    weight_decay = float(cfg.weight_decay)
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'module'):
            if hasattr(model.module, 'no_weight_decay'):
                skip = model.module.no_weight_decay()
        else:
            if hasattr(model, 'no_weight_decay'):
                skip = model.module.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, 
                                          filter_by_modules_names
                                          )
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_lower = cfg.optimizer.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    lr = cfg.learning_rate
    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    if opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError(f"Optimizer {cfg.optimizer} not supported")

    return optimizer

def build_scheduler_from_cfg(cfg, optimizer):
    num_epochs = cfg.epochs
    warmup_epochs = cfg.warmup_epochs
    warmup_lr = getattr(cfg, 'warmup_lr', 1.0e-6)
    decay_rate = cfg.decay_rate
    decay_epochs = cfg.decay_epochs

    noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(cfg, 'lr_noise_pct', 0.67),
        noise_std=getattr(cfg, 'lr_noise_std', 1.),
        noise_seed=getattr(cfg, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(cfg, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(cfg, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(cfg, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if cfg.sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **noise_args,
        )

    return lr_scheduler