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
    """
    Determine the layer index for a variable in a Vision Transformer (ViT) model.

    Args:
        var_name (str): The name of the variable in the model.
        num_max_layer (int): The maximum number of layers in the ViT model.

    Returns:
        int: The corresponding layer index for the given variable.
    """

    # Create a deep copy of the variable name to avoid modifying the original.
    var_name = copy.deepcopy(var_name)

    # Remove prefixes like 'module.' and 'encoder.' that may be present in variable names.
    var_name = var_name.replace('module.', '')
    var_name = var_name.replace('encoder.', '')

    # Check if the variable is related to tokens or embeddings, which belong to the initial layer (layer 0).
    if any(key in var_name for key in {"cls_token", "mask_token", "cls_pos", "pos_embed", "patch_embed"}):
        return 0

    # If the variable is related to relative position bias, assign it to the last layer.
    elif "rel_pos_bias" in var_name:
        return num_max_layer - 1

    # If the variable belongs to a specific transformer block, extract the block number and assign it to the corresponding layer.
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1]) 
        return layer_id + 1 

    # By default, assign the variable to the last layer if no other condition matches.
    else:
        return num_max_layer - 1



class LayerDecayValueAssigner(object):
    """
    A class to assign decay values to layers in a model, 

    Attributes:
        values (list): A list of decay values for each layer.
    """

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        """
        Get the decay scale for a specific layer.

        Args:
            layer_id (int): The ID of the layer for which the scale is needed.

        Returns:
            float: The decay value for the specified layer.
        """
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        """
        Determine the layer ID for a given variable name in the model.

        Args:
            var_name (str): The name of the variable.

        Returns:
            int: The corresponding layer ID for the given variable name.
        """
        # Use the helper function to map variable names to layer IDs.
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, 
                         filter_by_modules_names=None):
    """
    Organize model parameters into groups for optimized learning rate and weight decay settings.

    Args:
        model (nn.Module): The model containing the parameters.
        weight_decay (float): Default weight decay value for the parameters.
        skip_list (tuple): List of parameter name patterns to exclude from weight decay.
        get_num_layer (callable, optional): Function to map parameter names to layer IDs.
        get_layer_scale (callable, optional): Function to scale learning rates per layer.
        filter_by_modules_names (dict, optional): Dict specifying custom weight decay and learning rate
                                                  scaling for specific modules.

    Returns:
        list: A list of parameter group configurations for the optimizer.
    """
    parameter_group_names = {}  # To store human-readable parameter group names for debugging/logging.
    parameter_group_vars = {}   # To store actual parameter group configurations.

    for name, param in model.named_parameters():
        # Skip parameters that are frozen (require_grad is False).
        if not param.requires_grad:
            continue

        # Determine if the parameter should be excluded from weight decay.
        if len(param.shape) == 1 or name.endswith(".bias") or any(key in name for key in skip_list):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # If get_num_layer is provided, categorize the parameter by its layer ID.
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f"layer_{layer_id}_{group_name}"
        else:
            layer_id = None

        # Apply layer-specific learning rate scaling if get_layer_scale is provided.
        scale = get_layer_scale(layer_id) if get_layer_scale is not None else 1.0

        # If module-specific filtering is enabled, adjust group name and settings.
        if filter_by_modules_names is not None:
            filter_exist = False
            for module_name, module_settings in filter_by_modules_names.items():
                if module_name in name:
                    filter_exist = True
                    group_name = f"{module_name}_{group_name}"
                    this_weight_decay = module_settings.get('weight_decay', this_weight_decay)
                    scale *= module_settings.get('lr_scale', 1.0)
                    break

        # Initialize a new parameter group if it doesn't exist.
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

        # Add the parameter to the appropriate group.
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


def build_optimizer_from_cfg(cfg, model, filter_bias_and_bn=True, filter_by_modules_names=None, **kwargs):
    """
    Build an optimizer for a model based on a configuration.

    Args:
        cfg (object): Configuration object containing optimizer settings.
        model (nn.Module): The model whose parameters will be optimized.
        filter_bias_and_bn (bool): Whether to exclude biases and batch norm parameters from weight decay.
        filter_by_modules_names (dict, optional): Module-specific settings for weight decay and learning rate.
        **kwargs: Additional optimizer arguments.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    layer_decay = cfg.layer_decay

    # If layer decay is enabled, set up layer-wise scaling using LayerDecayValueAssigner.
    if 0. < layer_decay < 1.0:
        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(
            [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        )
        get_num_layer = assigner.get_layer_id
        get_layer_scale = assigner.get_scale
    else:
        get_num_layer, get_layer_scale = None, None

    # Ensure the model is an instance of nn.Module.
    assert isinstance(model, nn.Module)

    weight_decay = float(cfg.weight_decay)

    # Handle special filtering for biases and batch norm parameters.
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'module') and hasattr(model.module, 'no_weight_decay'):
            skip = model.module.no_weight_decay()
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()

        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, filter_by_modules_names)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    # Parse optimizer type and set up its arguments.
    opt_lower = cfg.optimizer.lower().split('_')[-1]
    lr = cfg.learning_rate
    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    # Create the optimizer based on the specified type.
    if opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError(f"Optimizer {cfg.optimizer} not supported")

    return optimizer


def build_scheduler_from_cfg(cfg, optimizer):
    """
    Build a learning rate scheduler for an optimizer based on a configuration.

    Args:
        cfg (object): Configuration object containing scheduler settings.
        optimizer (torch.optim.Optimizer): Optimizer to attach the scheduler to.

    Returns:
        Learning rate scheduler: Configured learning rate scheduler.
    """
    num_epochs = cfg.epochs
    warmup_epochs = cfg.warmup_epochs
    warmup_lr = getattr(cfg, 'warmup_lr', 1.0e-6)
    decay_rate = cfg.decay_rate
    decay_epochs = cfg.decay_epochs

    # Configure optional noise and cyclic learning rate arguments.
    noise_args = dict(
        noise_range_t=None,
        noise_pct=getattr(cfg, 'lr_noise_pct', 0.67),
        noise_std=getattr(cfg, 'lr_noise_std', 1.0),
        noise_seed=getattr(cfg, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(cfg, 'lr_cycle_mul', 1.0),
        cycle_decay=getattr(cfg, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(cfg, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None

    # Create a MultiStep learning rate scheduler.
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
