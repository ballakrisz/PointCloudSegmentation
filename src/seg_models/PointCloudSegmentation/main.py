import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from openpoints.transforms.data_transforms import PointCloudScaling, PointCloudJitter
from openpoints.building_blocks.segmentation.pcs import PCS
from openpoints.utils.validate import Validator
from openpoints.building_blocks.build import build_model_from_cfg, build_optimizer_from_cfg, build_scheduler_from_cfg
from openpoints.utils import AverageMeter
from openpoints.utils import save_checkpoint, load_checkpoint,\
     generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adds `src` to path
from data.load_data import ShapeNetSem

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg):
    """
    Train one epoch of the model

    Args:
        model: The model to train
        train_loader: The dataloader for the training data
        criterion: The loss function to use
        optimizer: The optimizer to use
        epoch: The current epoch
        cfg: The config file
    """
    # setup loss meter and model to training mode
    loss_meter = AverageMeter()
    model.train() 
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    # Process one epoch
    for idx, data in pbar:
        num_iter += 1
        batch_size = data[0].shape[0]
        # extract dcata from data loader
        xyz, features, obj_class, labels = data
        xyz = xyz.cuda()
        features = features.cuda().permute(0,2,1).contiguous()
        labels = labels.cuda()
        obj_class = obj_class.cuda()
        # Make predictions with the model
        logits = model(xyz, features, obj_class)
        logits = logits.transpose(1,2)
        labels = labels.long()
        # Calculate the loss and backpropagate
        loss = criterion(logits, labels)
        loss.backward()

        # Step the optimizer every cfg.step_per_update iterations 
        # For my training it was every 4 iterations to accumilate the gradients to simulate bigger batch size
        if num_iter == cfg.step_per_update:
            # Clip the gradients
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            # Adjust the weights based on the gradients and zero them out
            optimizer.step()
            optimizer.zero_grad()
        
        # Update the loss meter and the progress bar
        loss_meter.update(loss.item(), n=batch_size)
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.3f} "
                                 )
    train_loss = loss_meter.avg
    return train_loss

def main(cfg):
    # Build model based on cfg file
    model = build_model_from_cfg(cfg.MODEL).cuda()

    # Transformations for the dataloader
    scaling_trasnform = PointCloudScaling()
    jitter_transform = PointCloudJitter()
    transforms = [scaling_trasnform, jitter_transform]

    # create train datatset and dataloader
    train_dataset = ShapeNetSem(
        npoints=cfg.DATA.npoint, 
        split='train', 
        preload=cfg.DATA.preload, 
        use_normals=False,
        transforms=transforms
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.SOLVER.batch_size, 
        shuffle=True
        )
    
    # create val and test datatset and dataloader
    val_dataset = ShapeNetSem(
        npoints=cfg.DATA.npoint, 
        split='val', 
        preload=cfg.DATA.preload, 
        use_normals=False
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.SOLVER.batch_size, 
        shuffle=False
        )
    
    # create test datatset and dataloader
    test_dataset = ShapeNetSem(
        npoints=cfg.DATA.npoint, 
        split='test', 
        preload=cfg.DATA.preload, 
        use_normals=False
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.test_bs, 
        shuffle=False
        )

    # using the Cross Entropy loss for training
    criterion = torch.nn.CrossEntropyLoss()

    # If in test mode, load the model and run the validation on the test dataset (optionally visualize the results)
    if cfg.mode == 'test':
        state_dict = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
        model.load_state_dict(state_dict['model'])
        model.eval()
        validate_fn = Validator(
            model=model,
            val_loader=test_loader,
            cfg=cfg,
            visualize=cfg.visualize
        )
        metrics = validate_fn()
        print(metrics)
        return
    
    # If resuming training, load the model, scheduler and optimizer state
    if cfg.resume:
        state_dict = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
        model.load_state_dict(state_dict['model'])
        best_instance_miou = state_dict['ins_miou']
        best_class_miou = state_dict['cls_miou']
        best_class_macc = state_dict['cls_macc']
        best_accuracy = state_dict['accuracy']
        cfg.SOLVER.start_epoch = state_dict['epoch'] + 1
    else:
        best_instance_miou, best_class_miou, best_class_macc, best_accuracy = 0. , 0. , 0., 0.
    
    # Create the validator object
    validate_fn = Validator(
        model=model,
        val_loader=val_loader,
        cfg=cfg
    )

    # Build the optimizer from the cfg
    optimizer = build_optimizer_from_cfg(
        cfg.SOLVER, 
        model=model
        )
    
    # Build the scheduler from the cfg
    scheduler = build_scheduler_from_cfg(
        cfg.SOLVER, 
        optimizer=optimizer
        )
    
    # Initialize Tensorboard SummaryWriter with the log directory
    run_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
    tb_log_dir = f'/home/appuser/src/seg_models/PointCloudSegmentation/logs/{run_name}'
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Training loop
    for i, epoch in tqdm(enumerate(range(cfg.SOLVER.start_epoch, cfg.SOLVER.epochs)), total=cfg.SOLVER.epochs - cfg.SOLVER.start_epoch):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg)
        is_best = False
        # validate every cfg.val_freq epochs (for my tranining it was every 2 epochs)
        if epoch % cfg.val_freq == 0:
            metrics = validate_fn()
            # Extract the validation matrics
            val_ins_miou = metrics['instance_avg_iou']
            val_cls_miou = metrics['class_avg_iou']
            val_class_macc = metrics['class_avg_accuracy']
            val_accuracy = metrics['accuracy']
            # Update the best metrics if the current metrics are better
            if val_ins_miou > best_instance_miou:
                best_instance_miou = val_ins_miou
                best_class_miou = val_cls_miou
                best_class_macc = val_class_macc
                best_accuracy = val_accuracy
                best_epoch = epoch
                is_best = True
                print(f"Best instance mIoU: {best_instance_miou} at epoch {best_epoch}")
            # Log the metrics to tensorboard
            if writer is not None:
                writer.add_scalar('val_ins_miou', val_ins_miou, epoch)
                writer.add_scalar('val_class_miou', val_cls_miou, epoch)
                writer.add_scalar('val_class_macc', val_class_macc, epoch)
                writer.add_scalar('val_accuracy', val_accuracy, epoch)

        # Log the loss and learning rate to tensorboard
        lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        # Step the scheduler
        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        # Save the model checkpoint (It only saves the best model based on the instance mIoU)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'ins_miou': best_instance_miou,
                                             'cls_miou': best_class_miou,
                                             'cls_macc': best_class_macc,
                                             'accuracy': best_accuracy,},
                            is_best=is_best,
                            save_name=f'pcs_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
                            )
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # logger
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    # if testing, resume or val, resume the exp folder
    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    main(cfg)
