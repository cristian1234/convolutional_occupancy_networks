'''
Distributed Data Parallel training for corridor completion.

Launch with torchrun:
    torchrun --nproc_per_node=4 train_ddp.py configs/voxel_completion/corridor_grid64.yaml

Or with a specific master port:
    torchrun --nproc_per_node=4 --master_port=29500 train_ddp.py configs/voxel_completion/corridor_grid64.yaml

Resume from checkpoint:
    torchrun --nproc_per_node=4 train_ddp.py configs/voxel_completion/corridor_grid64.yaml --resume
'''
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import time
import datetime
import shutil
import matplotlib; matplotlib.use('Agg')

from src import config, data
from src.checkpoints import CheckpointIO
from src.conv_onet import training
from collections import defaultdict


def setup_ddp():
    '''Initialize distributed process group.'''
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    '''Destroy distributed process group.'''
    dist.destroy_process_group()


def is_main(rank):
    '''Check if this is the main process.'''
    return rank == 0


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train with DistributedDataParallel (multi-GPU).'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after N seconds.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint.')
    args = parser.parse_args()

    # DDP setup
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    if is_main(rank):
        print(f'DDP: {world_size} GPUs, backend=nccl')
        for i in range(world_size):
            if i == rank:
                print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

    # Config
    cfg = config.load_config(args.config, 'configs/default.yaml')
    t0 = time.time()

    # Shorthands
    out_dir = cfg['training']['out_dir']
    # Scale effective batch size: per-GPU batch * world_size
    per_gpu_batch = cfg['training']['batch_size']
    effective_batch = per_gpu_batch * world_size
    backup_every = cfg['training']['backup_every']
    exit_after = args.exit_after

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be maximize or minimize.')

    # Output directory (only rank 0)
    if is_main(rank):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

    # Wait for rank 0 to create dirs
    dist.barrier()

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg, return_idx=True)

    # Distributed sampler - handles data sharding across GPUs
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=cfg['training']['n_workers'],
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Validation only on rank 0
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        num_workers=cfg['training']['n_workers_val'],
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
    )

    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)

    # Wrap in DDP
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # Optimizer - scale LR linearly with world_size
    base_lr = 1e-4
    scaled_lr = base_lr * world_size
    optimizer = optim.Adam(ddp_model.parameters(), lr=scaled_lr)

    if is_main(rank):
        print(f'Per-GPU batch: {per_gpu_batch}, effective batch: {effective_batch}')
        print(f'Base LR: {base_lr}, scaled LR: {scaled_lr}')
        nparams = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {nparams:,}')

    # Trainer uses the underlying model (not DDP wrapper) for encode/decode
    # but DDP wrapper for the forward pass that computes gradients
    completion_weight = cfg['training'].get('completion_weight', 1.0)
    trainer = DDPTrainer(
        ddp_model, optimizer, device=device,
        input_type=cfg['data']['input_type'],
        threshold=cfg['test']['threshold'],
        completion_weight=completion_weight,
    )

    # Checkpoint
    # Save/load from underlying model (not DDP wrapper)
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    epoch_it = 0
    it = 0
    metric_val_best = -model_selection_sign * np.inf

    if args.resume:
        try:
            load_dict = checkpoint_io.load('model.pt')
            epoch_it = load_dict.get('epoch_it', 0)
            it = load_dict.get('it', 0)
            metric_val_best = load_dict.get(
                'loss_val_best', -model_selection_sign * np.inf)
            if is_main(rank):
                print(f'Resumed from it={it}, best={metric_val_best:.6f}')
        except FileExistsError:
            if is_main(rank):
                print('No checkpoint found, starting fresh.')

    if is_main(rank):
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Training intervals
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    eval_every = cfg['training'].get('eval_every', 0)
    iterative_test_every = cfg['training'].get('iterative_test_every', 0)

    if is_main(rank):
        print(f'Output: {out_dir}')
        print('Starting training...')

    # Training loop
    while True:
        epoch_it += 1
        train_sampler.set_epoch(epoch_it)  # Crucial for proper shuffling

        for batch in train_loader:
            it += 1
            loss = trainer.train_step(batch)

            if is_main(rank):
                logger.add_scalar('train/loss', loss, it)

                # Print
                if print_every > 0 and (it % print_every) == 0:
                    t = datetime.datetime.now()
                    elapsed = time.time() - t0
                    print(f'[Epoch {epoch_it:02d}] it={it:05d}, loss={loss:.4f}, '
                          f'time={elapsed:.0f}s, {t.hour:02d}:{t.minute:02d}')

                # Checkpoint
                if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                    print('Saving checkpoint')
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)

                # Backup
                if backup_every > 0 and (it % backup_every) == 0:
                    print('Backup checkpoint')
                    checkpoint_io.save(f'model_{it}.pt', epoch_it=epoch_it,
                                       it=it, loss_val_best=metric_val_best)

            # Validation (all ranks wait, but only rank 0 computes)
            if validate_every > 0 and (it % validate_every) == 0:
                if is_main(rank):
                    # Use the underlying model for eval (no DDP wrapper needed)
                    model.eval()
                    eval_dict = trainer.evaluate(val_loader)
                    metric_val = eval_dict[model_selection_metric]
                    print(f'Validation {model_selection_metric}: {metric_val:.4f}')

                    for k, v in eval_dict.items():
                        logger.add_scalar(f'val/{k}', v, it)

                    if model_selection_sign * (metric_val - metric_val_best) > 0:
                        metric_val_best = metric_val
                        print(f'New best model ({metric_val_best:.4f})')
                        checkpoint_io.save('model_best.pt', epoch_it=epoch_it,
                                           it=it, loss_val_best=metric_val_best)

                # All ranks wait for validation to finish
                dist.barrier()

            # Auto evaluation (rank 0 only)
            if (is_main(rank) and eval_every > 0 and (it % eval_every) == 0
                    and cfg['data']['input_type'] == 'voxel_masked'):
                try:
                    from src.evaluation import auto_eval, voxel_visualizer
                    model.eval()
                    metrics = auto_eval.evaluate(model, val_dataset, device)
                    auto_eval.log_metrics(metrics, it, out_dir)

                    # Snapshots
                    test_samples = [val_dataset[i] for i in range(min(5, len(val_dataset)))]
                    auto_eval.save_voxel_snapshots(
                        model, test_samples, it, out_dir, device)
                    voxel_visualizer.render_all(
                        model, test_samples, it, out_dir, device)

                    for k, v in metrics.items():
                        logger.add_scalar(f'eval/{k}', v, it)
                except Exception as e:
                    print(f'Auto-eval error at it={it}: {e}')

            # Iterative test (rank 0 only)
            if (is_main(rank) and iterative_test_every > 0
                    and (it % iterative_test_every) == 0
                    and cfg['data']['input_type'] == 'voxel_masked'):
                try:
                    from src.evaluation.iterative_test import run_iterative_test
                    model.eval()
                    run_iterative_test(
                        model, device, n_chunks=5,
                        out_dir=out_dir, iteration=it)
                except Exception as e:
                    print(f'Iterative test error at it={it}: {e}')

            # Exit
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                if is_main(rank):
                    print('Time limit reached. Exiting.')
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)
                cleanup_ddp()
                exit(3)

    cleanup_ddp()


class DDPTrainer(training.Trainer):
    '''Trainer adapted for DDP.

    The DDP-wrapped model handles gradient sync automatically.
    We just need to make sure encode_inputs and decode go through
    the DDP forward.
    '''

    def __init__(self, ddp_model, optimizer, device=None,
                 input_type='pointcloud', threshold=0.5,
                 completion_weight=1.0):
        # The DDP model wraps the original ConvONet
        self.ddp_model = ddp_model
        self.model = ddp_model.module  # underlying model for eval
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.threshold = threshold
        self.eval_sample = False
        self.completion_weight = completion_weight
        self.vis_dir = None

    def train_step(self, data):
        '''Training step using DDP model.'''
        self.ddp_model.train()
        self.optimizer.zero_grad()
        loss = self._compute_loss_ddp(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _compute_loss_ddp(self, data):
        '''Compute loss through DDP forward for gradient sync.'''
        from torch.nn import functional as F
        from src.common import add_key

        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        if 'pointcloud_crop' in data.keys():
            inputs = add_key(inputs, data.get('inputs.ind'),
                             'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            p = add_key(p, data.get('points.normalized'),
                        'p', 'p_n', device=device)

        # Use DDP model for forward (triggers gradient sync)
        c = self.ddp_model.module.encode_inputs(inputs)

        kwargs = {}
        logits = self.ddp_model.module.decode(p, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')

        # Weighted loss for completion zone
        if (self.input_type == 'voxel_masked' and
                self.completion_weight != 1.0 and
                'mask' in data):
            mask_vol = data['mask'].to(device)
            p_coords = p
            if isinstance(p, dict):
                p_coords = p['p']
            p_norm = (p_coords + 0.5).clamp(0, 1)
            grid_size = mask_vol.shape[-1]
            p_idx = (p_norm * (grid_size - 1)).long().clamp(0, grid_size - 1)
            batch_size = p_idx.size(0)
            point_mask = torch.zeros(batch_size, p_idx.size(1), device=device)
            for b in range(batch_size):
                point_mask[b] = mask_vol[b,
                                         p_idx[b, :, 0],
                                         p_idx[b, :, 1],
                                         p_idx[b, :, 2]]
            weights = torch.where(
                point_mask > 0.5,
                torch.ones_like(point_mask),
                torch.full_like(point_mask, self.completion_weight)
            )
            loss_i = loss_i * weights

        loss = loss_i.sum(-1).mean()
        return loss


if __name__ == '__main__':
    main()
