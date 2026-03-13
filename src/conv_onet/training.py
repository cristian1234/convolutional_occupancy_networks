import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer


# 3x3x3 neighbor-counting kernel for connectivity loss
_NEIGHBOR_KERNEL = None

def _get_neighbor_kernel(device):
    global _NEIGHBOR_KERNEL
    if _NEIGHBOR_KERNEL is None or _NEIGHBOR_KERNEL.device != device:
        k = torch.ones(1, 1, 3, 3, 3, device=device)
        k[0, 0, 1, 1, 1] = 0  # exclude center
        _NEIGHBOR_KERNEL = k
    return _NEIGHBOR_KERNEL


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
        completion_weight (float): extra weight on unknown-zone loss
        pos_weight (float): weight on occupied voxels in BCE (balances empty/occupied)
        connectivity_weight (float): weight for isolated-voxel penalty
    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False,
                 completion_weight=1.0, pos_weight=1.0,
                 connectivity_weight=0.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.completion_weight = completion_weight
        self.pos_weight = pos_weight
        self.connectivity_weight = connectivity_weight

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        batch_size = points.size(0)

        kwargs = {}

        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Completion-zone metrics if mask available
        if self.input_type == 'voxel_masked' and 'mask' in data:
            self._compute_completion_metrics(
                eval_dict, points_iou, occ_iou, p_out.probs,
                data['mask'], threshold
            )

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def _compute_completion_metrics(self, eval_dict, points, occ_gt, occ_pred_probs,
                                     mask_data, threshold):
        '''Compute per-zone metrics: completion IoU, precision, recall.'''
        device = self.device
        mask_vol = mask_data.to(device)
        p_coords = points
        if isinstance(p_coords, dict):
            p_coords = p_coords['p']
        p_norm = (p_coords + 0.5).clamp(0, 1)
        grid_size = mask_vol.shape[-1]
        p_idx = (p_norm * (grid_size - 1)).long().clamp(0, grid_size - 1)

        batch_size = p_idx.size(0)
        point_mask = torch.zeros(batch_size, p_idx.size(1), device=device)
        for b in range(batch_size):
            point_mask[b] = mask_vol[b, p_idx[b, :, 0], p_idx[b, :, 1], p_idx[b, :, 2]]

        # Unknown zone points
        unknown = point_mask < 0.5
        gt = (occ_gt >= 0.5)
        pred = (occ_pred_probs >= threshold)

        # Per-sample, then average
        comp_ious, comp_precs, comp_recalls = [], [], []
        for b in range(batch_size):
            unk_b = unknown[b]
            gt_b = gt[b][unk_b]
            pred_b = pred[b][unk_b]
            if unk_b.sum() == 0:
                continue
            tp = (gt_b & pred_b).sum().float()
            fp = (~gt_b & pred_b).sum().float()
            fn = (gt_b & ~pred_b).sum().float()
            union = tp + fp + fn
            comp_ious.append((tp / union).item() if union > 0 else 1.0)
            comp_precs.append((tp / (tp + fp)).item() if (tp + fp) > 0 else 1.0)
            comp_recalls.append((tp / (tp + fn)).item() if (tp + fn) > 0 else 1.0)

        if comp_ious:
            import numpy as np
            eval_dict['completion_iou'] = float(np.mean(comp_ious))
            eval_dict['completion_precision'] = float(np.mean(comp_precs))
            eval_dict['completion_recall'] = float(np.mean(comp_recalls))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        logits = self.model.decode(p, c, **kwargs).logits

        # BCE with pos_weight to balance occupied/empty
        pw = torch.tensor([self.pos_weight], device=device) if self.pos_weight != 1.0 else None
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none', pos_weight=pw)

        # Apply completion weight for voxel_masked input
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

        # Connectivity loss: penalize isolated occupied voxels
        if (self.connectivity_weight > 0.0 and
                self.input_type == 'voxel_masked'):
            conn_loss = self._connectivity_loss(logits, p, c)
            loss = loss + self.connectivity_weight * conn_loss

        return loss

    def _connectivity_loss(self, logits, p, c):
        '''Penalize occupied voxels with few occupied neighbors.

        Decodes a dense 64^3 grid, applies sigmoid, then uses a 3x3x3
        convolution to count neighbor occupancy. Occupied voxels with
        low neighbor count get penalized.
        '''
        device = self.device
        resolution = 64

        # Decode a dense grid for connectivity check
        with torch.no_grad():
            grid_points = make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (resolution,) * 3
            ).unsqueeze(0).expand(logits.size(0), -1, -1).to(device)

        grid_logits = self.model.decode(grid_points, c).logits
        grid_probs = torch.sigmoid(grid_logits)
        grid_vol = grid_probs.view(-1, 1, resolution, resolution, resolution)

        # Count occupied neighbors using 3x3x3 convolution
        kernel = _get_neighbor_kernel(device)
        neighbor_count = F.conv3d(grid_vol, kernel, padding=1)
        # Max possible neighbors = 26

        # Penalty: occupied voxels with few neighbors
        # isolated_penalty = prob * max(0, threshold - neighbor_count / 26)
        isolation = torch.clamp(3.0 - neighbor_count, min=0) / 3.0  # 0 if >=3 neighbors
        penalty = grid_vol * isolation
        return penalty.mean()
