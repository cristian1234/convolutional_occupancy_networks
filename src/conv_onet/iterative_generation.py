import torch
import torch.nn.functional as F
import numpy as np
import logging
from src.common import make_3d_grid, normalize_3d_coordinate
from src.utils.voxel_utils import detect_openings

logger = logging.getLogger(__name__)


class IterativeGenerator3D(object):
    ''' Iterative corridor generator using sliding window completion.

    Instead of generating whole chunks, slides a 64-voxel window forward
    by a small step_size, predicting only a narrow strip of new voxels
    at each step. This keeps ~90% of the window as known context.

    Args:
        model (nn.Module): trained ConvONet completion model
        device (device): pytorch device
        threshold (float): occupancy threshold
        chunk_size (int): size of the model's input window (64)
        step_size (int): new voxels to predict per step (default 4)
        generation_axis (int): axis along which to extend (0=X, 1=Y, 2=Z)
        padding (float): padding for coordinate normalization
        overlap_size (int): legacy param, converted to step_size
    '''

    def __init__(self, model, device=None, threshold=0.5,
                 chunk_size=64, step_size=4, overlap_size=None,
                 generation_axis=0, padding=0.1):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.generation_axis = generation_axis
        self.padding = padding

        # Legacy: if overlap_size given, convert
        if overlap_size is not None:
            self.step_size = min(step_size, chunk_size - overlap_size)
        else:
            self.step_size = step_size

    def generate_corridor(self, initial_voxels, n_steps=None,
                          max_chunks=10, return_intermediates=False):
        ''' Generate a corridor by sliding window completion.

        Args:
            initial_voxels (numpy array): starting voxel grid (D, H, W)
            n_steps (int): number of sliding steps
            max_chunks (int): legacy param, converted to n_steps
            return_intermediates (bool): return per-step snapshots

        Returns:
            numpy array: completed voxel grid
            list (optional): list of per-step windows
        '''
        self.model.eval()
        axis = self.generation_axis
        cs = self.chunk_size
        step = self.step_size

        if n_steps is None:
            # Legacy: approximate old behavior
            n_steps = max_chunks * cs // step

        corridor = initial_voxels.copy()
        intermediates = [] if return_intermediates else None

        for i in range(n_steps):
            current_length = corridor.shape[axis]

            # Extend corridor buffer with zeros for the new strip
            extend_shape = list(corridor.shape)
            extend_shape[axis] = step
            corridor = np.concatenate(
                [corridor, np.zeros(extend_shape, dtype=np.float32)],
                axis=axis
            )

            new_length = current_length + step

            # Extract 64-voxel window ending at the new frontier
            window_start = new_length - cs
            window_slices = [slice(None)] * 3
            window_slices[axis] = slice(window_start, new_length)
            window_voxels = corridor[tuple(window_slices)].copy()

            # Mask: everything known except the last `step` voxels
            mask = np.ones_like(window_voxels)
            mask_slices = [slice(None)] * 3
            mask_slices[axis] = slice(cs - step, cs)
            mask[tuple(mask_slices)] = 0.0

            # Run model completion
            completed = self._complete_chunk(window_voxels, mask)

            # Write only the new strip back into the corridor
            src_slices = [slice(None)] * 3
            src_slices[axis] = slice(cs - step, cs)

            dst_slices = [slice(None)] * 3
            dst_slices[axis] = slice(new_length - step, new_length)

            corridor[tuple(dst_slices)] = completed[tuple(src_slices)]

            if return_intermediates:
                intermediates.append(completed)

            if (i + 1) % 10 == 0:
                logger.info(f'Step {i+1}/{n_steps}')

        if return_intermediates:
            return corridor, intermediates
        return corridor

    def _complete_chunk(self, voxels, mask):
        ''' Complete a single chunk using the model.

        Args:
            voxels (numpy array): partial voxel grid (D, H, W)
            mask (numpy array): mask grid (D, H, W), 1=known, 0=predict

        Returns:
            numpy array: completed binary voxel grid
        '''
        # Prepare 2-channel input: (1, 2, D, H, W)
        voxels_t = torch.FloatTensor(voxels).unsqueeze(0).unsqueeze(0)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)
        x = torch.cat([voxels_t, mask_t], dim=1).to(self.device)

        # Encode
        with torch.no_grad():
            c = self.model.encode_inputs(x)

        # Create query points covering the full chunk
        resolution = self.chunk_size
        query_points = make_3d_grid(
            (-0.5,) * 3, (0.5,) * 3, (resolution,) * 3
        )
        query_points = query_points.unsqueeze(0).to(self.device)

        # Decode occupancy
        with torch.no_grad():
            occ_logits = self.model.decode(query_points, c).logits

        # Convert to probabilities and threshold
        occ_probs = torch.sigmoid(occ_logits).squeeze(0).cpu().numpy()
        occ_grid = occ_probs.reshape(resolution, resolution, resolution)

        # Threshold to binary
        completed = (occ_grid >= self.threshold).astype(np.float32)

        # Preserve known region exactly
        known_mask = mask > 0.5
        completed[known_mask] = voxels[known_mask]

        return completed

    def generate_mesh(self, voxels, output_path=None):
        ''' Extract mesh from voxel grid using marching cubes.

        Args:
            voxels (numpy array): binary voxel grid
            output_path (str): optional path to save mesh

        Returns:
            trimesh.Trimesh: extracted mesh
        '''
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            from src.utils import libmcubes
            padded = np.pad(voxels, 1, 'constant', constant_values=0)
            vertices, triangles = libmcubes.marching_cubes(padded, 0.5)
            vertices -= 1
            import trimesh
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            if output_path:
                mesh.export(output_path)
            return mesh

        padded = np.pad(voxels, 1, 'constant', constant_values=0)
        vertices, faces, normals, _ = marching_cubes(padded, level=0.5)
        vertices -= 1

        import trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces,
            vertex_normals=normals, process=False
        )

        if output_path:
            mesh.export(output_path)

        return mesh
