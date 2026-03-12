import torch
import torch.nn.functional as F
import numpy as np
import logging
from src.common import make_3d_grid, normalize_3d_coordinate
from src.utils.voxel_utils import (
    detect_openings, extract_overlap_region,
    place_overlap_in_chunk, merge_chunks
)

logger = logging.getLogger(__name__)


class IterativeGenerator3D(object):
    ''' Iterative corridor generator using scene completion.

    Takes a trained completion model and generates corridors chunk by chunk,
    using overlap regions for consistency.

    Args:
        model (nn.Module): trained ConvONet completion model
        device (device): pytorch device
        threshold (float): occupancy threshold
        chunk_size (int): size of each voxel chunk (assumes cubic)
        overlap_size (int): overlap between consecutive chunks
        generation_axis (int): axis along which to extend (0=X, 1=Y, 2=Z)
        padding (float): padding for coordinate normalization
    '''

    def __init__(self, model, device=None, threshold=0.5,
                 chunk_size=64, overlap_size=16,
                 generation_axis=0, padding=0.1):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.generation_axis = generation_axis
        self.padding = padding

    def generate_corridor(self, initial_voxels, max_chunks=10,
                          return_intermediates=False):
        ''' Generate a corridor by iteratively completing chunks.

        Args:
            initial_voxels (numpy array): starting voxel grid (D, H, W)
            max_chunks (int): maximum number of chunks to generate
            return_intermediates (bool): whether to return individual chunks

        Returns:
            numpy array: completed voxel grid
            list (optional): list of individual chunks if return_intermediates
        '''
        self.model.eval()
        chunks = [initial_voxels.copy()]
        chunk_shape = (self.chunk_size,) * 3

        for i in range(max_chunks - 1):
            logger.info(f'Generating chunk {i+1}/{max_chunks-1}')

            # Extract overlap from the end of the last chunk
            prev_chunk = chunks[-1]
            overlap = extract_overlap_region(
                prev_chunk, self.overlap_size,
                direction='forward', axis=self.generation_axis
            )

            # Place overlap at the beginning of new chunk
            new_chunk_voxels, mask = place_overlap_in_chunk(
                chunk_shape, overlap, self.overlap_size,
                direction='forward', axis=self.generation_axis
            )

            # Run completion
            completed = self._complete_chunk(new_chunk_voxels, mask)
            chunks.append(completed)

            # Check if there's an opening to continue
            openings = detect_openings(
                completed, axis=self.generation_axis, threshold=0.3
            )
            if not openings['has_opening_end']:
                logger.info(f'No opening detected at chunk {i+1}, stopping.')
                break

        # Merge all chunks
        merged = merge_chunks(
            chunks, self.overlap_size, axis=self.generation_axis
        )

        if return_intermediates:
            return merged, chunks
        return merged

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
            output_path (str): optional path to save .ply file

        Returns:
            trimesh.Trimesh: extracted mesh
        '''
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            from src.utils import libmcubes
            # Pad to ensure watertight
            padded = np.pad(voxels, 1, 'constant', constant_values=0)
            vertices, triangles = libmcubes.marching_cubes(padded, 0.5)
            vertices -= 1  # undo padding offset
            import trimesh
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            if output_path:
                mesh.export(output_path)
            return mesh

        # Use skimage marching cubes
        padded = np.pad(voxels, 1, 'constant', constant_values=0)
        vertices, faces, normals, _ = marching_cubes(padded, level=0.5)
        vertices -= 1  # undo padding offset

        import trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces,
            vertex_normals=normals, process=False
        )

        if output_path:
            mesh.export(output_path)

        return mesh
