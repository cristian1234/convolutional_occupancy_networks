'''
Iterative corridor generation script.

Usage:
    python generate_iterative.py \
        --input corridor_start.npy \
        --checkpoint out/corridor_completion/model.pt \
        --config configs/voxel_completion/corridor_grid64.yaml \
        --max_chunks 10 \
        --output_voxels result.npy \
        --output_mesh result.ply \
        --device mps
'''
import argparse
import os
import sys
import time
import numpy as np

# Enable MPS fallback for ops not yet supported (e.g. max_pool3d)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch

from src import config
from src.conv_onet.iterative_generation import IterativeGenerator3D
from src.utils.voxel_utils import save_voxels
from src.checkpoints import CheckpointIO


def main():
    parser = argparse.ArgumentParser(
        description='Generate corridors iteratively using trained completion model.'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to initial voxel grid (.npy)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/voxel_completion/corridor_grid64.yaml',
                        help='Path to config file')
    parser.add_argument('--max_chunks', type=int, default=10,
                        help='Maximum number of chunks to generate')
    parser.add_argument('--overlap', type=int, default=16,
                        help='Overlap size between chunks')
    parser.add_argument('--chunk_size', type=int, default=64,
                        help='Chunk size')
    parser.add_argument('--axis', type=int, default=0,
                        help='Generation axis (0=X, 1=Y, 2=Z)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Occupancy threshold')
    parser.add_argument('--output_voxels', type=str, default=None,
                        help='Output path for voxel grid (.npy)')
    parser.add_argument('--output_mesh', type=str, default=None,
                        help='Output path for mesh (.ply)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps')

    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')

    # Load config
    cfg = config.load_config(args.config, 'configs/default.yaml')

    # Load model
    model = config.get_model(cfg, device=device)
    checkpoint_io = CheckpointIO(
        os.path.dirname(args.checkpoint), model=model
    )
    checkpoint_io.load(os.path.basename(args.checkpoint))
    model.eval()
    print('Model loaded successfully')

    # Load input voxels
    initial_voxels = np.load(args.input).astype(np.float32)
    print(f'Input voxels shape: {initial_voxels.shape}')

    # Create generator
    generator = IterativeGenerator3D(
        model, device=device,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        generation_axis=args.axis,
        padding=cfg['data']['padding'],
    )

    # Generate
    print(f'Generating up to {args.max_chunks} chunks...')
    t0 = time.time()
    result, chunks = generator.generate_corridor(
        initial_voxels,
        max_chunks=args.max_chunks,
        return_intermediates=True,
    )
    elapsed = time.time() - t0

    print(f'Generated {len(chunks)} chunks in {elapsed:.2f}s')
    print(f'Result shape: {result.shape}')
    print(f'Occupied voxels: {(result >= 0.5).sum()} / {result.size} '
          f'({100 * (result >= 0.5).mean():.1f}%)')

    # Save voxels
    if args.output_voxels:
        save_voxels(result, args.output_voxels)
        print(f'Voxels saved to {args.output_voxels}')

    # Save mesh
    if args.output_mesh:
        print('Extracting mesh...')
        mesh = generator.generate_mesh(result, args.output_mesh)
        print(f'Mesh saved to {args.output_mesh}')
        print(f'  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}')


if __name__ == '__main__':
    main()
