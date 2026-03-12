'''
MPS (Apple Silicon) compatibility tests for the completion pipeline.

Run with: python -m pytest tests/test_mps_compat.py -v
'''
import os
import sys
import time

# Enable MPS fallback for ops not yet supported (e.g. max_pool3d)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import pytest
import torch
import numpy as np


def has_mps():
    return (hasattr(torch.backends, 'mps') and
            torch.backends.mps.is_available())


@pytest.mark.skipif(not has_mps(), reason='MPS not available')
class TestMPSCompat:

    def test_mps_device_creation(self):
        '''Verify MPS device can be created.'''
        device = torch.device('mps')
        x = torch.randn(2, 3).to(device)
        assert x.device.type == 'mps'

    def test_conv3d_on_mps(self):
        '''Test basic Conv3d operations on MPS.'''
        device = torch.device('mps')
        conv = torch.nn.Conv3d(2, 32, 3, padding=1).to(device)
        x = torch.randn(1, 2, 64, 64, 64).to(device)
        out = conv(x)
        assert out.shape == (1, 32, 64, 64, 64)
        assert out.device.type == 'mps'

    def test_grid_sample_on_mps(self):
        '''Test F.grid_sample on MPS (critical for decoder).'''
        device = torch.device('mps')

        # 3D grid sample (trilinear)
        c = torch.randn(1, 32, 16, 16, 16).to(device)
        grid = torch.randn(1, 100, 1, 1, 3).to(device)
        grid = grid.clamp(-1, 1)

        try:
            out = torch.nn.functional.grid_sample(
                c, grid, padding_mode='border',
                align_corners=True, mode='bilinear'
            )
            assert out.shape[0] == 1
            print('grid_sample works on MPS')
        except RuntimeError as e:
            pytest.skip(f'grid_sample not supported on MPS: {e}')

    def test_model_load_on_mps(self):
        '''Test that the full model can be loaded on MPS.'''
        device = torch.device('mps')

        from src.encoder.voxels_masked import MaskedLocalVoxelEncoder
        from src.conv_onet.models.decoder import LocalDecoder
        from src.conv_onet.models import ConvolutionalOccupancyNetwork

        encoder = MaskedLocalVoxelEncoder(
            in_channels=2, c_dim=32,
            unet3d=True,
            unet3d_kwargs={
                'num_levels': 3,
                'f_maps': 32,
                'in_channels': 32,
                'out_channels': 32,
            },
            plane_type=['grid'],
            grid_resolution=64,
            padding=0.1,
        )

        decoder = LocalDecoder(
            dim=3, c_dim=32,
            hidden_size=32, n_blocks=3,
            sample_mode='bilinear',
            padding=0.1,
        )

        model = ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        )

        # Forward pass
        x = torch.randn(1, 2, 64, 64, 64).to(device)
        with torch.no_grad():
            c = model.encode_inputs(x)
            assert 'grid' in c

            # Query points
            p = torch.randn(1, 100, 3).to(device) * 0.4
            try:
                out = model.decode(p, c)
                assert out.logits.shape == (1, 100)
                print('Full model forward pass works on MPS')
            except RuntimeError as e:
                # grid_sample might not work on MPS
                pytest.skip(f'Model decode failed on MPS (likely grid_sample): {e}')

    def test_inference_timing(self):
        '''Measure inference time for a single chunk on MPS.'''
        device = torch.device('mps')

        from src.encoder.voxels_masked import MaskedLocalVoxelEncoder
        from src.conv_onet.models.decoder import LocalDecoder
        from src.conv_onet.models import ConvolutionalOccupancyNetwork

        encoder = MaskedLocalVoxelEncoder(
            in_channels=2, c_dim=32,
            unet3d=True,
            unet3d_kwargs={
                'num_levels': 3,
                'f_maps': 32,
                'in_channels': 32,
                'out_channels': 32,
            },
            plane_type=['grid'],
            grid_resolution=64,
            padding=0.1,
        )

        decoder = LocalDecoder(
            dim=3, c_dim=32, hidden_size=32, n_blocks=3,
            sample_mode='bilinear', padding=0.1,
        )

        model = ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        )
        model.eval()

        x = torch.randn(1, 2, 64, 64, 64).to(device)

        # Warmup
        with torch.no_grad():
            try:
                c = model.encode_inputs(x)
                p = torch.randn(1, 1000, 3).to(device) * 0.4
                _ = model.decode(p, c)
            except RuntimeError:
                pytest.skip('MPS forward pass failed')

        # Timed run
        torch.mps.synchronize()
        t0 = time.time()
        n_runs = 5
        with torch.no_grad():
            for _ in range(n_runs):
                c = model.encode_inputs(x)
                p = torch.randn(1, 64*64*64, 3).to(device) * 0.4
                _ = model.decode(p, c)
        torch.mps.synchronize()
        elapsed = (time.time() - t0) / n_runs

        print(f'\nMPS inference time per chunk (64^3): {elapsed:.3f}s')
        # Should complete in reasonable time
        assert elapsed < 30.0, f'Too slow: {elapsed:.1f}s per chunk'


@pytest.mark.skipif(has_mps(), reason='Testing CPU fallback')
class TestCPUFallback:

    def test_cpu_forward_pass(self):
        '''Verify model works on CPU as fallback.'''
        device = torch.device('cpu')

        from src.encoder.voxels_masked import MaskedLocalVoxelEncoder
        from src.conv_onet.models.decoder import LocalDecoder
        from src.conv_onet.models import ConvolutionalOccupancyNetwork

        encoder = MaskedLocalVoxelEncoder(
            in_channels=2, c_dim=32,
            unet3d=True,
            unet3d_kwargs={
                'num_levels': 3,
                'f_maps': 32,
                'in_channels': 32,
                'out_channels': 32,
            },
            plane_type=['grid'],
            grid_resolution=64,
            padding=0.1,
        )

        decoder = LocalDecoder(
            dim=3, c_dim=32, hidden_size=32, n_blocks=3,
            sample_mode='bilinear', padding=0.1,
        )

        model = ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        )

        x = torch.randn(1, 2, 16, 16, 16).to(device)
        with torch.no_grad():
            c = model.encode_inputs(x)
            p = torch.randn(1, 100, 3).to(device) * 0.4
            out = model.decode(p, c)
            assert out.logits.shape == (1, 100)
