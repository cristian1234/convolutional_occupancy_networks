import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.voxels import LocalVoxelEncoder


class MaskedLocalVoxelEncoder(LocalVoxelEncoder):
    ''' 3D-convolutional encoder for masked voxel input (scene completion).

    Extends LocalVoxelEncoder to accept 2-channel input:
        Channel 0: voxel occupancy (known geometry)
        Channel 1: mask (1=known, 0=to predict)

    All other functionality (UNet3D, scatter, grid/plane features) is inherited.

    Args:
        Same as LocalVoxelEncoder, plus:
        in_channels (int): number of input channels (default: 2)
    '''

    def __init__(self, in_channels=2, dim=3, c_dim=128,
                 unet=False, unet_kwargs=None,
                 unet3d=False, unet3d_kwargs=None,
                 plane_resolution=512, grid_resolution=None,
                 plane_type='xz', kernel_size=3, padding=0.1):
        # Initialize parent with default 1-channel conv_in (will be replaced)
        super().__init__(
            dim=dim, c_dim=c_dim,
            unet=unet, unet_kwargs=unet_kwargs,
            unet3d=unet3d, unet3d_kwargs=unet3d_kwargs,
            plane_resolution=plane_resolution,
            grid_resolution=grid_resolution,
            plane_type=plane_type,
            kernel_size=kernel_size,
            padding=padding,
        )
        # Replace conv_in to accept in_channels instead of 1
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(in_channels, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(in_channels, c_dim, kernel_size, padding=1)

        self.in_channels = in_channels

    def forward(self, x):
        ''' Forward pass.

        Args:
            x (tensor): input tensor of shape (B, in_channels, D, H, W)
                         or (B, D, H, W) for backward compat with 1-channel
        '''
        # Handle input shape
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W) for backward compat
            x = x.unsqueeze(1)
        # x is now (B, C, D, H, W)

        batch_size = x.size(0)
        device = x.device
        D, H, W = x.size(2), x.size(3), x.size(4)
        n_voxel = D * H * W

        # Voxel 3D coordinates
        coord1 = torch.linspace(-0.5, 0.5, D).to(device)
        coord2 = torch.linspace(-0.5, 0.5, H).to(device)
        coord3 = torch.linspace(-0.5, 0.5, W).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand(batch_size, D, H, W)
        coord2 = coord2.view(1, 1, -1, 1).expand(batch_size, D, H, W)
        coord3 = coord3.view(1, 1, 1, -1).expand(batch_size, D, H, W)
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)

        # Acquire voxel-wise feature from multi-channel input
        # x is already (B, C, D, H, W), no need to unsqueeze
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        return fea
