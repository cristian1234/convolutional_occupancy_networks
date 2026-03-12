from src.encoder import (
    pointnet, voxels, pointnetpp, voxels_masked
)


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
    'voxel_masked_local': voxels_masked.MaskedLocalVoxelEncoder,
}
