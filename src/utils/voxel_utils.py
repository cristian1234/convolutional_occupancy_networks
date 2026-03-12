import numpy as np
from collections import deque


def load_voxels(path):
    ''' Load voxel grid from .npy file.

    Args:
        path (str): path to .npy file

    Returns:
        numpy array: voxel grid
    '''
    return np.load(path).astype(np.float32)


def save_voxels(voxels, path):
    ''' Save voxel grid to .npy file.

    Args:
        voxels (numpy array): voxel grid
        path (str): output path
    '''
    np.save(path, voxels.astype(np.float32))


def detect_openings(voxels, axis=0, threshold=0.3):
    ''' Detect openings (exits) in a voxel corridor along a given axis.

    An opening is a slice where the ratio of empty voxels in the interior
    is above the threshold, indicating the corridor continues.

    Args:
        voxels (numpy array): binary voxel grid (D, H, W)
        axis (int): axis along which to check for openings (0=X, 1=Y, 2=Z)
        threshold (float): minimum empty ratio to consider an opening

    Returns:
        dict: {
            'has_opening_start': bool,
            'has_opening_end': bool,
            'start_slice': int or None,
            'end_slice': int or None,
        }
    '''
    size = voxels.shape[axis]
    result = {
        'has_opening_start': False,
        'has_opening_end': False,
        'start_slice': None,
        'end_slice': None,
    }

    for end_name, slice_idx in [('start', 0), ('end', size - 1)]:
        if axis == 0:
            slc = voxels[slice_idx, :, :]
        elif axis == 1:
            slc = voxels[:, slice_idx, :]
        else:
            slc = voxels[:, :, slice_idx]

        total = slc.size
        empty = (slc < 0.5).sum()
        ratio = empty / total

        if ratio > threshold:
            result[f'has_opening_{end_name}'] = True
            result[f'{end_name}_slice'] = slice_idx

    return result


def extract_overlap_region(voxels, overlap_size, direction='forward', axis=0):
    ''' Extract the overlap region from a voxel grid.

    Args:
        voxels (numpy array): voxel grid (D, H, W)
        overlap_size (int): number of voxels in overlap
        direction (str): 'forward' = take from end, 'backward' = take from start
        axis (int): axis along which to extract

    Returns:
        numpy array: overlap region
    '''
    slices = [slice(None)] * 3
    size = voxels.shape[axis]

    if direction == 'forward':
        slices[axis] = slice(size - overlap_size, size)
    else:
        slices[axis] = slice(0, overlap_size)

    return voxels[tuple(slices)].copy()


def place_overlap_in_chunk(chunk_shape, overlap_voxels, overlap_size, direction='forward', axis=0):
    ''' Place overlap voxels into a new empty chunk and create corresponding mask.

    Args:
        chunk_shape (tuple): shape of the new chunk (D, H, W)
        overlap_voxels (numpy array): the overlap region
        overlap_size (int): size of overlap
        direction (str): 'forward' = place at start, 'backward' = place at end
        axis (int): axis along which to place

    Returns:
        tuple: (chunk_voxels, mask) both of shape chunk_shape
    '''
    chunk = np.zeros(chunk_shape, dtype=np.float32)
    mask = np.zeros(chunk_shape, dtype=np.float32)

    slices = [slice(None)] * 3
    if direction == 'forward':
        # Overlap from previous chunk goes at the start of the new chunk
        slices[axis] = slice(0, overlap_size)
    else:
        slices[axis] = slice(chunk_shape[axis] - overlap_size, chunk_shape[axis])

    chunk[tuple(slices)] = overlap_voxels
    mask[tuple(slices)] = 1.0

    return chunk, mask


def merge_chunks(chunks, overlap_size, axis=0):
    ''' Merge multiple voxel chunks with linear blending in overlap regions.

    Args:
        chunks (list): list of voxel grids (numpy arrays)
        overlap_size (int): number of voxels in overlap
        axis (int): axis along which chunks are concatenated

    Returns:
        numpy array: merged voxel grid
    '''
    if len(chunks) == 0:
        raise ValueError("No chunks to merge")
    if len(chunks) == 1:
        return chunks[0].copy()

    result = chunks[0].copy()

    for i in range(1, len(chunks)):
        chunk = chunks[i]
        chunk_size = chunk.shape[axis]
        non_overlap = chunk_size - overlap_size

        # Create blending weights for overlap region
        weights = np.linspace(1.0, 0.0, overlap_size).astype(np.float32)
        # Reshape weights for broadcasting
        shape = [1, 1, 1]
        shape[axis] = overlap_size
        weights = weights.reshape(shape)

        # Extract overlap regions
        result_slices = [slice(None)] * 3
        result_slices[axis] = slice(result.shape[axis] - overlap_size, result.shape[axis])

        chunk_slices = [slice(None)] * 3
        chunk_slices[axis] = slice(0, overlap_size)

        # Blend overlap
        blended = (
            weights * result[tuple(result_slices)] +
            (1.0 - weights) * chunk[tuple(chunk_slices)]
        )
        result[tuple(result_slices)] = blended

        # Append non-overlap part
        new_slices = [slice(None)] * 3
        new_slices[axis] = slice(overlap_size, chunk_size)
        result = np.concatenate([result, chunk[tuple(new_slices)]], axis=axis)

    return result


def flood_fill_3d(voxels, start_coords):
    ''' 3D flood fill from starting coordinates in empty space.

    Args:
        voxels (numpy array): binary voxel grid (D, H, W)
        start_coords (list of tuples): starting (d, h, w) coordinates

    Returns:
        numpy array: boolean mask of reachable empty voxels
    '''
    shape = voxels.shape
    visited = np.zeros(shape, dtype=bool)
    empty = voxels < 0.5

    queue = deque()
    for coord in start_coords:
        if (0 <= coord[0] < shape[0] and
            0 <= coord[1] < shape[1] and
            0 <= coord[2] < shape[2] and
            empty[coord] and not visited[coord]):
            queue.append(coord)
            visited[coord] = True

    while queue:
        d, h, w = queue.popleft()
        for dd, dh, dw in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nd, nh, nw = d+dd, h+dh, w+dw
            if (0 <= nd < shape[0] and 0 <= nh < shape[1] and 0 <= nw < shape[2]
                    and not visited[nd, nh, nw] and empty[nd, nh, nw]):
                visited[nd, nh, nw] = True
                queue.append((nd, nh, nw))

    return visited


def compute_connectivity(known_voxels, completed_voxels):
    ''' Compute connectivity ratio: how much of the empty space in
    the completed region is reachable from the known region.

    Args:
        known_voxels (numpy array): voxels in known region (binary)
        completed_voxels (numpy array): full voxel grid including completion

    Returns:
        float: connectivity ratio (0 to 1)
    '''
    empty = completed_voxels < 0.5
    total_empty = empty.sum()
    if total_empty == 0:
        return 1.0

    # Find empty voxels adjacent to known occupied voxels as seed points
    known_occ = known_voxels >= 0.5
    seeds = []
    shape = completed_voxels.shape
    for d in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                if empty[d, h, w] and known_occ[d, h, w] is False:
                    # Check if adjacent to known occupied
                    for dd, dh, dw in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        nd, nh, nw = d+dd, h+dh, w+dw
                        if (0 <= nd < shape[0] and 0 <= nh < shape[1] and 0 <= nw < shape[2]
                                and known_occ[nd, nh, nw]):
                            seeds.append((d, h, w))
                            break

    if not seeds:
        # Fallback: use center of known empty space
        known_empty = (known_voxels < 0.5)
        coords = np.argwhere(known_empty)
        if len(coords) > 0:
            center = tuple(coords[len(coords)//2])
            seeds = [center]
        else:
            return 0.0

    reachable = flood_fill_3d(completed_voxels, seeds)
    reachable_count = reachable.sum()

    return float(reachable_count) / float(total_empty)
