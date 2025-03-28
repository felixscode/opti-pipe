from opti_pipe.utils import Config

from scipy.signal import convolve2d
import numpy as np
from functools import partial
import shapely
import matplotlib.pyplot as plt
import multiprocessing as mp
import torch
import torch.nn.functional as F

Model = type("Model", (), {})

def _pos_to_index_mapper(minx, maxx, miny, maxy, x_steps, y_steps, x, y):
    if maxx == minx or maxy == miny:
        raise ValueError("Invalid envelope dimensions: max must be greater than min.")

    # Normalize x and y to a value between 0 and 1.
    norm_x = (x - minx) / (maxx - minx)
    norm_y = (y - miny) / (maxy - miny)
    
    # Multiply by number of steps to get the index.
    # Using int() here essentially floors the result.
    i = int(norm_x * x_steps)
    j = int(norm_y * y_steps)
    
    # Ensure that indices are within bounds (if x == maxx, we force the last index)
    i = min(max(i, 0), x_steps - 1)
    j = min(max(j, 0), y_steps - 1)
    
    return i, j

def _index_to_pos_mapper(minx, maxx, miny, maxy, x_steps, y_steps, j, i):
    if x_steps <= 0 or y_steps <= 0:
        raise ValueError("Grid steps must be greater than zero.")

    if not (0 <= i < x_steps) or not (0 <= j < y_steps):
        raise ValueError("Index out of bounds.")

    # Compute step sizes
    x_step_size = (maxx - minx) / x_steps
    y_step_size = (maxy - miny) / y_steps

    # Map index to position (center of the grid cell)
    x = minx + (i + 0.5) * x_step_size
    y = maxy - (j + 0.5) * y_step_size

    return x, y

def _get_init_heat_matrix(model: Model, resolution) -> tuple:
    """
    Initialize the heat matrix for the model.
    """
    floor_envelope = model.floor.geometry.envelope
    minx, miny, maxx, maxy = floor_envelope.bounds
    x_range = maxx - minx
    y_range = maxy - miny
    x_steps = int(x_range / resolution)
    y_steps = int(y_range / resolution)
    heat_matrix = np.zeros((x_steps, y_steps))
    index_mapper = partial(_pos_to_index_mapper, minx, maxx, miny, maxy, x_steps, y_steps)
    pos_mapper = partial(_index_to_pos_mapper, minx, maxx, miny, maxy, x_steps, y_steps)
    return heat_matrix, index_mapper, pos_mapper

def _process_index_tuples(index_tuple, pipes, pos_mapper, resolution):
    i, j = index_tuple
    pos = pos_mapper(i, j)
    point = shapely.geometry.Point(pos)
    val = 0
    for pipe, heat in pipes:
        if pipe.distance(point) < resolution:
            val = heat
            break
    return ((i, j), val)

def _get_pipe_mask(model, matrix, pos_mapper, resolution):
    index_tuples = [(i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    pipes = [(pipe.geometry, pipe.heat) for pipe in model.pipes]
    # Use multiprocessing to speed-up mask generation.
    with mp.Pool() as pool:
        results = pool.map(partial(_process_index_tuples, pipes=pipes, pos_mapper=pos_mapper, resolution=resolution), index_tuples)
    
    pipe_mask = {t: v for t, v in results if v != 0}
    return pipe_mask

def apply_mask(matrix, pipe_mask):
    if pipe_mask:
        indices = tuple(zip(*pipe_mask.keys()))
        matrix[indices] = list(pipe_mask.values())

def _apply_kernel(matrix, kernel, pipe_mask, iterations):
    """
    Applies convolution using PyTorch for performance improvement.
    The matrix and kernel are converted to torch tensors (and moved to GPU if available).
    After each iteration, the pipe_mask is reapplied.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert matrix to torch tensor and add batch and channel dimensions: [1, 1, H, W]
    mat = torch.tensor(matrix, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Convert kernel to torch tensor and reshape to [1, 1, kH, kW]
    kernel_t = torch.tensor(kernel, dtype=torch.float32, device=device)
    kernel_t = kernel_t.unsqueeze(0).unsqueeze(0)
    
    # Pre-calculate padding to achieve "same" convolution
    pad = kernel.shape[0] // 2

    # Convert pipe_mask to index tensors if not empty.
    if pipe_mask:
        idx0, idx1 = zip(*pipe_mask.keys())
        idx0 = torch.tensor(idx0, dtype=torch.long, device=device)
        idx1 = torch.tensor(idx1, dtype=torch.long, device=device)
        mask_values = torch.tensor(list(pipe_mask.values()), dtype=torch.float32, device=device)
    else:
        idx0, idx1, mask_values = None, None, None

    def apply_mask_torch(mat_tensor):
        if idx0 is not None:
            # Reapply the mask: note that mat_tensor shape is [1, 1, H, W]
            mat_tensor[0, 0, idx0, idx1] = mask_values
        return mat_tensor

    # Initial mask application
    mat = apply_mask_torch(mat)

    for _ in range(iterations):
        # Convolution operation with padding for same output size.
        mat = F.conv2d(mat, kernel_t, padding=pad)
        # Reapply pipe mask
        mat = apply_mask_torch(mat)

    # Remove batch and channel dimensions and move back to CPU, then convert to numpy array.
    result = mat.squeeze().cpu().numpy()
    return result

def _make_kernel(size):
    """Create a Gaussian kernel with the given size."""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    sigma = size / 6.0  # Approximation to cover 99.7% within the kernel
    center = size // 2
    kernel = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel /= np.sum(kernel)  # Normalize the kernel
    return kernel

def get_heat_distribution(model: Model, config: Config, resolution: float) -> np.ndarray:
    """
    Distribute heat from the distributor to the rooms using a GPU-accelerated convolution.
    """
    heat_matrix, index_mapper, pos_mapper = _get_init_heat_matrix(model, resolution)
    pipe_mask = _get_pipe_mask(model, heat_matrix, pos_mapper, resolution)
    
    # Apply initial pipe mask
    indices = tuple(zip(*pipe_mask.keys()))
    heat_matrix[indices] = list(pipe_mask.values())
    
    kernel = _make_kernel(config.heat.conv_kernel_size)
    heat_matrix = _apply_kernel(heat_matrix, kernel=kernel, pipe_mask=pipe_mask, iterations=config.heat.conv_iterations)
    
    # Normalize the heat matrix between 0 and 1.
    heat_matrix = (heat_matrix - np.min(heat_matrix)) / (np.max(heat_matrix) - np.min(heat_matrix))
    
    return heat_matrix

def render_heat(model, config, resolution, floor):
    heat_matrix = get_heat_distribution(model, config, resolution)
    minx, miny, maxx, maxy = floor.envelope.bounds
    extent = [minx, maxx, miny, maxy]

    plt.imshow(heat_matrix, extent=extent, cmap='coolwarm', interpolation='nearest', alpha=0.5)
    plt.show()
