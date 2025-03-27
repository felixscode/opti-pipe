from opti_pipe.utils import Config
from opti_pipe.models import Model
import numpy as np
from functools import partial
import shapely


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
def _index_to_pos_mapper(minx, maxx, miny, maxy, x_steps, y_steps, i, j):

    if x_steps <= 0 or y_steps <= 0:
        raise ValueError("Grid steps must be greater than zero.")

    if not (0 <= i < x_steps) or not (0 <= j < y_steps):
        raise ValueError("Index out of bounds.")

    # Compute step sizes
    x_step_size = (maxx - minx) / x_steps
    y_step_size = (maxy - miny) / y_steps

    # Map index to position (center of the grid cell)
    x = minx + (i + 0.5) * x_step_size
    y = miny + (j + 0.5) * y_step_size

    return x, y
def _get_init_heat_matrix(model: Model, resolution) -> Model:
    """
    Initialize the heat matrix for the model.
    """
    floor_envelope = model.floor.geometry.envelope
    minx, miny, maxx, maxy = floor_envelope.bounds
    x_range = maxx - minx
    y_range = maxy - miny
    x_steps = int(x_range / resolution)
    y_steps = int(y_range / resolution)
    return np.zeros((x_steps, y_steps)),partial(_pos_to_index_mapper,minx, maxx, miny, maxy, x_steps, y_steps),partial(_index_to_pos_mapper,minx, maxx, miny, maxy, x_steps, y_steps)

def _get_pipe_mask(model,matrix,pos_mapper,resultion):
    for pipe in model.pipes:
        pipe_geom = pipe.geometry
        for i in matrix:
            for j in matrix:
                x,y = pos_mapper(i,j)
                if pipe_geom.distance(shapely.Point((x,y))) < resultion:
                    yield {(i,j):pipe.heat}




def get_heat_distribution(model: Model, config: Config,resolution:float) -> Model:
    """
    Distribute heat from the distributor to the rooms.
    """
    heat_matrix,index_mapper,pos_mapper = _get_init_heat_matrix(model, resolution)
    pipe_mask = dict(_get_pipe_mask(model.pipes,pos_mapper,resolution))

    
    