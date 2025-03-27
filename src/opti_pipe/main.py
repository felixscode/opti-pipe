from opti_pipe.models import *
from opti_pipe.utils import Config,load_config
import random


def _get_random_floor_corners():
    width,lenght = 1,10
    while width / lenght < 0.3:
        x = random.uniform(5,20)
        y = random.uniform(5,20)
        width,lenght = min([x,y]),max([x,y])
    return ((0,0),(width,0),(width,lenght),(0,lenght))

def make_random_init_model(config:Config) -> Model:


    floor = Floor(config,_get_random_floor_corners())
    distributor = Distributor(config)
    Model(config,10,)