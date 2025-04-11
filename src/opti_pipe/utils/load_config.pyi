from dataclasses import dataclass
from datetime import datetime
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Heat:
    resolution: float
    conv_iterations: int
    conv_kernel_size: int


@dataclass(frozen=True)
class Config:
    description: str
    pipe: "Pipe"
    floor: "Floor"
    inlet: "Inlet"
    outlet: "Outlet"
    distributor: "Distributor"
    connector: "Connector"
    node: "Node"
    heat: "Heat"


@dataclass(frozen=True)
class Node:
    fill_color: str
    edge_color: str
    edge_width: int
    buffer_radius: float


@dataclass(frozen=True)
class Inlet:
    fill_color: str
    edge_color: str
    edge_width: int


@dataclass(frozen=True)
class Distributor:
    fill_color: str
    edge_color: str
    edge_width: int
    buffer_radius: float


@dataclass(frozen=True)
class Floor:
    fill_color: str
    edge_color: str
    edge_width: int


@dataclass(frozen=True)
class Pipe:
    color: str
    width: float


@dataclass(frozen=True)
class Connector:
    fill_color: str
    edge_color: str
    edge_width: int
    buffer_radius: float


@dataclass(frozen=True)
class Outlet:
    fill_color: str
    edge_color: str
    edge_width: int


def load_config(config_path: str) -> Config: ...
