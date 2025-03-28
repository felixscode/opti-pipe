from abc import ABC, abstractmethod
from enum import Enum
from shapely.geometry import Point,Polygon,LineString
from shapely.plotting import plot_polygon,plot_points
from opti_pipe.utils import Config
from opti_pipe.heat_distribution import render_heat
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Component(ABC):
    def __init__(self,config):
        self.config = config

    @abstractmethod
    def _render(self):
        pass


class Floor(Component):
    def __init__(self,config,corners:tuple[tuple[int,int]]):
        super().__init__(config)
        self.corners = corners
        self.geometry = Polygon(corners)
    
    def _render(self):
        plot_polygon(self.geometry,color="lightgrey",edgecolor="black",add_points=False)


class NodeType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    CORNER = "corner"
    GRID = "grid"


class Node(Component):
    def __init__(self,config,x:int,y:int,node_type:NodeType):
        super().__init__(config)
        self.x = round(x,2)
        self.y = round(y,2)
        self.id = f"{self.x}_{self.y}_{node_type}"
        self.type = node_type
        self.geometry = Point(self.x,self.y)

    def set_heat(self,heat_output:float):
        if not self.type == NodeType.OUTPUT:
            raise ValueError('Only output nodes can have heat')
        self.heat = heat_output

    def is_neighbor(self,other,cell_size):
        if not any([self.x == other.x ,self.y == other.y]):
            return False
        if self.distance(other) > cell_size*1.5:
            return False
        return True
    
    def shares_parent(self,other):
        if self.x == other.x:
            return True
        if self.y == other.y:
            return True
        return False

    def distance(self,other):
        return self.geometry.distance(other.geometry)


    def _match_color(self):
        if hasattr(self,'heat'):
            return 'red'
        if self.type == NodeType.CORNER:
            return 'green'
        if self.type == NodeType.GRID:
            return 'grey'
        if self.type in (NodeType.INPUT,NodeType.OUTPUT):
            return 'blue'
        raise ValueError('Node type not supported')
    
    def _render(self):
        color =self._match_color()
        plot_points(self.geometry,color=color)
        

class Distributor(Component):
    def __init__(self,config:Config,nodes:tuple[Node],heat_per_node:float):
        super().__init__(config)
        self.heat_per_node = heat_per_node
        self.inlets = tuple(filter(lambda x: x.type == NodeType.INPUT, nodes))
        self.outlets = tuple(filter(lambda x: x.type == NodeType.OUTPUT, nodes))
        for node in self.outlets:
            node.set_heat(heat_per_node)
        self.geometry = self._get_geometry(self.inlets,self.outlets)
        if len(self.inlets) < 1:
            raise ValueError('Distributor must more than one input')
        if len(self.outlets) < 1:
            raise ValueError('Distributor must more than one output')
        if len(self.inlets) != len(self.outlets):
            raise ValueError('Distributor must have equal number of input and output')
    
    def _get_geometry(self,inlets,outlets):
        line = LineString([inlet.geometry for inlet in inlets] + [outlet.geometry for outlet in outlets])
        return line.buffer(self.config.distributor.buffer_radius)
    
    def iter_nodes(self):
        for inlet in self.inlets:
            yield inlet
        for outlet in self.outlets:
            yield outlet
    
    def _render(self):
        plot_polygon(self.geometry,color="purple",add_points=False,edgecolor="black")
        for inlet in self.inlets:
            inlet._render()
        for outlet in self.outlets:
            outlet._render()

class Pipe(Component):
    def __init__(self,config, input:Node, output:Node,corners:tuple[Node]):
        super().__init__(config)
        self.input = input
        self.output = output
        self.corners = corners
        self.geometry = self._get_geometry()
        self.heat = self.output.heat
  
    
    def _get_geometry(self):
        line = LineString([self.input.geometry] + [corner.geometry for corner in self.corners] + [self.output.geometry])
        return line.buffer(self.config.pipe.width)
        
    def _render(self):
        plot_polygon(self.geometry,add_points=False,color=self.config.pipe.color)
    
    @staticmethod
    def from_path(config,path):
        input_node = next(filter(lambda x: x.type == NodeType.INPUT,path))
        output_node = next(filter(lambda x: x.type == NodeType.OUTPUT,path))
        path.remove(input_node)
        path.remove(output_node)
        corners = path
        return Pipe(config,input_node,output_node,corners)


class RoomConnection(Component):
    def __init__(self,config, input:Node, output:Node,heat_loss:float):
        super().__init__(config)
        self.input = input
        self.output = output
        self.geometry = self._get_geometry()
        self.heat_loss = heat_loss
        

    def set_heat_output(self,heat_input:float):
        self.heat_output = heat_input - self.heat_loss
        self.output.set_heat(self.heat_output)

    def iter_nodes(self):
        yield self.input
        yield self.output

    def _get_geometry(self):
        line = LineString([self.input.geometry,self.output.geometry])
        return line.buffer(self.config.distributor.buffer_radius)
    
    def _render(self):
        plot_polygon(self.geometry,color="lightgreen",add_points=False,edgecolor="black")
        self.input._render()
        self.output._render()

class Model():
    def __init__(self,config,target_heat_input:float,floor:Floor,distributor:Distributor,room_connections:tuple[RoomConnection]):
        self.config = config
        self.target_heat_input = target_heat_input
        self.floor = floor
        self.distributor = distributor
        self.pipes = tuple()
        self.connectors = room_connections
        for connector in room_connections:
            connector.set_heat_output(distributor.heat_per_node)

    def add(self,component:Component):
        match component:
            case Pipe():
                self.pipes += (component,)
            case Floor():
                raise ValueError('Only one floor allowed')
            case Distributor(config):
                raise ValueError('Only one distributor allowed')
            case RoomConnection(config):
                self.connectors += (component,)
            case _:
                raise ValueError('Component not supported')

    def add_graph(self,cell_size:float):
        self.graph = Graph(self.config,self,cell_size)



    def render(self,figsize:tuple[int,int]= (10,10),show_graph:bool=False):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.axis('off')  # Turn off the axis
        plt.grid(visible=False)
        if len(self.pipes) > 0:
            render_heat(self,self.config,self.config.heat.resolution,self.floor.geometry)
        self.floor._render()
        if show_graph:
            if hasattr(self,'graph'):
                self.graph._render()
        for pipe in self.pipes:
            pipe._render()
        for connector in self.connectors:
            connector._render()
        self.distributor._render()
        # plt.show()




class Graph():
    def __init__(self,config,model,cell_size:int):
        self.config = config
        self.floor = model.floor
        self.connectors = (model.distributor, ) + model.connectors
        self.cell_size = cell_size
        self.nodes = self.make_grid()
        self.graph = self.build_graph()

    def iter_nodes(self):
        for node in self.nodes:
            yield node
        for connector in self.connectors:
            for node in connector.iter_nodes():
                yield node

    def make_grid(self):
        envelope = self.floor.geometry.envelope
        minx,miny,maxx,maxy = envelope.bounds
        grid = []
        n_x_nodes = int((maxx - minx) / self.cell_size)
        if n_x_nodes % 2 != 0:
            n_x_nodes += 1
        n_y_nodes = int((maxy - miny) / self.cell_size)
        if n_y_nodes % 2 != 0:
            n_y_nodes += 1
        for x in np.linspace(minx,maxx,n_x_nodes):
            for y in np.linspace(miny,maxy,n_y_nodes):
                if self.floor.geometry.contains(Point(x,y)):
                    grid.append(Node(self.config,x,y,NodeType.GRID))
        return tuple(grid)
    
    def nodes_as_dict(self,nodes):
        for node in nodes:
            yield (node.id,{"x":node.x,"y":node.y})
    
    def build_graph(self):
        g = nx.Graph()
        g.add_nodes_from(self.nodes_as_dict(self.nodes))
        for i,node in enumerate(self.nodes):
            for other_node in self.nodes[i:]:
                if node.id != other_node.id and node.is_neighbor(other_node,self.cell_size):
                    g.add_edge(node.id,other_node.id)
        for connector in self.connectors:
            nodes = sorted(list(connector.iter_nodes()),key=lambda n: (n.x,n.y))
            g.add_nodes_from(self.nodes_as_dict(nodes))
            nearest_grid_nodes = Utils.find_n_nearest_nodes(self.nodes,connector.geometry.centroid, len(nodes))
            nearest_grid_nodes = sorted(nearest_grid_nodes,key=lambda n: (n.x,n.y))
            for i in range(len(nearest_grid_nodes)):
                g.add_edge(nodes[i].id,nearest_grid_nodes[i].id)
            
        return g
    
    def _render(self):
        layout = {node.id: (node.x, node.y) for node in self.nodes}
        for connector in self.connectors:
            for node in connector.iter_nodes():
                layout[node.id] = (node.x, node.y)
        nx.draw(self.graph, pos=layout, node_size=20, node_color="grey")

    
class Utils():
    @staticmethod
    def find_n_nearest_nodes(nodes,point,n:int):
        return sorted(nodes, key=lambda x: point.distance(x.geometry))[:n]