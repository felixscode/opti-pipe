from abc import ABC, abstractmethod
from opti_pipe.models import NodeType,Pipe,Model
import networkx as nx
from functools import partial
import shapely
from opti_pipe.utils import Config

class Utils:
    @staticmethod
    def prune_visited_nodes(visited_nodes,graph: nx.Graph):
        # return graph
        for node in visited_nodes:
            if graph.has_node(node.id):
                graph.remove_node(node.id)
        return graph
    
    @staticmethod
    def _get_direction(node1, node2):
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return (0, 0)
        return (dx / length, dy / length)
    
    @staticmethod
    def _make_poly_point_and_direction(point,direction,end,width):
        x,y = point.x,point.y
        length = point.distance(end)
        dx,dy = direction[0]*length,direction[1]*length
        line = shapely.LineString([(x-dx,y-dy),(x+dx,y+dy)])
        return line.buffer(width/2,cap_style="square")
    
    @staticmethod
    def get_centerline(model:Model,grid_size):

        floor = model.floor
        distributor = model.distributor
        connectors = model.connectors
        pipe_width = grid_size


        _dir:tuple[float,float] = Utils._get_direction(distributor.inlets[0],distributor.outlets[0])
        _centroid:shapely.Point = distributor.geometry.centroid
        width0 = distributor.inlets[0].geometry.distance(distributor.inlets[1].geometry)
        width1 = distributor.outlets[0].geometry.distance(distributor.outlets[1].geometry)
        width = max(width0,width1) + 2*pipe_width
        # reverse the direction (flip 90 degrees) and setting point far away from the centroid
        polys = [Utils._make_poly_point_and_direction(_centroid,_dir[::-1],shapely.Point(1000,1000),width)]
        for elem in connectors:
            _dir:tuple[float,float]  = Utils._get_direction(elem.input,elem.output)
            _centroid:shapely.Point = elem.geometry.centroid
            width = elem.input.geometry.distance(elem.output.geometry) + 2*pipe_width
            polys.append(Utils._make_poly_point_and_direction(_centroid,_dir[::-1],polys[0],width))
        target_area = shapely.unary_union(polys)
        return floor.geometry.intersection(target_area)

class Router(ABC):
    def __init__(self,config,model):
        self.config = config
        self.model = model
        self.node_pairs = self.find_node_pairs()

    
    def _find(self,nodes):
        if len(nodes) == 2:
            return (tuple(nodes),)
        next_node = nodes.pop()
        _avalibe_nodes = [node for node in nodes if not node.shares_parent(next_node)]
        closest_node = min(_avalibe_nodes,key=lambda x: next_node.geometry.distance(x.geometry))
        nodes.remove(closest_node)
        return ((next_node,closest_node),) + self._find(nodes)
    
    def find_node_pairs(self):
        nodes = tuple(self.model.graph.iter_nodes())
        input_nodes = [node for node in nodes if node.type == NodeType.INPUT]
        output_nodes = [node for node in nodes if node.type == NodeType.OUTPUT]
        input_node_pairs = self._find(input_nodes)
        output_node_pairs = self._find(output_nodes)
        return output_node_pairs + input_node_pairs 

    @abstractmethod
    def route(self):
        pass


class NaiveRouter(Router):
    def __init__(self,config,model,grid_size):
        super().__init__(config,model)
        self.grid_size = grid_size
    

    @staticmethod
    def weight_func(center_line:shapely.MultiLineString,node1,node2,_):

        n1x,n1y = float(node1.split("_")[0]),float(node1.split("_")[1])
        n2x,n2y = float(node2.split("_")[0]),float(node2.split("_")[1])
        nx = (float(n1x) + float(n2x))/2
        ny = (float(n1y) + float(n2y))/2
        dx = abs(n1x-n2x)
        dy = abs(n1y-n2y)

        return center_line.distance(shapely.Point(nx,ny))
    
    def route(self) -> Model:
        nodes = tuple(self.model.graph.iter_nodes())
        centerline = Utils.get_centerline(self.model,self.grid_size)
        w_func = partial(self.weight_func,centerline)
        _graph = self.model.graph.graph.copy()
        for input_node,output_node in self.node_pairs:
            try:
                path = nx.shortest_path(_graph,input_node.id,output_node.id,weight=w_func)
            except Exception:
                continue
            path_nodes = [next(filter(lambda x: x.id == node_id,nodes),None) for node_id in path]
            if None in path_nodes:
                raise ValueError("Node not found in graph")
            pipe = Pipe.from_path(self.config,path_nodes)
            self.model.add(pipe)
            _graph = Utils.prune_visited_nodes(path_nodes,_graph)
        return self.model


class TsmRouter(Router):
    def __init__(self,config,model,grid_size):
        super().__init__(config,model)
        self.grid_size = grid_size
