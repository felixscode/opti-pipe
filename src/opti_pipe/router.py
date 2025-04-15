from abc import ABC, abstractmethod
from opti_pipe.models import NodeType, Pipe, Model, Node
import networkx as nx
from functools import partial
import shapely
from opti_pipe.utils import Config


class Utils:
    @staticmethod
    def prune_visited_nodes(visited_nodes, graph: nx.Graph):
        # return graph
        for node in visited_nodes:
            if graph.has_node(node.id):
                graph.remove_node(node.id)
        return graph

    @staticmethod
    def _get_direction(node1, node2):
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return (0, 0)
        return (dx / length, dy / length)

    @staticmethod
    def _make_line_point_and_direction(point, direction, length):
        x, y = point.x, point.y

        dx, dy = direction[0] * length, direction[1] * length
        return shapely.LineString([(x - dx, y - dy), (x + dx, y + dy)])

    @staticmethod
    def _get_width_from_geometry(geom):
        _envelope = geom.envelope
        minx, miny, maxx, maxy = _envelope.bounds
        x_range = maxx - minx
        y_range = maxy - miny
        return max(x_range, y_range)

    @staticmethod
    def _get_connector_node(model: Model, input_node: Node, output_node: Node):
        for connector in model.connectors:
            for conn_node in connector.iter_nodes():
                if conn_node in [input_node, output_node]:
                    return conn_node, connector
        raise ValueError("No connector node found for input and output nodes")

    @staticmethod
    def get_centerline(model: Model, input_node: Node, output_node: Node):

        distributor_node = next(filter(lambda x: x in model.distributor.iter_nodes(), (input_node, output_node)), None)

        if distributor_node is None:
            raise ValueError("Distributor node not found in input or output nodes")
        _distributor_node = next(iter(model.graph.graph[distributor_node.id]), distributor_node.id)
        distributor_node = next(iter(filter(lambda x: x.id == _distributor_node, model.graph.nodes)), None)
        if distributor_node is None:
            raise ValueError("Distributor node not found in graph")
        _dist_dir: tuple[float, float] = Utils._get_direction(model.distributor.inlets[0], model.distributor.outlets[0])
        dist_line = Utils._make_line_point_and_direction(
            distributor_node.geometry,
            _dist_dir[::-1],
            model.floor.geometry.envelope.boundary.length,
        )

        conn_node, connector = Utils._get_connector_node(model, input_node, output_node)
        _conn_node = next(iter(model.graph.graph[conn_node.id]), conn_node)
        conn_node = next(filter(lambda x: x.id == _conn_node, model.graph.nodes), None)
        _conn_dir: tuple[float, float] = Utils._get_direction(connector.input, connector.output)
        conn_line = Utils._make_line_point_and_direction(
            conn_node.geometry, _conn_dir[::-1], model.floor.geometry.envelope.boundary.length
        )

        multi_line = shapely.MultiLineString([dist_line, conn_line])
        return model.floor.geometry.intersection(multi_line)


class Router(ABC):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        _center = self.model.distributor.geometry.centroid
        self.node_pairs = sorted(
            tuple(self.find_node_pairs()), key=lambda x: x[0].geometry.distance(_center)
        )  # sort by distance to distributor center to process nodes to the inner first

    def find_node_pairs(self):
        room_inputs = [con.input for con in self.model.connectors]
        room_outputs = [con.output for con in self.model.connectors]
        distributor_inputs = list(self.model.distributor.inlets)
        distributor_outputs = list(self.model.distributor.outlets)
        if len(distributor_inputs) != len(room_inputs) or len(distributor_outputs) != len(room_outputs):
            raise ValueError("Mismatch between distributor and room nodes")
        for ri in room_inputs:
            closest_node = min(distributor_inputs, key=lambda x: ri.distance(x))
            distributor_inputs.remove(closest_node)
            yield closest_node, ri
        for ro in room_outputs:
            closest_node = min(distributor_outputs, key=lambda x: ro.distance(x))
            distributor_outputs.remove(closest_node)
            yield closest_node, ro

    @abstractmethod
    def route(self):
        pass


class NaiveRouter(Router):
    def __init__(self, config, model, grid_size):
        super().__init__(config, model)
        self.grid_size = grid_size

    @staticmethod
    def weight_func(center_line: shapely.MultiLineString, node1, node2, _):
        n1x, n1y = float(node1.split("_")[0]), float(node1.split("_")[1])
        n2x, n2y = float(node2.split("_")[0]), float(node2.split("_")[1])
        nx = (float(n1x) + float(n2x)) / 2
        ny = (float(n1y) + float(n2y)) / 2
        # dx = abs(n1x - n2x)
        # dy = abs(n1y - n2y)
        # bias = int(not 0.0 in (dx, dy)) * 100
        # if bias == 100:
        #     print(f"Bias applied between {node1} and {node2}")
        return center_line.distance(shapely.Point(nx, ny))

    def route(self) -> Model:
        nodes = tuple(self.model.graph.iter_nodes())
        _graph = self.model.graph.graph.copy()

        for input_node, output_node in self.node_pairs:
            centerline = Utils.get_centerline(self.model, input_node, output_node)
            w_func = partial(self.weight_func, centerline)
            try:
                path = nx.shortest_path(_graph, input_node.id, output_node.id, weight=w_func)
            except Exception:
                print(f"Path not found between {input_node.id} and {output_node.id}")

            path_nodes = [next(filter(lambda x: x.id == node_id, nodes), None) for node_id in path]
            if None in path_nodes:
                raise ValueError("Node not found in graph")
            pipe = Pipe.from_path(self.config, path_nodes, self.model.distributor)
            self.model.add(pipe)
            _graph = Utils.prune_visited_nodes(path_nodes + [input_node, output_node], _graph)
        return self.model


class OptiRouter(Router):
    def __init__(self, config, model, grid_size):
        super().__init__(config, model)
        self.grid_size = grid_size

    def route(self):
        nodes = tuple(self.model.graph.iter_nodes())
