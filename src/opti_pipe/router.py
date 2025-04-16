from abc import ABC, abstractmethod
from opti_pipe.models import NodeType, Pipe, Model, Node
import networkx as nx
from functools import partial
import shapely
from opti_pipe.utils import Config
import time
from shapely.geometry import MultiPoint


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


class DFSSolver():

    @staticmethod
    def get_weight(x,visited_nodes,direction_bias):
        x0,y0 = visited_nodes[-1].split("_")[:2]
        x1,y1 = x.split("_")[:2]
        _dir = (abs(float(x1)-float(x0)),abs(float(y1)-float(y0)))
        match direction_bias, round(_dir[0],1), round(_dir[1],1):
            case "h", 0.0, _:
                return 1
            case "h", _, 0.0:
                return 0
            case "h", _, _:
                return 0.0
            case "v", 0.0, _:
                return 0
            case "v", _, 0.0:
                return 1
            case "v", _, _:
                return 0

    @staticmethod
    def prune(possible_next, visited_nodes,graph):
        """
        Prune the list of possible next nodes based on the visited nodes.
        This function can be customized to implement specific pruning logic.
        """
        possible_next =  [n for n in possible_next if n not in visited_nodes] # remove visited nodes
        return possible_next

    @staticmethod
    def _get_solver_gen(graph, visited, goal_node, direction_bias="v"):
        """
        A DFS/backtracking generator that yields all possible solutions from the
        current 'visited' path (a tuple) ending at visited[-1] to the goal_node.
        
        Parameters:
        graph: a directed graph object with method neighbors(node) that returns neighbors
        visited: a tuple of nodes representing the current path (e.g. (start_node,))
        goal_node: the target node that we're trying to reach
        direction_bias: (optional) a parameter to bias neighbor ordering via get_weight

        Yields:
        Each valid complete path (as a tuple) from the start to the goal_node.
        """
        # Base case: if current node is the goal, yield the current path.
        if visited[-1] == goal_node:
            yield visited
            return

        # Get candidate neighbors that haven't been visited in this path.

        possible_next = [n for n in graph.neighbors(visited[-1]) if n not in visited]
        
        # If there are no candidates, then this branch cannot extend; backtrack.
        if not possible_next:
            return

        # Sort the candidate nodes using a heuristic; this bias may help order paths.
        weighted = sorted(possible_next, key=lambda x: DFSSolver.get_weight(x, visited, direction_bias))

        
        # For each candidate, extend the visited path and recursively yield from the deeper call.
        for node in weighted:
            new_path = visited + (node,)  # Create an extended path
            # Use 'yield from' to yield every solution from the recursive call.
            yield from DFSSolver._get_solver_gen(graph, new_path, goal_node, direction_bias)

    @staticmethod
    def solve(graph,start_node, goal_node,time_budget=1):
        v_solver = DFSSolver._get_solver_gen(graph, (start_node,), goal_node, direction_bias="v")
        h_solver = DFSSolver._get_solver_gen(graph, (start_node,), goal_node, direction_bias="h")
        solutions = []
        start_time = time.time()
        while time.time() - start_time < time_budget:
            try:
                v_solution = next(v_solver)
                h_solution = next(h_solver)
                solutions.append(v_solution)
                solutions.append(h_solution)
            except StopIteration:
                break
        if len(solutions) == 0:
            raise ValueError("No solution found consider increasing time budget and make sure the graph is connected")
        return sorted(solutions, key=lambda x: len(x),reverse=True)[0]  # sort by length of the path (we want the longest path)


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
        self.model.graph.graph = _graph
        return self.model


class HeuristicRouter(Router):
    def __init__(self, config, model, grid_size):
        super().__init__(config, model)
        self.grid_size = grid_size
        self._naive_router = NaiveRouter(config, model, grid_size)
        self.model = self._naive_router.route()


    def _get_multi_point_from_graph(self,graph,nodes):
        return MultiPoint([node.geometry for node in nodes if node.id in graph])
        
    def _find_best_pipe(self,subgraph,pipes,nodes,grid_size):
        shapely_points = self._get_multi_point_from_graph(subgraph,nodes)
        potential_pipes = []
        for pipe in pipes:
            if pipe.geometry.distance(shapely_points) < grid_size*1.1:
                potential_pipes.append(pipe)
        if not potential_pipes:
            raise ValueError("No potential pipes found")
        best_pipe = max(potential_pipes,key=lambda x: x.heat)
        return best_pipe
    
    def _build_trav_graph(self,grid_size,pipe,subgraph):
        search_node_ids = tuple(subgraph | set(n.id for n in pipe.corners))
        _nodes = [n for n in self.model.graph.iter_nodes() if n.id in search_node_ids] + [pipe.input, pipe.output]
        g = nx.DiGraph()
        g.add_nodes_from(self.model.graph.nodes_as_dict(_nodes))
        for i, node in enumerate(_nodes):
            for other_node in _nodes[:]:
                if node.id != other_node.id and node.is_neighbor(other_node, grid_size):
                    g.add_edge(node.id, other_node.id)
        g.add_nodes_from(self.model.graph.nodes_as_dict([pipe.input,pipe.output]))
        g.add_edge(pipe.input.id, pipe.corners[0].id)
        g.add_edge( pipe.corners[-1].id,pipe.output.id)
        return g
    
    def route(self):
        subgraphs = tuple(nx.connected_components(self.model.graph.graph))
        for subgraph in subgraphs:
            pipe = self._find_best_pipe(subgraph,self.model.pipes,tuple(self.model.graph.iter_nodes()),self.grid_size)
            self.model.pipes = tuple(p for p in self.model.pipes if p != pipe)
            trav_graph = self._build_trav_graph(self.grid_size,pipe,subgraph)
            solution = DFSSolver.solve(trav_graph,pipe.input.id,pipe.output.id,time_budget=1)
            nodes = tuple(self.model.graph.nodes) + tuple([pipe.input, pipe.output]) + tuple(pipe.corners)
            path_nodes = [next(filter(lambda x: x.id == node_id, nodes), None) for node_id in solution]
            pipe = Pipe.from_path(self.config, path_nodes, self.model.distributor)
            self.model.add(pipe)
        return self.model



