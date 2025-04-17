from abc import ABC, abstractmethod
from opti_pipe.models import NodeType, Pipe, Model, Node
import networkx as nx
from functools import partial
import shapely
from opti_pipe.utils import Config
import time
from shapely.geometry import MultiPoint
from typing import List, Tuple, Optional, Iterable, Any, Union


class Utils:
    @staticmethod
    def prune_visited_nodes(visited_nodes: List[Node], graph: nx.Graph) -> nx.Graph:
        """
        Remove visited nodes from the graph to prevent them from being used in future paths.
        
        Args:
            visited_nodes: List of nodes that have been visited and should be removed
            graph: The graph to prune nodes from
            
        Returns:
            The pruned graph with visited nodes removed
        """
        for node in visited_nodes:
            if graph.has_node(node.id):
                graph.remove_node(node.id)
        return graph

    @staticmethod
    def _get_direction(node1: Node, node2: Node) -> Tuple[float, float]:
        """
        Calculate the normalized direction vector from node1 to node2.
        
        Args:
            node1: Source node
            node2: Target node
            
        Returns:
            A tuple of (dx, dy) representing the normalized direction vector
        """
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return (0, 0)
        return (dx / length, dy / length)

    @staticmethod
    def _make_line_point_and_direction(
        point: shapely.Point, 
        direction: Tuple[float, float], 
        length: float
    ) -> shapely.LineString:
        """
        Create a line from a point extending in both directions based on a direction vector.
        
        Args:
            point: The center point
            direction: The direction vector (normalized)
            length: The total length of the line
            
        Returns:
            A LineString geometry extending from the point in both directions
        """
        x, y = point.x, point.y
        dx, dy = direction[0] * length, direction[1] * length
        return shapely.LineString([(x - dx, y - dy), (x + dx, y + dy)])

    @staticmethod
    def _get_width_from_geometry(geom: shapely.Geometry) -> float:
        """
        Get the maximum width of a geometry by calculating its envelope dimensions.
        
        Args:
            geom: The geometry to measure
            
        Returns:
            The maximum dimension (width or height) of the geometry's envelope
        """
        _envelope = geom.envelope
        minx, miny, maxx, maxy = _envelope.bounds
        x_range = maxx - minx
        y_range = maxy - miny
        return max(x_range, y_range)

    @staticmethod
    def _get_connector_node(
        model: Model, 
        input_node: Node, 
        output_node: Node
    ) -> Tuple[Node, Any]:
        """
        Find the connector node and its parent connector that matches one of the given nodes.
        
        Args:
            model: The model containing connectors
            input_node: The input node to check
            output_node: The output node to check
            
        Returns:
            A tuple of (connector_node, connector) where connector_node is either 
            input_node or output_node and connector is its parent connector
            
        Raises:
            ValueError: If no connector node is found that matches input_node or output_node
        """
        for connector in model.connectors:
            for conn_node in connector.iter_nodes():
                if conn_node in [input_node, output_node]:
                    return conn_node, connector
        raise ValueError("No connector node found for input and output nodes")

    @staticmethod
    def get_centerline(
        model: Model, 
        input_node: Node, 
        output_node: Node
    ) -> shapely.Geometry:
        """
        Calculate the centerline between a distributor node and a connector node.
        
        This method creates two lines:
        1. A line through the distributor in the direction perpendicular to the distributor's orientation
        2. A line through the connector in the direction perpendicular to the connector's orientation
        
        It then finds the intersection of these lines with the floor area to create a path guide.
        
        Args:
            model: The model containing the floor, distributor and connectors
            input_node: One of the nodes to connect (typically from distributor)
            output_node: The other node to connect (typically from a room connector)
            
        Returns:
            A MultiLineString or other geometry representing the centerline path guide
            
        Raises:
            ValueError: If distributor node cannot be found or linked to the graph
        """
        # Find which node belongs to the distributor
        distributor_node = next(filter(lambda x: x in model.distributor.iter_nodes(), 
                                      (input_node, output_node)), None)

        if distributor_node is None:
            raise ValueError("Distributor node not found in input or output nodes")
            
        # Get a neighbor of the distributor node in the graph to use as the starting point
        _distributor_node = next(iter(model.graph.graph[distributor_node.id]), distributor_node.id)
        distributor_node = next(iter(filter(lambda x: x.id == _distributor_node, model.graph.nodes)), None)
        
        if distributor_node is None:
            raise ValueError("Distributor node not found in graph")
            
        # Create a line perpendicular to distributor orientation
        _dist_dir: Tuple[float, float] = Utils._get_direction(model.distributor.inlets[0], 
                                                             model.distributor.outlets[0])
        # Reverse direction to get perpendicular line (swap x,y and make one negative)
        dist_line = Utils._make_line_point_and_direction(
            distributor_node.geometry,
            _dist_dir[::-1],  # Perpendicular direction by swapping components
            model.floor.geometry.envelope.boundary.length,  # Make line long enough to cross floor
        )

        # Find the connector node and its parent
        conn_node, connector = Utils._get_connector_node(model, input_node, output_node)
        
        # Get a neighbor node of the connector in the graph
        _conn_node = next(iter(model.graph.graph[conn_node.id]), conn_node)
        conn_node = next(filter(lambda x: x.id == _conn_node, model.graph.nodes), None)
        
        # Create a line perpendicular to connector orientation
        _conn_dir: Tuple[float, float] = Utils._get_direction(connector.input, connector.output)
        conn_line = Utils._make_line_point_and_direction(
            conn_node.geometry, 
            _conn_dir[::-1],  # Perpendicular direction
            model.floor.geometry.envelope.boundary.length
        )

        # Create a multi-line string and find its intersection with the floor
        multi_line = shapely.MultiLineString([dist_line, conn_line])
        return model.floor.geometry.intersection(multi_line)


class DFSSolver():
    """
    A depth-first search solver that finds paths between nodes in a graph with direction biasing.
    
    This solver is used to find paths that maximize length while respecting directional preferences,
    which helps create more efficient pipe routing layouts.
    """

    @staticmethod
    def get_weight(x: str, visited_nodes: Tuple[str, ...], direction_bias: str) -> float:
        """
        Calculate a weight for a node based on directional bias preferences.
        
        This function determines which nodes to prioritize during path finding based on 
        whether we prefer horizontal ("h") or vertical ("v") movements.
        
        Args:
            x: The node ID being evaluated
            visited_nodes: Tuple of previously visited node IDs
            direction_bias: Either "h" for horizontal bias or "v" for vertical bias
            
        Returns:
            A weight value (lower values are preferred during sorting):
            - For horizontal bias: prefer horizontal movements (x changes, y constant)
            - For vertical bias: prefer vertical movements (y changes, x constant)
        """
        # Extract coordinates from node IDs 
        x0, y0 = visited_nodes[-1].split("_")[:2]  # Last visited node coordinates
        x1, y1 = x.split("_")[:2]                  # Candidate node coordinates
        
        # Calculate direction vector (using absolute values since we only care about axis changes)
        _dir = (abs(float(x1) - float(x0)), abs(float(y1) - float(y0)))
        
        # Determine weight based on direction bias and movement direction
        # Lower weights (0) are preferred over higher weights (1)
        match direction_bias, round(_dir[0], 1), round(_dir[1], 1):
            case "h", 0.0, _:  # Vertical move with horizontal bias: less preferred
                return 1
            case "h", _, 0.0:  # Horizontal move with horizontal bias: preferred
                return 0
            case "h", _, _:    # Diagonal move with horizontal bias: least preferred
                return 0.0
            case "v", 0.0, _:  # Vertical move with vertical bias: preferred
                return 0
            case "v", _, 0.0:  # Horizontal move with vertical bias: less preferred
                return 1
            case "v", _, _:    # Diagonal move with vertical bias: least preferred
                return 0

    @staticmethod
    def prune(possible_next: List[str], visited_nodes: List[str], graph: nx.Graph) -> List[str]:
        """
        Prune the list of possible next nodes based on the visited nodes.
        
        This filters out already visited nodes to prevent cycles in the path.
        
        Args:
            possible_next: List of candidate next node IDs
            visited_nodes: List of already visited node IDs
            graph: The graph being traversed
            
        Returns:
            Filtered list of nodes that haven't been visited yet
        """
        return [n for n in possible_next if n not in visited_nodes]

    @staticmethod
    def _get_solver_gen(
        graph: nx.Graph, 
        visited: Tuple[str, ...], 
        goal_node: str, 
        direction_bias: str = "v"
    ) -> Iterable[Tuple[str, ...]]:
        """
        A DFS/backtracking generator that yields all possible solutions from the
        current 'visited' path (a tuple) ending at visited[-1] to the goal_node.
        
        Args:
            graph: A directed graph object with method neighbors(node) that returns neighbors
            visited: A tuple of node IDs representing the current path (e.g. (start_node,))
            goal_node: The target node ID that we're trying to reach
            direction_bias: A parameter to bias neighbor ordering ("h" for horizontal, "v" for vertical)
        
        Yields:
            Each valid complete path (as a tuple) from the start to the goal_node
        """
        # Base case: if current node is the goal, yield the current path
        if visited[-1] == goal_node:
            yield visited
            return

        # Get candidate neighbors that haven't been visited in this path
        possible_next = [n for n in graph.neighbors(visited[-1]) if n not in visited]
        
        # If there are no candidates, then this branch cannot extend; backtrack
        if not possible_next:
            return

        # Sort the candidate nodes using the direction bias heuristic
        weighted = sorted(possible_next, key=lambda x: DFSSolver.get_weight(x, visited, direction_bias))
        
        # For each candidate, extend the visited path and recursively yield from the deeper call
        for node in weighted:
            new_path = visited + (node,)  # Create an extended path
            # Use 'yield from' to yield every solution from the recursive call
            yield from DFSSolver._get_solver_gen(graph, new_path, goal_node, direction_bias)

    @staticmethod
    def solve(
        graph: nx.Graph,
        start_node: str, 
        goal_node: str,
        time_budget: float = 1
    ) -> Tuple[str, ...]:
        """
        Find the longest path between start_node and goal_node within the given time budget.
        
        This method runs two parallel DFS searches with different directional biases:
        - Vertical bias: prefers paths that change y-coordinates
        - Horizontal bias: prefers paths that change x-coordinates
        
        It collects solutions from both approaches within the time limit and returns
        the longest path found, which gives us more pipe coverage.
        
        Args:
            graph: The graph to search
            start_node: The starting node ID
            goal_node: The target node ID
            time_budget: Maximum time in seconds to spend searching
            
        Returns:
            The longest valid path (as a tuple of node IDs) from start_node to goal_node
            
        Raises:
            ValueError: If no path is found within the time budget
        """
        # Create generators for both vertical and horizontal biased searches
        v_solver = DFSSolver._get_solver_gen(graph, (start_node,), goal_node, direction_bias="v")
        h_solver = DFSSolver._get_solver_gen(graph, (start_node,), goal_node, direction_bias="h")
        
        solutions = []
        start_time = time.time()
        
        # Collect solutions until time budget is exhausted
        while time.time() - start_time < time_budget:
            try:
                # Get next solution from each solver
                v_solution = next(v_solver)
                h_solution = next(h_solver)
                solutions.append(v_solution)
                solutions.append(h_solution)
            except StopIteration:
                # One or both generators are exhausted
                break
                
        if len(solutions) == 0:
            raise ValueError("No solution found; consider increasing time budget and make sure the graph is connected")
            
        # Return the longest path found (we want maximum pipe coverage)
        return sorted(solutions, key=lambda x: len(x), reverse=True)[0]


class Router(ABC):
    """
    Abstract base class for pipe routing algorithms.
    
    A Router takes a model with a floor, distributor, and room connections
    and calculates optimal pipe paths to connect them based on specific routing strategies.
    """
    
    def __init__(self, config: Config, model: Model):
        """
        Initialize the router with configuration and model.
        
        Args:
            config: Configuration settings
            model: The model containing floor, distributor, and room connections
        """
        self.config = config
        self.model = model
        _center = self.model.distributor.geometry.centroid
        # Sort node pairs by distance to distributor center to process inner nodes first
        self.node_pairs = sorted(
            tuple(self.find_node_pairs()), 
            key=lambda x: x[0].geometry.distance(_center)
        )

    def find_node_pairs(self) -> Iterable[Tuple[Node, Node]]:
        """
        Find matching pairs of nodes between the distributor and room connections.
        
        This method finds the optimal pairing between:
        - Distributor input nodes and room input nodes
        - Distributor output nodes and room output nodes
        
        The pairing is based on minimizing the distance between nodes.
        
        Returns:
            An iterator of (distributor_node, room_node) pairs to be connected
            
        Raises:
            ValueError: If there's a mismatch between distributor and room node counts
        """
        # Collect all input and output nodes
        room_inputs = [con.input for con in self.model.connectors]
        room_outputs = [con.output for con in self.model.connectors]
        distributor_inputs = list(self.model.distributor.inlets)
        distributor_outputs = list(self.model.distributor.outlets)
        
        # Verify matching counts
        if len(distributor_inputs) != len(room_inputs) or len(distributor_outputs) != len(room_outputs):
            raise ValueError("Mismatch between distributor and room nodes")
            
        # Match inputs: for each room input, find closest distributor input
        for ri in room_inputs:
            closest_node = min(distributor_inputs, key=lambda x: ri.distance(x))
            distributor_inputs.remove(closest_node)  # Remove to avoid reusing
            yield closest_node, ri
            
        # Match outputs: for each room output, find closest distributor output
        for ro in room_outputs:
            closest_node = min(distributor_outputs, key=lambda x: ro.distance(x))
            distributor_outputs.remove(closest_node)  # Remove to avoid reusing
            yield closest_node, ro

    @abstractmethod
    def route(self) -> Model:
        """
        Calculate and add pipe paths to the model.
        
        This method must be implemented by concrete router classes to define
        their specific pipe routing strategies.
        
        Returns:
            The updated model with pipes added
        """
        pass


class NaiveRouter(Router):
    """
    A straightforward pipe routing algorithm that uses Dijkstra's shortest path with centerline guidance.
    
    This router finds paths that follow the centerline between the distributor and room connections,
    creating relatively direct paths between node pairs. It uses distance from the centerline
    as the weight function for path finding.
    """
    
    def __init__(self, config: Config, model: Model, grid_size: float):
        """
        Initialize the NaiveRouter.
        
        Args:
            config: Configuration settings
            model: The model to route pipes through
            grid_size: The size of grid cells used for graph discretization
        """
        super().__init__(config, model)
        self.grid_size = grid_size

    @staticmethod
    def weight_func(
        center_line: shapely.MultiLineString, 
        node1: str, 
        node2: str, 
        _: Any
    ) -> float:
        """
        Calculate the edge weight based on distance from centerline.
        
        This weight function prioritizes edges that are closer to the centerline
        between distributor and room connection.
        
        Args:
            center_line: A MultiLineString representing the ideal path centerline
            node1: Source node ID
            node2: Target node ID
            _: Unused parameter (required by networkx interface)
            
        Returns:
            The weight value based on the midpoint distance to the centerline
        """
        # Extract coordinates from node IDs
        n1x, n1y = float(node1.split("_")[0]), float(node1.split("_")[1])
        n2x, n2y = float(node2.split("_")[0]), float(node2.split("_")[1])
        
        # Calculate the midpoint of the edge
        nx = (n1x + n2x) / 2
        ny = (n1y + n2y) / 2
        
        # Weight is the distance from edge midpoint to centerline
        # Edges closer to centerline get lower weights and are preferred
        return center_line.distance(shapely.Point(nx, ny))

    def route(self) -> Model:
        """
        Route pipes using the naive centerline-guided approach.
        
        This method:
        1. Creates a centerline between each node pair
        2. Uses Dijkstra's shortest path with a centerline distance weight function
        3. Converts the path to a pipe and adds it to the model
        4. Prunes used nodes from the graph to prevent overlap
        
        Returns:
            The updated model with pipes added
        """
        # Get all nodes from the graph
        nodes = tuple(self.model.graph.iter_nodes())
        # Create a copy of the graph to modify
        _graph = self.model.graph.graph.copy()

        # Process each pair of nodes to connect
        for input_node, output_node in self.node_pairs:
            # Get centerline to guide the path
            centerline = Utils.get_centerline(self.model, input_node, output_node)
            # Create a partial function with the centerline for path weight calculation
            w_func = partial(self.weight_func, centerline)
            
            try:
                # Find shortest path using Dijkstra's algorithm with centerline weights
                path = nx.shortest_path(_graph, input_node.id, output_node.id, weight=w_func)
            except Exception:
                print(f"Path not found between {input_node.id} and {output_node.id}")
                continue

            # Convert path node IDs to actual Node objects
            path_nodes = [next(filter(lambda x: x.id == node_id, nodes), None) for node_id in path]
            if None in path_nodes:
                raise ValueError("Node not found in graph")
                
            # Create a pipe from the path and add it to the model
            pipe = Pipe.from_path(self.config, path_nodes, self.model.distributor)
            self.model.add(pipe)
            
            # Remove used nodes from the graph to prevent overlap in future paths
            _graph = Utils.prune_visited_nodes(path_nodes + [input_node, output_node], _graph)
            
        # Update the model's graph with the pruned graph
        self.model.graph.graph = _graph
        return self.model


class HeuristicRouter(Router):
    """
    An advanced router that improves upon the NaiveRouter by maximizing pipe coverage.
    
    This router first creates initial paths using the NaiveRouter, then identifies
    isolated subgraphs (areas not covered by pipes) and attempts to reroute existing
    pipes through these areas to maximize floor coverage, using a DFS-based approach.
    """
    
    def __init__(self, config: Config, model: Model, grid_size: float,time_budget):
        """
        Initialize the HeuristicRouter.
        
        Args:
            config: Configuration settings
            model: The model to route pipes through
            grid_size: The size of grid cells used for graph discretization
        """
        super().__init__(config, model)
        self.grid_size = grid_size
        self.time_budget = time_budget
        # First create basic routes using NaiveRouter as a starting point
        self._naive_router = NaiveRouter(config, model, grid_size)
        self.model = self._naive_router.route()

    def _get_multi_point_from_graph(self, graph: set, nodes: Tuple[Node, ...]) -> MultiPoint:
        """
        Create a MultiPoint geometry from nodes that exist in the graph.
        
        Args:
            graph: A set of node IDs in the subgraph
            nodes: Tuple of Node objects to filter
            
        Returns:
            A MultiPoint geometry containing points from nodes that exist in the graph
        """
        return MultiPoint([node.geometry for node in nodes if node.id in graph])
        
    def _find_best_pipe(self, 
                       subgraph: set, 
                       pipes: Tuple[Pipe, ...], 
                       nodes: Tuple[Node, ...], 
                       grid_size: float) -> List[Pipe]:
        """
        Find pipes that are close to a subgraph and could be rerouted through it.
        
        This method identifies pipes that are within a certain distance of the
        subgraph nodes and sorts them by heat value (prioritizing hotter pipes).
        
        Args:
            subgraph: Set of node IDs in the isolated subgraph
            pipes: Tuple of existing pipes in the model
            nodes: All nodes in the graph
            grid_size: The size of grid cells used for graph discretization
            
        Returns:
            List of pipes sorted by heat value (highest first) that could be rerouted
            
        Raises:
            ValueError: If no potential pipes are found
        """
        # Create a MultiPoint geometry from subgraph nodes
        shapely_points = self._get_multi_point_from_graph(subgraph, nodes)
        potential_pipes = []
        
        # Find pipes that are close to the subgraph
        for pipe in pipes:
            # Use slightly larger threshold than grid_size to include nearby pipes
            if pipe.geometry.distance(shapely_points) < grid_size * 1.1:
                potential_pipes.append(pipe)
                
        if not potential_pipes:
            raise ValueError("No potential pipes found")
            
        # Sort pipes by heat value (higher heat is preferred for rerouting)
        best_pipes = sorted(potential_pipes, key=lambda x: x.heat, reverse=True)
        return best_pipes
    
    def _build_trav_graph(self, 
                         grid_size: float, 
                         pipe: Pipe, 
                         subgraph: set) -> nx.DiGraph:
        """
        Build a traversal graph for pipe rerouting through a subgraph.
        
        This creates a directed graph connecting the pipe's input and output nodes
        through the subgraph, allowing for DFS path-finding.
        
        Args:
            grid_size: The size of grid cells used for graph discretization
            pipe: The pipe to reroute
            subgraph: Set of node IDs in the isolated subgraph
            
        Returns:
            A directed graph for finding a path through the subgraph
        """
        # Collect node IDs from both subgraph and the pipe corners
        search_node_ids = tuple(subgraph | set(n.id for n in pipe.corners))
        
        # Get the actual Node objects for these IDs
        _nodes = [n for n in self.model.graph.iter_nodes() if n.id in search_node_ids] + [pipe.input, pipe.output]
        
        # Create a directed graph
        g = nx.DiGraph()
        g.add_nodes_from(self.model.graph.nodes_as_dict(_nodes))
        
        # Add edges between neighboring nodes
        for i, node in enumerate(_nodes):
            for other_node in _nodes[:]:
                if node.id != other_node.id and node.is_neighbor(other_node, grid_size):
                    g.add_edge(node.id, other_node.id)
                    
        # Ensure pipe input and output nodes are included
        g.add_nodes_from(self.model.graph.nodes_as_dict([pipe.input, pipe.output]))
        
        # Add special edges to ensure connectivity with existing pipe
        g.add_edge(pipe.input.id, pipe.corners[0].id)
        g.add_edge(pipe.corners[-1].id, pipe.output.id)
        
        return g
    
    def _reroute_pipe(self, subgraph: set, pipe: Pipe) -> Pipe:
        """
        Reroute a pipe through a subgraph to maximize coverage.
        
        This method uses the DFSSolver to find a longer path through the
        subgraph, creating more efficient pipe coverage.
        
        Args:
            subgraph: Set of node IDs in the isolated subgraph
            pipe: The pipe to reroute
            
        Returns:
            A new pipe with an optimized path
        """
        # Build a traversal graph for finding a path
        trav_graph = self._build_trav_graph(self.grid_size, pipe, subgraph)
        
        # Use DFSSolver to find the longest path through the subgraph
        solution = DFSSolver.solve(trav_graph, pipe.input.id, pipe.output.id, time_budget=self.time_budget)
        
        # Collect all possible nodes
        nodes = tuple(self.model.graph.nodes) + tuple([pipe.input, pipe.output]) + tuple(pipe.corners)
        
        # Convert path node IDs to actual Node objects
        path_nodes = [next(filter(lambda x: x.id == node_id, nodes), None) for node_id in solution]
        
        # Create a new pipe from the path
        _pipe = Pipe.from_path(self.config, path_nodes, self.model.distributor)
        return _pipe
    
    def route(self) -> Model:
        """
        Improve pipe routing by rerouting pipes through isolated areas.
        
        This method:
        1. Identifies isolated subgraphs (areas not covered by pipes)
        2. For each subgraph, finds nearby pipes that could be rerouted through it
        3. Attempts to reroute these pipes to maximize coverage
        4. Keeps the rerouted pipe only if it's longer than the original
        
        Returns:
            The updated model with optimized pipe routes
        """
        # Find all connected components (isolated subgraphs) in the remaining graph
        subgraphs = tuple(nx.connected_components(self.model.graph.graph))
        
        for subgraph in subgraphs:
            # Find pipes that are close to this subgraph
            pipes = self._find_best_pipe(subgraph, self.model.pipes, 
                                        tuple(self.model.graph.iter_nodes()), self.grid_size)
            
            for pipe in pipes:
                # Remove the original pipe from the model
                self.model.pipes = tuple(p for p in self.model.pipes if p != pipe)
                
                # Attempt to reroute the pipe through the subgraph
                _pipe = self._reroute_pipe(subgraph, pipe)
                
                # Only use the new pipe if it has more corners (better coverage)
                if len(_pipe.corners) > len(pipe.corners):
                    self.model.add(_pipe)
                    break  # We've successfully rerouted one pipe, move to next subgraph
                else:
                    # Keep the original pipe if the new one isn't better
                    self.model.add(pipe)
                    
        return self.model




