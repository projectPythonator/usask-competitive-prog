import string
from typing import List, Tuple, Set
from sys import setrecursionlimit


type Num = int | float
type IntList = List[int]
type FloatList = List[float]
type BoolList = List[float]
type NumList = List[Num]
type TupleListMST = List[Tuple[Num, int, int]]
type EdgeTupleList = List[Tuple[int, int]]
type EdgeTypeList = List[Tuple[int, int, int]]


setrecursionlimit(10000000)  # 10 million should be good enough for most contest problems
class UnionFindDisjointSets:
    """This Data structure is for none directional disjoint sets."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n        # optional optimization
        self.set_sizes = [1] * n   # optional information
        self.num_sets = n          # optional information
        
    def find_set(self, u):
        """Recursively find which set u belongs to. Memoize on the way back up.

        Complexity: Time: O(α(n)) -> O(1), inverse ackerman practically constant
                    Space: Amortized O(1) stack space
        """
        root_parent = u if self.parent[u] == u else self.find_set(self.parent[u])
        self.parent[u] = root_parent
        return root_parent
        
    def is_same_set(self, u, v):
        """Checks if u and v in same set. TIME and SPACE Complexity is the same as find_set"""
        return self.find_set(u) == self.find_set(v)

    def union_set(self, u, v):
        """Join the set that contains u with the set that contains v.

        Complexity: Time: O(α(n)) -> O(1), inverse ackerman practically constant
                    Space: Amortized O(1) stack space
        """
        if not self.is_same_set(u, v):
            u_parent, v_parent = self.find_set(u), self.find_set(v)
            if self.rank[u_parent] > self.rank[v_parent]:   # keep u_parent shorter than v_parent
                u_parent, v_parent = v_parent, u_parent

            self.parent[u_parent] = v_parent                     # this line joins u with v
            if self.rank[u_parent] == self.rank[v_parent]:       # an optional speedup
                self.rank[v_parent] += 1
            self.set_sizes[v_parent] += self.set_sizes[u_parent] # u -> v so add size_u to size_v
            self.num_sets -= 1

    def size_of_u(self, u): #optional information
        """Gives you the size of set u. TIME and SPACE Complexity is the same as find_set"""
        return self.set_sizes[self.find_set(u)]

######################################################################################

# from math import log2
from collections import deque
from heapq import heappush, heappop, heapify
from sys import setrecursionlimit

setrecursionlimit(100000)

class Graph:
    def __init__(self, v: int, e: int, r=None, c=None):
        self.num_edges: int = e
        self.num_nodes: int = v
        self.num_rows: int = r
        self.num_cols: int = c

        self.adj_list = []
        self.adj_list_trans = [] # for topological sort
        self.adj_matrix = []
        self.edge_list = []
        self.grid = []

        self.data_to_code: dict[object, int] = {}
        self.code_to_data: list[object] = []

    def convert_data_to_code(self, data: object) -> int:
        """Converts data to the form: int u | 0 <= u < |V|, stores (data, u) pair, then return u."""
        if data not in self.data_to_code:
            self.data_to_code[data] = len(self.code_to_data)
            self.code_to_data.append(data) # can be replaced with a count variable if space needed
        return self.data_to_code[data]

    def add_edge_u_v_wt_into_directed_graph(self, u: int, v: int, wt: Num=None, data: Num =None):
        """A pick and choose function will convert u, v into index form then add it to the structure
        you choose.
        """
        u: int = self.convert_data_to_code(u) # omit if u,v is in the form: int u | 0 <= u < |V|
        v: int = self.convert_data_to_code(v) # omit if u,v is in the form: int u | 0 <= u < |V|

        self.adj_list[u].append(v)
        # self.adj_list[u].append((v, wt))    # Adjacency list usage with weights
        self.adj_matrix[u][v] = wt          # Adjacency matrix usage
        self.edge_list.append((wt, u, v))   # Edge list usage
        # the following lines come as a pair-set used in max flow algorithm and are used in tandem.
        self.edge_list.append([v, wt, data])
        self.adj_list[u].append(len(self.edge_list) - 1)

    def add_edge_u_v_wt_into_undirected_graph(self, u: object, v: object, wt: int|float=None):
        """undirected graph version of the previous function"""
        self.add_edge_u_v_wt_into_undirected_graph(u, v, wt)
        self.add_edge_u_v_wt_into_undirected_graph(v, u, wt)

    def fill_grid_graph(self, new_grid: list[list[object]]):
        self.num_rows = len(new_grid)
        self.num_cols = len(new_grid[0])
        self.grid = [[self.convert_data_to_code(el) for el in row] for row in new_grid]

INF: int = 2**31
# turn these into enums later
UNVISITED: int = -1
EXPLORED: int  = -2
VISITED: int   = -3
# turn these into enums later
TREE: int = 0
BIDIRECTIONAL: int = 1
BACK: int = 2
FORWARD: int = 3

class GraphAlgorithms:
    def __init__(self, new_graph):
        self.graph: Graph = new_graph
        self.dfs_counter: int   = 0
        self.dfs_root: int      = 0
        self.root_children: int = 0
        self.region_num: int    = 0

        self.dir_rc = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.mst_node_list: TupleListMST = []
        self.dist: NumList = []
        self.visited: IntList = []
        self.topo_sort_node_list: IntList = []
        self.parent: IntList = []
        self.low_values: IntList = []
        self.articulation_nodes: BoolList = []
        self.bridge_edges: EdgeTupleList = []
        self.directed_edge_type: EdgeTypeList = [] # change last int to enum sometime
        self.component_region: IntList = []
        self.decrease_finish_order: IntList = []
        self.nodes_on_stack: IntList = []
        self.node_state: IntList = [] # change to enum type sometime
        self.bipartite_colouring: BoolList = []
        self.last: IntList = []

    def flood_fill_via_dfs(self, row: int, col: int, old_val: object, new_val: object): # retest needed
        """Computes flood fill graph traversal via recursive depth first search. Use on grid graphs.

        Complexity: Time: O(|V| + |E|), Space: O(|V|): for grids usually |V|=row*col and |E|=4*|V|
        More uses: Region Colouring, Connectivity, Area/island Size, misc
        Input
            row, col: integer pair representing current grid position
            old_val, new_val: unexplored state, the value of an explored state
        """
        self.graph.grid[row][col] = new_val
        for row_mod, col_mod in self.dir_rc:
            new_row, new_col = row + row_mod, col + col_mod
            if (0 <= new_row < self.graph.num_rows
                    and 0 <= new_col < self.graph.num_cols
                    and self.graph.grid[new_row][new_col] == old_val):
                self.flood_fill_via_dfs(new_row, new_col, old_val, new_val)

    def flood_fill_via_bfs(self, start_row: int, start_col: int, old_val: object, new_val: object): # retest needed
        """Computes flood fill graph traversal via breadth first search. Use on grid graphs.

        Complexity: Time: O(|V| + |E|), Space: O(|V|): for grids usually |V|=row*col and |E|=4*|V|
        More uses: previous uses tplus shortest connected pah
        """
        queue = deque([(start_row, start_col)])
        while queue:
            row, col = queue.popleft()
            for row_mod, col_mod in self.dir_rc:
                new_row, new_col = row + row_mod, col + col_mod
                if (0 <= new_row < self.graph.num_rows
                    and 0 <= new_col < self.graph.num_cols
                    and self.graph.grid[new_row][new_col] == old_val):
                    self.graph.grid[new_row][new_col] = new_val
                    queue.append((new_row, new_col))

    def min_spanning_tree_via_kruskals_and_heaps(self): # tested
        """Computes mst of graph G stored in edge_list, space optimized via heap.

        Complexity per call: Time: O(|E|log |V|), Space: O(|E|) + Union_Find
        More uses: finding min spanning tree
        Variants: min spanning subgraph and forrest, max spanning tree, 2nd min best spanning tree
        Optimization: We use a heap to make space comp. O(|E|). instead of O(|E|log |E|)
        when using sort, however edge_list is CONSUMED.
        """
        heapify(self.graph.edge_list)
        ufds = UnionFindDisjointSets(self.graph.num_nodes)
        min_spanning_tree: TupleListMST = []
        while self.graph.edge_list and ufds.num_sets > 1:
            wt, u, v = heappop(self.graph.edge_list) # use w, uv = ... for single cord storage
            #v,u = uv%self.num_nodes, uv//self.num_nodes
            if not ufds.is_same_set(u, v):
                min_spanning_tree.append((wt, u, v))
                ufds.union_set(u, v)
        self.mst_node_list = min_spanning_tree
        
    def prims_visit_adj_matrix(self, u: int, not_visited: Set[int], mst_best_dist: NumList, heap):
        """Find min weight edge in adjacency matrix implementation of prims.

        Complexity per call: Time: O(|V|), Space: O(1)
        """
        # NEEDS FIXING 
        not_visited.remove(u)
        for v in not_visited:
            wt = self.graph.adj_matrix[u][v]
            if wt <= mst_best_dist[v]:
                mst_best_dist[v] = wt
                heappush(heap, (wt, v, u)) # fix this
    
    def prims_visit_adj_list(self, u: int, not_visited: BoolList, mst_best_dist: NumList, heap): # retest needed
        """Find min weight edge in adjacency list implementation of prims.

        Complexity per call: Time: O(|V|log |V|), Space: increase by O(|V|)
        """
        not_visited[u] = False
        for v, wt in self.graph.adj_list[u]:
            if wt <= mst_best_dist[v] and not_visited[v]:
                mst_best_dist[v] = wt
                heappush(heap, (wt, v, u))
    
    def min_spanning_tree_via_prims(self): # retest needed
        """Computes mst of graph G stored in adj_list.

        Complexity: Time: O(|E|log |V|) or O(|V|^2), Space: O(|E|) or O(|V|^2)
        More uses: gets a different min spamming tree than kruskal's
        """
        not_visited = [True] * self.graph.num_nodes
        mst_best_dist = [INF] * self.graph.num_nodes
        heap, min_spanning_tree, nodes_taken = [], [], 0
        self.prims_visit_adj_list(0, not_visited, mst_best_dist, heap)
        while heap and nodes_taken < self.graph.num_nodes:
            wt, v, u = heappop(heap)
            if not_visited[v]:
                self.prims_visit_adj_list(v, not_visited, mst_best_dist, heap)
                min_spanning_tree.append((wt, v, u))
                nodes_taken += 1
        self.mst_node_list = min_spanning_tree
        self.mst_node_list.sort()

    def breadth_first_search_vanilla_template(self, source: int): # retest needed
        """Template for distance based bfs traversal from node source.

        Complexity per call: Time: O(|V| + |E|), Space: O(|V|)
        More uses: connectivity, shortest path on monotone weighted graphs
        """
        distance = [UNVISITED] * self.graph.num_nodes
        queue, distance[source] = deque([source]), 0
        while queue:
            u = queue.popleft()
            for v in self.graph.adj_list[u]:
                if distance[v] == UNVISITED:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        self.dist = distance

    def topology_sort_via_tarjan_helper(self, u: int): # retest
        """Recursively explore unvisited graph via dfs.

        Complexity per call: Time: O(|V|), Space: O(|V|) at deepest point
        """
        self.visited[u] = VISITED
        for v in self.graph.adj_list[u]:
            if self.visited[v] == UNVISITED:
                self.topology_sort_via_tarjan_helper(v)
        self.topo_sort_node_list.append(u)
        
    def topology_sort_via_tarjan(self): # retest
        """Compute a topology sort via tarjan method, on adj_list.

        Complexity per call: Time: O(|V| + |E|), Space: O(|V|)
        More Uses: produces a DAG, topology sorted graph, build dependencies
        """
        self.visited = [UNVISITED] * self.graph.num_nodes
        self.topo_sort_node_list = []
        for u in range(self.graph.num_nodes):
            if self.visited[u] == UNVISITED:
                self.topology_sort_via_tarjan_helper(u)
        self.topo_sort_node_list = self.topo_sort_node_list[::-1]

    def topology_sort_via_kahns(self): # retest
        """Compute a topology sort via kahn's method, on adj_list.

        Complexity per call: Time: O(|E|log|V|), Space: O(|V|)
        More uses: different ordering as tarjan's method
        bonus: heaps allow for custom ordering (i.e. use lowest indices first)
        """
        in_degree = [0] * self.graph.num_nodes
        for list_of_u in self.graph.adj_list:
            for v in list_of_u:
                in_degree[v] += 1
        topo_sort = []
        heap = [u for u, el in enumerate(in_degree) if el == 0]
        heapify(heap)
        while heap:
            u = heappop(heap)
            topo_sort.append(u)
            for v in self.graph.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] <= 0:
                    heappush(heap, v)
        self.topo_sort_node_list = topo_sort

    def amortized_heap_fix(self, heap):
        """Should we need |V| space this will ensure that while still being O(log|V|)"""
        tmp = [-1] * self.graph.num_nodes
        for wt, v in heap:
            if tmp[v] == -1:
                tmp[v] = wt
        heap = [(wt, v) for v, wt in enumerate(tmp) if wt != -1]
        heapify(heap)

    def single_source_shortest_path_dijkstras(self, source: int, sink: int=None): # retest
        """It is Dijkstra's pathfinder using heaps.

        Complexity per call: Time: O(|E|log |V|), Space: O(|V|)
        More uses: shortest path on state based graphs
        Input:
            source: can be a single nodes or list of nodes
            sink: the goal node
        """
        distance, parents = [INF] * self.graph.num_nodes, [UNVISITED] * self.graph.num_nodes
        distance[source], parents[source] = 0, source
        heap = [(0, source)]
        while heap:
            cur_dist, u = heappop(heap)
            if distance[u] < cur_dist:
                continue
            # if u == sink: return cur_dist # uncomment this line for fast return
            for v, wt in self.graph.adj_list[u]:
                if distance[v] > cur_dist + wt:
                    distance[v] = cur_dist + wt
                    parents[v] = u
                    heappush(heap, (distance[v], v))
        self.dist = distance
        self.parent = parents
    
    def all_pairs_shortest_path_floyd_warshall(self): # tested
        """Computes essentially a matrix operation on a graph.

        Complexity per call: Time: O(|V|^3), Space: O(|V|^2)
        More uses: Shortest path, Connectivity.
        Variants: Transitive closure, Maximin and Minimax path, Cheapest negative cycle,
                  Finding diameter of a graph, Finding SCC of a directed graph.
        """
        matrix = self.graph.adj_matrix
        for k in range(self.graph.num_nodes):
            for i in range(self.graph.num_nodes):
                for j in range(self.graph.num_nodes):
                    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])

    def apsp_floyd_warshall_neg_cycles(self): #needs test
        matrix = self.graph.adj_matrix
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                for k in range(self.graph.num_nodes):
                    if matrix[k][k] < 0 and matrix[i][k] != INF and matrix[k][j] != INF:
                        matrix[i][j] = -INF

    def articulation_point_and_bridge_helper_via_dfs(self, u: int): # retest needed
        """Recursion part of the dfs. It kind of reminds me of how Union find works.

        Complexity per call: Time: O(|E|), Space: O(|V|)
        """
        self.visited[u] = self.dfs_counter
        self.low_values[u] = self.visited[u]
        self.dfs_counter += 1
        for v in self.graph.adj_list[u]:
            if self.visited[v] == UNVISITED:
                self.parent[v] = u
                if u == self.dfs_root:
                    self.root_children += 1
                self.articulation_point_and_bridge_helper_via_dfs(v)
                if self.low_values[v] >= self.visited[u]:
                    self.articulation_nodes[u] = True
                    if self.low_values[v] > self.visited[u]:
                        self.bridge_edges.append((u, v))
                self.low_values[u] = min(self.low_values[u], self.low_values[v])
            elif v != self.parent[u]:
                self.low_values[u] = min(self.low_values[u], self.visited[v])

    def articulation_points_and_bridges_via_dfs(self): # retest needed
        """Generates the name on an adj_list based graph.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        More uses: finding the sets of single edge and vertex removals that disconnect the graph.
        Bridges stored as edges and points are True values in articulation_nodes.
        """
        self.dfs_counter = 0
        for u in range(self.graph.num_nodes):
            if self.visited[u] == UNVISITED:
                self.dfs_root = u
                self.root_children = 0
                self.articulation_point_and_bridge_helper_via_dfs(u)
                self.articulation_nodes[self.dfs_root] = (self.root_children > 1)

    def cycle_check_on_directed_graph_helper(self, u: int): # retest needed
        """Recursion part of the dfs. It is modified to list various types of edges.

        Complexity per call: Time: O(|E|), Space: O(|V|) at deepest call
        More uses: listing edge types: Tree, Bidirectional, Back, Forward/Cross edge. On top of
        listing Explored, Visited, and Unvisited.
        """
        self.visited[u] = EXPLORED
        for v in self.graph.adj_list[u]:
            edge_type: int = TREE
            if self.visited[v] == UNVISITED:
                edge_type = TREE
                self.parent[v] = u
                self.cycle_check_on_directed_graph_helper(v)
            elif self.visited[v] == EXPLORED:
                edge_type = BIDIRECTIONAL if v == self.parent[u] else BACK # graph is not DAG.
            elif self.visited[v] == VISITED:
                edge_type = FORWARD
            self.directed_edge_type.append((u, v, edge_type))
        self.visited[u] = VISITED

    def cycle_check_on_directed_graph(self): # retest needed
        """Determines if a graph is cyclic or acyclic via dfs.

        Complexity per call: Time: O(|E| + |V|),
                            Space: O(|E|) if you label each edge O(|V|) otherwise.
        More uses: Checks if graph is acyclic(DAG) which can open potential for efficient algorithms
        """
        self.visited = [UNVISITED] * self.graph.num_nodes
        self.directed_edge_type = [] # can be swapped out for a marked variable
        for u in range(self.graph.num_nodes):
            if self.visited[u] == UNVISITED:
                self.cycle_check_on_directed_graph_helper(u)
  
    def strongly_connected_components_of_graph_kosaraju_helper(self, u: int, pass_one: bool): # retest needed
        """Pass one explore G and build stack, Pass two mark the SCC regions on transposition of G.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        """
        self.visited[u] = VISITED
        self.component_region[u] = self.region_num
        neighbours: IntList = self.graph.adj_list[u] if pass_one else self.graph.adj_list_trans[u]
        for v in neighbours:
            if self.visited[v] == UNVISITED:
                self.strongly_connected_components_of_graph_kosaraju_helper(v, pass_one)
        if pass_one:
            self.decrease_finish_order.append(u)

    def strongly_connected_components_of_graph_kosaraju(self):  # retest needed
        """Marks the SCC of a directed graph using Kosaraju's method.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        More Uses: Labeling and Identifying SCC regions(marks regions by numbers).
        """
        self.visited = [UNVISITED] * self.graph.num_nodes
        self.component_region = [0] * self.graph.num_nodes
        for u in range(self.graph.num_nodes):
            if self.visited[u] == UNVISITED:
                self.strongly_connected_components_of_graph_kosaraju_helper(u, True)
        self.visited = [UNVISITED] * self.graph.num_nodes
        self.region_num = 1
        for u in reversed(self.decrease_finish_order):
            if self.visited[u] == UNVISITED:
                self.strongly_connected_components_of_graph_kosaraju_helper(u, False)
                self.region_num += 1
    
    def strongly_connected_components_of_graph_tarjans_helper(self, u: int): # retest needed
        """Recursive part of tarjan's, pre-order finds the SCC regions, marks regions post-order.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        """
        self.low_values[u] = self.node_state[u] = self.dfs_counter
        self.dfs_counter += 1
        self.nodes_on_stack.append(u)
        self.visited[u] = VISITED
        for v in self.graph.adj_list[u]:
            if self.node_state[v] == UNVISITED:
                self.strongly_connected_components_of_graph_tarjans_helper(v)
            if self.visited[v] == VISITED:
                self.low_values[u] = min(self.low_values[u], self.low_values[v])
        if self.low_values[u] == self.node_state[u]:
            self.region_num += 1
            while True:
                v: int = self.nodes_on_stack.pop()
                self.visited[v], self.component_region[v] = UNVISITED, self.region_num
                if u == v:
                    break

    def strongly_connected_components_of_graph_tarjans(self): # retest needed
        """Marks the SCC regions of a directed graph using tarjan's method.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        More uses: Labeling and Identifying SCC regions(marks regions by numbers).
        """
        max_v = self.graph.num_nodes
        self.visited, self.node_state = [UNVISITED] * max_v, [UNVISITED] * max_v
        self.low_values, self.component_region = [0] * max_v, [0] * max_v
        self.nodes_on_stack, self.region_num, self.dfs_counter = [], 0, 0
        for u in range(self.graph.num_nodes):
            if self.node_state[u] == UNVISITED:
                self.strongly_connected_components_of_graph_tarjans_helper(u)

    def bipartite_check_on_graph_helper(self, source: int, color: IntList): # retest needed
        """Uses bfs to check if the graph region connected to source is bipartite.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        """
        queue = deque([source])
        color[source] = 0
        is_bipartite = True
        while queue and is_bipartite:
            u = queue.popleft()
            for v in self.graph.adj_list[u]:
                if color[v] == UNVISITED:
                    color[v] = not color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    is_bipartite = False
                    break
        return is_bipartite

    def bipartite_check_on_graph(self): # retest needed
        """Checks if a graph has the bipartite property.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        More Uses: check bipartite property, labeling a graph for 2 coloring if it is bipartite.
        """
        is_bipartite, color = True, [UNVISITED] * self.graph.num_nodes
        for u in range(self.graph.num_nodes):
            if color[u] == UNVISITED:
                is_bipartite = is_bipartite and self.bipartite_check_on_graph_helper(u, color)
                if not is_bipartite:
                    break
        self.bipartite_colouring = color if is_bipartite else None


    def max_flow_find_augmenting_path_helper(self, source: int, sink: int): # new, testing needed
        """Will check if augmenting path in the graph from source to sink exists via bfs.

        Complexity per call: Time: O(|E| + |V|), Space O(|V|)
        Input
            source: the node which we are starting from.
            sink: the node which we are ending on.
        """
        distance, parents = [-1] * self.graph.num_nodes, [(-1, -1)] * self.graph.num_nodes
        queue, distance[source] = deque([source]), 0
        while queue:
            u = queue.popleft()
            if u == sink:
                self.dist, self.parent = distance, [el for el in parents]
                return True
            for idx in self.graph.adj_list[u]:
                v, cap, flow = self.graph.edge_list[idx]
                if cap - flow > 0 and distance[v] == -1:
                    distance[v] = distance[u] + 1
                    parents[v] = (u, idx)
                    queue.append(v)
        self.dist, self.parent = [], []
        return False

    def send_flow_via_augmenting_path(self, source: int, sink: int, flow_in: Num): # testing needed
        """Function to recursively emulate sending a flow. returns min pushed flow.

        Complexity per call: Time: O(|V|), Space O(|V|)
        Uses: preorder finds the min pushed_flow post order mutates edge_list based on that flow.
        Input:
            source: in this function it's technically the goal node
            sink: the current node we are observing
            flow_in: the smallest flow found on the way down
        """
        if source == sink:
            return flow_in
        u, edge_ind = self.parent[sink]
        _, edge_cap, edge_flow = self.graph.edge_list[edge_ind]
        pushed_flow = self.send_flow_via_augmenting_path(
            source, u, min(flow_in, edge_cap - edge_flow))
        self.graph.edge_list[edge_ind][2] = edge_flow + pushed_flow
        self.graph.edge_list[edge_ind ^ 1][2] -= pushed_flow
        return pushed_flow

    def send_max_flow_via_dfs(self, u: int, sink: int, flow_in: Num): # testing needed
        """Function to recursively emulate sending a flow via dfs. Returns min pushed flow.

        Complexity per call: Time: O(|E| * |V|), Space O(|V|)
        More uses: a more efficient way of sending a flow
        Input:
            u: is the current node to be observed.
            sink: is the goal node (we might be able to just put it as instance var?).
            flow_in: the smallest flow found on the way down.
        """
        if u == sink or flow_in == 0:
            return flow_in
        start, end = self.last[u], len(self.graph.adj_list[u])
        for i in range(start, end):
            self.last[u], edge_ind = i, self.graph.adj_list[u][i]
            v, edge_cap, edge_flow = self.graph.edge_list[edge_ind]
            if self.dist[v] != self.dist[u] + 1:
                continue
            pushed_flow = self.send_max_flow_via_dfs(v, sink, min(flow_in, edge_cap - edge_flow))
            if pushed_flow != 0:
                self.graph.edge_list[edge_ind][2] = edge_flow + pushed_flow
                self.graph.edge_list[edge_ind ^ 1][2] -= pushed_flow
                return pushed_flow
        return 0

    def max_flow_via_edmonds_karp(self, source: int, sink: int):
        """Compute max flow using edmonds karp's method.

        Complexity per call: Time: O(|V| * |E|^2), Space O(|V|)
        More Uses: max flow of the graph, min cut of the graph.
        """
        max_flow = 0
        while self.max_flow_find_augmenting_path_helper(source, sink):
            flow = self.send_flow_via_augmenting_path(source, sink, INF)
            if flow == 0:
                break
            max_flow += flow
        return max_flow

    def max_flow_via_dinic(self, source: int, sink: int):
        """Compute max flow using Dinic's method.

        Complexity per call: Time: O(|E| * |V|^2), Space O(|V|)
        More Uses: faster than the one above for most cases.
        """
        max_flow = 0
        while self.max_flow_find_augmenting_path_helper(source, sink):
            self.last = [0] * self.graph.num_nodes
            flow = self.send_max_flow_via_dfs(source, sink, INF)
            while flow != 0:
                max_flow += flow
                flow = self.send_max_flow_via_dfs(source, sink, INF)
        return max_flow

from math import isqrt, log, gcd, prod
from itertools import takewhile

class MathAlgorithms:
    def __init__(self):
        """Only take what you need. This list needs to be global or instance level or passed in."""
        self.mod_p = 0
        self.binomial = {}
        self.fact = []
        self.inv_fact = []
        self.min_primes_list = []
        self.catalan_numbers = []
        self.primes_sieve = []
        self.primes_list = []
        self.primes_set = set()
        self.prime_factors = []
        self.mrpt_known_bounds = []
        self.mrpt_known_tests = []
        self.fibonacci_list = []
        self.fibonacci_dict = {}
        self.fibonacci_dict = {0: 0, 1: 1, 2: 1}
        
    def is_prime_triv(self, n):
        """Tests if n is prime via divisors up to sqrt(n)."""
        if n <= 3: return n > 1
        if n%2 == 0 or n%3 == 0: return False
        limit = isqrt(n) + 1
        for p in range(5, limit+1, 6):
            if n % p == 0 or n % (p+2) == 0:
                return False
        return True

    def sieve_of_eratosthenes(self, n):
        """Generates list of primes up to n via eratosthenes method.

        Complexity: Time: O(n lnln(n)), Space: post call O(n/ln(n)), mid-call O(n)
        Variants: number and sum of prime factors, of diff prime factors, of divisors, and euler phi
        """
        limit, prime_sieve = isqrt(n) + 1, [True] * (n + 1)
        prime_sieve[0] = prime_sieve[1] = False
        for i in range(2, limit):
            if prime_sieve[i]:
                for j in range(i*i, n+1, i):
                    prime_sieve[j] = False
        self.primes_list = [2] + [i for i in enumerate(prime_sieve) if i]
    
    def sieve_of_eratosthenes_optimized(self, n):
        """Odds only optimized version of the previous method

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n/ln(n)), mid-call O(n/2)
        """
        sqrt_n, limit = ((isqrt(n) - 3)//2) + 1, ((n - 3)//2) + 1
        primes_sieve = [True] * limit
        for i in range(sqrt_n):
            if primes_sieve[i]:
                prime = 2*i + 3
                start = (prime*prime - 3)//2
                for j in range(start, limit, prime):
                    primes_sieve[j] = False
        self.primes_list = [2] + [2*i + 3 for i, el in enumerate(primes_sieve) if el]

    def sieve_of_min_primes(self, n):
        """Stores the min prime for each number up to n.

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n)
        """
        min_primes = [0] * (n + 1)
        min_primes[1] = 1
        for prime in reversed(self.primes_list):
            min_primes[prime] = prime
            start, end, step = prime*prime, n+1, prime if prime==2 else 2*prime
            for j in range(start, end, step):
                min_primes[j] = prime
        self.min_primes_list = min_primes

    def sieve_of_eratosthenes_variants(self, n):
        """Seven variants of prime sieve listed above.

        Complexity:
            function 1: Time: O(n lnln(n)), Space: O(n)
            function 3: Time: O(n log(n)), Space: O(n)
            function 2: Time: O(n lnln(n) log(n)), Space: O(n)
        """
        def euler_phi_plus_sum_and_number_of_diff_prime_factors(limit):
            """This is basically same as sieve just using different ops. Complexity function 1."""
            num_diff_pf = [0] * (limit + 1)
            sum_diff_pf = [0] * (limit + 1)
            phi = [i for i in range(limit + 1)]
            for i in range(2, limit):
                if num_diff_pf[i] == 0:
                    for j in range(i, limit, i):
                        num_diff_pf[j] += 1
                        sum_diff_pf[j] += i
                        phi[j] = (phi[j]//i) * (i-1)
            self.num_diff_prime_factors = sum_diff_pf
            self.sum_diff_prime_factors = sum_diff_pf
            self.euler_phi = phi
            
        def num_and_sum_of_divisors(limit):
            """Does a basic sieve. Complexity function 2."""
            num_div = [1] * (limit + 1)
            sum_div = [1] * (limit + 1)
            for i in range(2, limit):
                for j in range(i, limit, i):
                    num_div[j] += 1
                    sum_div[j] += i
            self.num_divisors = num_div
            self.sum_divisors = sum_div

        def num_and_sum_of_prime_factors(limit):
            """This uses similar idea to sieve but avoids divisions. Complexity function 3."""
            num_pf = [0] * (limit + 1)
            sum_pf = [0] * (limit + 1)
            self.sieve_of_eratosthenes_optimized(limit)
            for prime in self.primes_list:
                exponent_limit = int(log(limit, prime)) + 1
                for exponent in range(1, exponent_limit):
                    prime_to_exponent = prime**exponent
                    for i in range(prime_to_exponent, limit + 1, prime_to_exponent):
                        sum_pf[i] += prime
                        num_pf[i] += 1
            self.num_prime_factors = num_pf
            self.sum_prime_factors = sum_pf
    
    def gen_set_primes(self):
        self.primes_set=set(self.primes_list)

    def prime_factorize_n(self, n):
        """A basic prime factorization of n function. without primes its just O(sqrt(n))

        Complexity: Time: O(sqrt(n)/ln(sqrt(n))), Space: O(log n)
        Variants: number and sum of prime factors, of diff prime factors, of divisors, and euler phi
        """
        limit, prime_factors = isqrt(n) + 1, []
        for prime in takewhile(lambda x: x < limit, self.primes_list):
            if n % prime == 0:
                while n % prime == 0:
                    n //= prime
                    prime_factors.append(prime)
        if n > 1: prime_factors.append(n)
        return prime_factors

    def prime_factorize_n_log_n(self, n):
        """An optimized prime factorization of n function based on min primes already sieved

        Complexity: Time: O(log n), Space: O(log n)
        """
        prime_factors = []
        app = prime_factors.append
        while n > 1:
            prime = self.min_primes_list[n]
            app(prime)
            n = n // prime
        return prime_factors

    def prime_factorize_n_variants(self, n):
        """Covers all the variants listed above, holds the same time complexity with O(1) space."""
        limit = isqrt(n) + 1
        sum_diff_prime_factors, num_diff_prime_factors = 0, 0
        sum_prime_factors, num_prime_factors = 0, 0
        sum_divisors, num_divisors, euler_phi = 1, 1, n
        for prime in takewhile(lambda x: x < limit, self.primes_list):
            if n % prime == 0:
                mul, total = prime, 1  # for sum of divisors
                power = 0              # for num of divisors
                while n % prime == 0:
                    n //= prime
                    power += 1         # for num prime factors, num divisors, and sum prime factors
                    total += mul       # for sum divisors
                    mul *= prime       # for sum divisors
                sum_diff_prime_factors += prime
                num_diff_prime_factors += 1
                sum_prime_factors += (prime * power)
                num_prime_factors += power
                num_divisors *= (power + 1)
                sum_divisors *= total
                euler_phi -= (euler_phi//prime)
        if n > 1:
            num_diff_prime_factors += 1
            sum_diff_prime_factors += n
            num_prime_factors += 1
            sum_prime_factors += n
            num_divisors *= 2
            sum_divisors *= (n + 1)
            euler_phi -= (euler_phi // n)
        return num_diff_prime_factors

    def is_composite(self, a, d, n, s):
        """The witness test of miller rabin.

        Complexity per call: Time O(log^3(n)), Space: O(2**s) bits
        """
        if 1 == pow(a, d, n):
            return False
        for i in range(s):
            if n-1 == pow(a, d * 2**i, n):
                return False
        return True

    def miller_rabin_primality_test(self, n, precision_for_huge_n=16):
        """Probabilistic primality test with error rate of 4^(-k) past 341550071728321.

        Complexity per call: Time O(k log^3(n)), Space: O(2**s) bits
        """
        if n in self.primes_set:
            return True
        if any((n%self.primes_list[p] == 0) for p in range(50)) or n < 2 or n == 3215031751:
            return False
        d, s = n-1, 0
        while d % 2 == 0:
            d, s = d//2, s+1
        for i, bound in enumerate(self.mrpt_known_bounds):
            if n < bound:
                return not any(self.is_composite(self.mrpt_known_tests[j], d, n, s)
                               for j in range(i))
        return not any(self.is_composite(self.primes_list[j], d, n, s)
                       for j in range(precision_for_huge_n))
    
    def miller_rabin_primality_test_prep(self):
        """This function needs to be called before miller rabin"""
        self.mrpt_known_bounds = [1373653, 25326001, 118670087467,
                                  2152302898747, 3474749660383, 341550071728321]
        self.mrpt_known_tests = [2, 3, 5, 7, 11, 13, 17]
        self.sieve_of_eratosthenes(1000) # comment out if different size needed
        self.gen_set_primes() # comment out if already have bigger size

    #test this against stevens
    def extended_euclid_recursive(self, a, b):
        """Solves coefficients of Bezout identity: ax + by = gcd(a, b), recursively

        Complexity per call: Time: O(log n), Space: O(log n) at the deepest call.
        """
        if 0 == b:
            return 1, 0, a
        x, y, d = self.extended_euclid_recursive(b, a%b)
        return y, x-y*(a//b), d

    def extended_euclid_iterative(self, a, b):
        """Solves coefficients of Bezout identity: ax + by = gcd(a, b), iteratively.

        Complexity per call: Time: O(log n) about twice as fast in python vs above, Space: O(1)
        Optimizations and notes:
            divmod and abs are used to help deal with big numbers, remove if < 2**64 for speedup.
        """
        last_remainder, remainder = abs(a), abs(b)
        x, y, last_x, last_y = 0, 1, 1, 0
        while remainder:
            last_remainder, (quotient, remainder) = remainder, divmod(last_remainder, remainder)
            x, last_x = last_x - quotient * x, x
            y, last_y = last_y - quotient * y, y
        return -last_x if a < 0 else last_x, -last_y if b < 0 else last_y, last_remainder

    def safe_modulo(self, a, n): #needs test
        """Existence is much for c++ which doesn't always handle % operator nicely.
        use ((a % n) + n) % n for getting proper mod of a potential negative value
        use (a + b) % --> ((a % n) + (b % n)) % n for operations sub out + for * and -
        """
        return ((a % n) + n) % n

    def modular_linear_equation_solver(self, a, b, n):
        """Solves gives the solution x in ax = b(mod n).

        Complexity per call: Time: O(log n), Space: O(d)
        """
        x, y, d = self.extended_euclid_iterative(a, n)
        if 0 == b % d:
            x = (x * (b//d)) % n
            return [(x + i*(n//d)) % n for i in range(d)]
        return []

    def linear_diophantine_1(self, a, b, c):
        """Solves for x, y in ax + by = c. From stanford icpc 2013-14

        Complexity per call: Time: O(log n), Space: O(1).
        Notes: order matters? 25x + 18y = 839 != 18x + 25y = 839
        """
        d = gcd(a, b)
        if c % d == 0:
            x = c//d * self.mod_inverse(a//d, b//d)
            return x, (c - a * x) // b
        return -1, -1

    def linear_diophantine_2(self, a, b, c):
        """Solves for x0, y0 in x = x0 + (b/d)n, y = y0 - (a/d)n.
        derived from ax + by = c, d = gcd(a, b), and d|c.
        Can further derive into: n = x0 (d/b), and n = y0 (d/a).

        Complexity per call: Time: O(log n), Space: O(1).
        Optimizations and notes:
            unlike above this function order doesn't matter if a != b
            for a speedup call math.gcd(a, b) at start and return accordingly on two lines
        """
        x, y, d = self.extended_euclid_iterative(a, b)
        return (-1, -1) if c % d != 0 else (x * (c // d), y * (c // d))


    def mod_inverse(self, b, m): #needs test
        """Solves b^(-1) (mod m).

        Complexity per call: Time: O(log n), Space: O(1)
        """
        x, y, d = self.extended_euclid_iterative(b, m)
        return None if d != 1 else x % m  # -1 instead of None if we intend to go on with the prog

    def chinese_remainder_theorem_1(self, remainders, modulos):
        """Steven's CRT version to solve x in x = r[0] (mod m[0]) ... x = r[n-1] (mod m[n-1]).

        Complexity per call: Time: O(n log n), Space: O(1)? O(mt) bit size:
        Optimizations:
            prod is used from math since 3.8,
            we use mod mt in the forloop since x might get pretty big.
        """
        mt, x = prod(modulos), 0
        for i, modulo in enumerate(modulos):
            p = mt // modulo
            x = (x + (remainders[i] * self.mod_inverse(p, modulo) * p)) % mt
        return x


    # stanford icpc 2013-14
    def chinese_remainder_theorem_helper(self, mod1, rem1, mod2, rem2): #needs test
        """Chinese remainder theorem (special case): find z such that z % m1 = r1, z % m2 = r2.
        Here, z is unique modulo M = lcm(m1, m2). Return (z, M).  On failure, M = -1.
        from: stanford icpc 2016

        Complexity per call: Time: O(log n), Space: O(1)
        """
        s, t, d = self.extended_euclid_iterative(mod1, mod2)
        if rem1 % d != rem2 % d:
            mod3, sremmod, tremmod = mod1*mod2, s*rem2*mod1, t*rem1*mod2
            return ((sremmod + tremmod) % mod3) // d, mod3 // d
        return 0, -1
    
    # from stanford icpc 2013-14
    def chinese_remainder_theorem_2(self, remainders, modulos):
        """Chinese remainder theorem: find z such that z % m[i] = r[i] for all i.  Note that the
        solution is unique modulo M = lcm_i (m[i]).  Return (z, M). On failure, M = -1. Note that
        we do not require the r[i]'s to be relatively prime.
        from: stanford icpc 2016

        Complexity per call: Time: O(n log n), Space: O(1)? O(mt) bit size
        """
        z_m = remainders[0], modulos[0]
        for i, modulo in enumerate(modulos[1:], 1):
            z_m = self.chinese_remainder_theorem_helper(z_m[1], z_m[0], modulo, remainders[i])
            if -1 == z_m[1]:
                break
        return z_m

    def fibonacci_n_iterative(self, n):
        """Classic fibonacci solver. Generates answers from 0 to n inclusive.

        Complexity per call: Time: O(n), Space: O(n).
        """
        fib_list = [0] * (n + 1)
        fib_list[1] = 1
        for i in range(2, n+1):
            fib_list[i] = fib_list[i - 1] + fib_list[i - 2]
        self.fibonacci_list = fib_list

    def fibonacci_n_dp_1(self, n):
        """Dynamic programming way to compute the nth fibonacci.

        Complexity per call: Time: O(log n), Space: increase by O(log n).
        """
        if n in self.fibonacci_dict:
            return self.fibonacci_dict[n]
        f1 = self.fibonacci_n_dp_1(n // 2 + 1)
        f2 = self.fibonacci_n_dp_1((n - 1) // 2)
        self.fibonacci_dict[n] = (f1 * f1 + f2 * f2 if n & 1 else f1 * f1 - f2 * f2)
        return self.fibonacci_dict[n]
    
    #this needs testing 
    def generate_catalan_n(self, n):
        """Generate catalan up to n iteratively.

        Complexity per call: Time: O(n), Space: O(n * 2^(log n)).
        """
        catalan = [0] * (n+1)
        catalan[0] = 1
        for i in range(n-1):
            catalan[i + 1] = catalan[i] * (4*i + 2) // (i + 2)
        self.catalan_numbers = catalan

    def generate_catalan_n_mod_inverse(self, n, p):
        """Generate catalan up to n iteratively cat n % p.

        Complexity per call: Time: O(n log n), Space: O(n * (2^(log n)%p)).
        Variants: use prime factors of the factorial to cancel out the primes
        """
        catalan = [0] * (n+1)
        catalan[0] = 1
        for i in range(n-1):
            catalan[i+1] = ((4*i + 2)%p * catalan[i]%p * pow(i+1, p-2, p)) % p
        self.catalan_numbers = catalan

    def c_n_k(self, n, k):
        """Computes C(n, k) % p. From competitive programming 4.

        Complexity per call: v1: Time: O(log n), v2 Time: O(1), Space: O(1).
        v1 is uncommented and did not code inv_fact, v2 is the commented out line.
        """
        if n < k:
            return 0
        return (self.fact[n] * pow(self.fact[k], self.mod_p - 2, self.mod_p)
                * pow(self.fact[n - k], self.mod_p - 2, self.mod_p)) % self.mod_p
        # return 0 if n < k else (self.fact[n] * self.inv_fact[k] * self.inv_fact[n-k]) % self.mod_p

    def binomial_coefficient_n_mod_p_prep(self, max_n, mod_p):
        """Does preprocessing for binomial coefficients. From competitive programming 4.

        Complexity per call: Time: O(n), Space: O(n).
        Optimization and notes: use uncommented lines for C(n, k) % p in O(1) time
        """
        factorial_mod_p = [1] * max_n
        for i in range(1, max_n):
            factorial_mod_p[i] = (factorial_mod_p[i - 1] * i) % mod_p
        self.mod_p, self.fact = mod_p, factorial_mod_p
        # inverse_factorial_mod_p = [0] * max_n
        # inverse_factorial_mod_p[-1] = pow(factorial_mod_p[-1], mod_p-2, mod_p)
        # for i in range(max_n-2, -1, -1):
        #     inverse_factorial_mod_p[i] = (inverse_factorial_mod_p[i+1] * (i+1)) % mod_p
        # self.inv_fact = inverse_factorial_mod_p
            
    def binomial_coefficient_dp(self, n, k):
        """Uses the recurrence to calculate binomial coefficient.
        
        Complexity per call: Time: O(n*k) I think, Space: O(n*k).
        """
        if n == k or 0 == k:
            return 1
        if (n, k) not in self.binomial:
            self.binomial[(n, k)] = self.binomial_coefficient_dp(n-1, k) + self.binomial_coefficient_dp(n-1, k-1)
        return self.binomial[(n, k)]


from math import isclose, dist, sin, cos, acos, sqrt, fsum, pi
# remember to sub stuff out for integer ops when you want only integers 
# for ints need to change init, eq and
# hard code these in for performance speedup
CCW = 1  # counterclockwise
CW = -1  # clockwise
CL = 0   # collinear
EPS = 1e-12 # used in some spots


def pairwise(seq):
    it = iter(seq); next(it)
    return zip(iter(seq), it)


class Pt2d:
    __slots__ = ("x", "y")
    # def __init__(self, x_val, y_val): self.x, self.y = map(float, (x_val, y_val))

    def __init__(self, x_val, y_val): self.x, self.y = x_val, y_val

    def __add__(self, other): return Pt2d(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Pt2d(self.x - other.x, self.y - other.y)
    def __mul__(self, scale): return Pt2d(self.x * scale, self.y * scale)
    def __truediv__(self, scale): return Pt2d(self.x / scale, self.y / scale)
    def __floordiv__(self, scale): return Pt2d(self.x // scale, self.y // scale)

    # def __eq__(self, other): return isclose(self.x, other.x) and isclose(self.y, other.y)
    def __eq__(self, other): return self.x == other.x and self.y == other.y

    # def __lt__(self, other):
    #     return self.x < other.x if not isclose(self.x, other.x) else self.y < other.y
    def __lt__(self, other): return self.x < other.x if self.x != other.x else self.y < other.y

    def __str__(self): return "{} {}".format(self.x, self.y)
    # def __str__(self): return "(x = {:20}, y = {:20})".format(self.x, self.y)
    def __round__(self, n): return Pt2d(round(self.x, n), round(self.y, n))
    def __hash__(self): return hash((self.x, self.y))

    def get_tup(self): return self.x, self.y


class Pt3d:
    def __init__(self, x_val, y_val, z_val): 
        self.x, self.y, self.z = map(float, (x_val, y_val, z_val))

    def __add__(self, other): return Pt3d(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other): return Pt3d(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, scale): return Pt3d(self.x * scale, self.y * scale, self.z * scale)
    def __truediv__(self, scale): return Pt3d(self.x / scale, self.y / scale, self.z / scale)
    def __floordiv__(self, scale): return Pt3d(self.x // scale, self.y // scale, self.z // scale)

    def __eq__(self, other): 
        return isclose(self.x, other.x) and isclose(self.y, other.y) and isclose(self.z, other.z)
    def __lt__(self, other):
        return False if self == other else (self.x, self.y, self.z) < (other.x, other.y, other.y)

    def __str__(self): return "{} {} {}".format(self.x, self.y, self.z)
    # def __str__(self): return "(x = {:20}, y = {:20}), z = {:20})".format(self.x, self.y, self.z)
    def __round__(self, n): return Pt3d(round(self.x, n), round(self.y, n), round(self.z, n))
    def __hash__(self): return hash((self.x, self.y, self.z))

class QuadEdge:
    __slots__ = ("origin", "rot", "o_next", "used")
    def __init__(self):
        self.origin = None
        self.rot = None
        self.o_next = None
        self.used = False

    def rev(self): return self.rot.rot
    def l_next(self): return self.rot.rot.rot.o_next.rot
    def o_prev(self): return self.rot.o_next.rot
    def dest(self): return self.rot.rot.origin

class QuadEdgeDataStructure:
    def __init__(self):
        pass

    def make_edge(self, in_pt, out_pt):
        e1 = QuadEdge()
        e2 = QuadEdge()
        e3 = QuadEdge()
        e4 = QuadEdge()
        e1.origin = in_pt
        e2.origin = out_pt
        e3.origin = None
        e4.origin = None
        e1.rot = e3
        e2.rot = e4
        e3.rot = e2
        e4.rot = e1
        e1.o_next = e1
        e2.o_next = e2
        e3.o_next = e4
        e4.o_next = e3
        return e1

    def splice(self, a, b):
        a.o_next.rot.o_next, b.o_next.rot.o_next =  b.o_next.rot.o_next, a.o_next.rot.o_next
        a.o_next, b.o_next = b.o_next, a.o_next

    def delete_edge(self, edge):
        self.splice(edge, edge.o_prev())
        self.splice(edge.rev(), edge.rev().o_prev())
        # del edge.rot.rot.rot
        # del edge.rot.rot
        # del edge.rot
        # del edge

    def connect(self, a, b):
        e = self.make_edge(a.dest(), b.origin)
        self.splice(e, a.l_next())
        self.splice(e.rev(), b)
        return e

class GeometryAlgorithms:
    def __init__(self):
        self.x_ordering = None
        self.quad_edges = QuadEdgeDataStructure()

    def compare_ab(self, a, b):
        """Compare a b, for floats and ints. It is useful when you want set values to observe.
        paste directly into code and drop isclose for runtime speedup."""
        return 0 if isclose(a, b) else -1 if a<b else 1

    def dot_product(self, a, b):
        """Compute the scalar product a.b of a,b equivalent to: a . b"""
        return a.x*b.x + a.y*b.y
    def cross_product(self, a, b):
        """Computes the scalar value perpendicular to a,b equivalent to: a x b"""
        return a.x*b.y - a.y*b.x

    def distance_normalized(self, a, b):
        """Normalized distance between two points a, b equivalent to: sqrt(a^2 + b^2) = distance."""
        return dist(a.get_tup(), b.get_tup())
    def distance(self, a, b):
        """Squared distance between two points a, b equivalent to: a^2 + b^2 = distance."""
        return self.dot_product(a - b, a - b)

    def rotate_cw_90_wrt_origin(self, pt):
        """Compute a point rotation on pt. Just swap x and y and negate x."""
        return Pt2d(pt.y, -pt.x)

    def rotate_ccw_90_wrt_origin(self, pt):
        """Compute a point rotation on pt. Just swap x and y and negate y."""
        return Pt2d(-pt.y, pt.x)

    def rotate_ccw_rad_wrt_origin(self, pt, rad):
        """Compute a counterclockwise point rotation on pt. Accurate only for floating point cords.
        formula: x = (x cos(rad) - y sin(rad)), y = (x sin(rad) + y cos (rad)).

        Complexity per call: Time: O(1), Space: O(1).
        Optimizations: calculate cos and sin outside the return, so you don't double call each.
        """
        return Pt2d(pt.x * cos(rad) - pt.y * sin(rad),
                    pt.x * sin(rad) + pt.y * cos(rad))

    def point_c_rotation_wrt_line_ab(self, a, b, c):
        """Determine orientation of c wrt line ab, in terms of collinear clockwise counterclockwise.
        Since 2d cross-product is the area of the parallelogram, we can use this to accomplish this.

        Complexity per call: Time: O(1), Space: O(1).
        Returns collinear(cl): 0, counterclockwise(ccw): 1, clockwise(cw): -1
        Optimizations: if x,y are ints, use 0 instead of 0.0 or just paste the code here directly.
        """
        return self.compare_ab(self.cross_product(b - a, c - a), 0.0)

    def angle_point_c_wrt_line_ab(self, a, b, c):
        """For a line ab and point c, determine the angle of a to b to c in radians.
        formula: arc-cos(dot(vec_ab, vec_cb) / sqrt(dist_sq(vec_ab) * dist_sq(vec_cb))) = angle

        Complexity per call: Time: O(1), Space: O(1).
        Optimizations: for accuracy we sqrt both distances can remove if distances are ints.
        """
        vector_ab, vector_cb = a-b, c-b
        dot_ab_cb = self.dot_product(vector_ab, vector_cb)
        dist_sq_ab = self.dot_product(vector_ab, vector_ab)
        dist_sq_cb = self.dot_product(vector_cb, vector_cb)
        return acos(dot_ab_cb / (sqrt(dist_sq_ab) * sqrt(dist_sq_cb)))
        # return acos(dot_ab_cb / sqrt(dist_sq_ab * dist_sq_cb))

    def project_pt_c_to_line_ab(self, a, b, c):
        """Compute the point closest to c on the line ab.
        formula: pt = a + u x vector_ba, where u is the scalar projection of vector_ca onto
        vector_ba via dot-product

        Complexity per call: Time: O(1), Space: O(1).
        """
        vec_ba, vec_ca = b-a, c-a
        return a + vec_ba*(self.dot_product(vec_ca, vec_ba) / self.dot_product(vec_ba, vec_ba))

    def project_pt_c_to_line_seg_ab(self, a, b, c):
        """Compute the point closest to c on the line segment ab.
        Rule if a==b, then if c closer to a or b, otherwise we can just use the line version.

        Complexity per call: Time: O(1), Space: O(1).
        Optimizations: use compare_ab on the last line if needed better accuracy.
        """
        vec_ba, vec_ca = b-a, c-a
        dist_sq_ba = self.dot_product(vec_ba, vec_ba)
        if self.compare_ab(dist_sq_ba, 0.0) == 0: # a == b return either, maybe turn into a==b??
            return a
        u = self.dot_product(vec_ca, vec_ba) / dist_sq_ba
        return a if u < 0.0 else b if u > 1.0 else self.project_pt_c_to_line_ab(a, b, c)

    def distance_pt_c_to_line_ab(self, a, b, c):
        """Just return the distance between c and the projected point :)."""
        return self.distance_normalized(c, self.project_pt_c_to_line_ab(a, b, c))

    def distance_pt_c_to_line_seg_ab(self, a, b, c):
        """Same as above, just return the distance between c and the projected point :)."""
        return self.distance_normalized(c, self.project_pt_c_to_line_seg_ab(a, b, c))
    
    def is_parallel_lines_ab_and_cd(self, a, b, c, d):
        """Two lines are parallel if the cross_product between vec_ba and vec_cd is 0."""
        vec_ba, vec_cd = b - a, c - d
        return self.compare_ab(self.cross_product(vec_ba, vec_cd), 0.0) == 0

    def is_collinear_lines_ab_and_cd_1(self, a, b, c, d):
        """Old function. a!=b and c!=d and then returns correctly"""
        return (self.is_parallel_lines_ab_and_cd(a, b, c, d)
                and self.is_parallel_lines_ab_and_cd(b, a, a, c)
                and self.is_parallel_lines_ab_and_cd(d, c, c, a))

    def is_collinear_lines_ab_and_cd_2(self, a, b, c, d):
        """Two lines are collinear iff a!=b and c!=d, and both c and d are collinear to line ab."""
        return (self.point_c_rotation_wrt_line_ab(a, b, c) == 0
                and self.point_c_rotation_wrt_line_ab(a, b, d) == 0)

    def is_segments_intersect_ab_to_cd(self, a, b, c, d):
        """4 distinct points as two lines intersect if they are collinear and at least one of the
         end points c or d are in between a and b otherwise, TODO"""
        if self.is_collinear_lines_ab_and_cd_2(a, b, c, d):
            lo, hi = (a, b) if a < b else (b, a)
            return lo <= c <= hi or lo <= d <= hi
        a_val = self.cross_product(d - a, b - a) * self.cross_product(c - a, b - a)
        c_val = self.cross_product(a - c, d - c) * self.cross_product(b - c, d - c)
        return not(a_val>0 or c_val>0)

    def is_lines_intersect_ab_to_cd(self, a, b, c, d):
        """Two lines intersect if they aren't parallel or if they collinear."""
        return (not self.is_parallel_lines_ab_and_cd(a, b, c, d)
                or self.is_collinear_lines_ab_and_cd_2(a, b, c, d))

    def pt_lines_intersect_ab_to_cd(self, a, b, c, d):
        """Compute the intersection point between two lines.
        Explain TODO
        """
        vec_ba, vec_ca, vec_cd = b-a, c-a, c-d
        return a + vec_ba*(self.cross_product(vec_ca, vec_cd) / self.cross_product(vec_ba, vec_cd))

    def pt_line_seg_intersect_ab_to_cd(self, a, b, c, d):
        """Same as for line intersect but this time we need to use a specific formula.
        Formula: TODO"""
        x, y, cross_prod = c.x-d.x, d.y-c.y, self.cross_product(d, c)
        u = abs(y*a.x + x*a.y + cross_prod)
        v = abs(y*b.x + x*b.y + cross_prod)
        return Pt2d((a.x * v + b.x * u) / (v + u), (a.y * v + b.y * u) / (v + u))

    def is_point_p_in_circle_c_radius_r(self, p, c, r):
        """Computes True if point p in circle False otherwise. Use <= for circumference inclusion"""
        return self.compare_ab(self.distance_normalized(p, c), r) < 0

    def pt_circle_center_given_pt_abc(self, a, b, c):
        """Find the center of a circle based of 3 distinct points
        TODO add in teh formula
        """
        ab, ac = (a+b)/2, (a+c)/2
        ab_rot = self.rotate_cw_90_wrt_origin(a - ab) + ab
        ac_rot = self.rotate_cw_90_wrt_origin(a - ac) + ac
        return self.pt_lines_intersect_ab_to_cd(ab, ab_rot, ac, ac_rot)

    def pts_line_ab_intersects_circle_cr(self, a, b, c, r):
        """Compute the point(s) that line ab intersects circle c radius r. from stanford 2016
        TODO add in the formula
        """
        vec_ba, vec_ac = b-a, a-c
        dist_sq_ba = self.dot_product(vec_ba, vec_ba)
        dist_sq_ac_ba = self.dot_product(vec_ac, vec_ba)
        dist_sq_ac = self.dot_product(vec_ac, vec_ac) - r * r
        dist_sq = dist_sq_ac_ba*dist_sq_ac_ba - dist_sq_ba*dist_sq_ac
        result = self.compare_ab(dist_sq, 0.0)
        if result >= 0:
            first_intersect = c + vec_ac + vec_ba*(-dist_sq_ac_ba + sqrt(dist_sq + EPS))/dist_sq_ba
            second_intersect = c + vec_ac + vec_ba*(-dist_sq_ac_ba - sqrt(dist_sq))/dist_sq_ba
            return first_intersect if result == 0 else first_intersect, second_intersect
        return None # no intersect 

    def pts_two_circles_intersect_cr1_cr2(self, c1: Pt2d, c2: Pt2d, r1, r2):
        """I think this is the points on the circumference but not fully sure. from stanford 2016
        TODO add in teh formula
        """
        center_dist = self.distance_normalized(c1, c2)
        if (self.compare_ab(center_dist, r1 + r2) <= 0
                <= self.compare_ab(center_dist + min(r1, r2), max(r1, r2))):
            x = (center_dist*center_dist - r2*r2 + r1*r1)/(2*center_dist)
            y = sqrt(r1*r1 - x*x)
            v = (c2-c1)/center_dist
            pt1, pt2 = c1 + v * x, self.rotate_ccw_90_wrt_origin(v) * y
            return (pt1 + pt2) if self.compare_ab(y, 0.0) <= 0 else (pt1+pt2, pt1-pt2)
        return None # no overlap

    def pt_tangent_to_circle_cr(self, c, r, p):
        """Find the two points that create tangent lines from p to the circumference.
        TODO add in teh formula
        """
        vec_pc = p-c
        x = self.dot_product(vec_pc, vec_pc)
        dist_sq = x - r*r
        result = self.compare_ab(dist_sq, 0.0)
        if result >= 0:
            dist_sq = dist_sq if result else 0
            q1 = vec_pc * (r*r / x)
            q2 = self.rotate_ccw_90_wrt_origin(vec_pc * (-r * sqrt(dist_sq) / x))
            return [c+q1-q2, c+q1+q2]
        return []

    def tangents_between_2_circles(self, c1, r1, c2, r2):
        """Between two circles there should be at least 4 points that make two tangent lines.
        TODO add in teh formula
        """
        r_tangents = []
        if self.compare_ab(r1, r2) == 0:
            c2c1 = c2 - c1
            multiplier = r1/sqrt(self.dot_product(c2c1, c2c1))
            tangent = self.rotate_ccw_90_wrt_origin(c2c1 * multiplier) # need better name
            r_tangents = [(c1+tangent, c2+tangent), (c1-tangent, c2-tangent)]
        else:
            ref_pt = ((c1 * -r2) + (c2 * r1)) / (r1 - r2)
            ps = self.pt_tangent_to_circle_cr(c1, r1, ref_pt)
            qs = self.pt_tangent_to_circle_cr(c2, r2, ref_pt)
            r_tangents = [(ps[i], qs[i]) for i in range(min(len(ps), len(qs)))]
        ref_pt = ((c1 * r2) + (c2 * r1)) / (r1 + r2)
        ps = self.pt_tangent_to_circle_cr(c1, r1, ref_pt)
        qs = self.pt_tangent_to_circle_cr(c2, r2, ref_pt)
        for i in range(min(len(ps), len(qs))):
            r_tangents.append((ps[i], qs[i]))
        return r_tangents

    def sides_of_triangle_abc(self, a, b, c):
        """Compute the side lengths of a triangle."""
        dist_ab = self.distance_normalized(a, b)
        dist_bc = self.distance_normalized(b, c)
        dist_ca = self.distance_normalized(c, a)
        return dist_ab, dist_bc, dist_ca

    def pt_p_in_triangle_abc(self, a, b, c, p):
        """Compute if a point is in or on a triangle. If all edges return the same orientation this
        should return true and the point should be in or on the triangle."""
        return (self.point_c_rotation_wrt_line_ab(a, b, p) >= 0
                and self.point_c_rotation_wrt_line_ab(b, c, p) >= 0
                and self.point_c_rotation_wrt_line_ab(c, a, p) >= 0)

    def perimeter_of_triangle_abc(self, side_ab, side_bc, side_ca):
        """Computes the perimeter of triangle given the side lengths."""
        return side_ab + side_bc + side_ca

    def triangle_area_bh(self, base, height):
        """Simple triangle area formula: area = b*h/2."""
        return base*height/2

    def triangle_area_from_heron_abc(self, side_ab, side_bc, side_ca):
        """Compute heron's formula which gives us the area of a triangle given the side lengths."""
        s = self.perimeter_of_triangle_abc(side_ab, side_bc, side_ca) / 2
        return sqrt(s * (s-side_ab) * (s-side_bc) * (s-side_ca))

    def triangle_area_from_cross_product_abc(self, a, b, c):
        """Compute triangle area, via cross-products of the pairwise sides ab, bc, ca."""
        return (self.cross_product(a, b) + self.cross_product(b, c) + self.cross_product(c, a))/2

    # def incircle_radis_of_triangle_abc_helper(self, ab, bc, ca):
    #     area = self.triangle_area_from_heron_abc(ab, bc, ca)
    #     perimeter = self.perimeter_of_triangle_abc(ab, bc, ca) / 2
    #     return area/perimeter

    def incircle_radius_of_triangle_abc(self, a, b, c):
        """Computes the radius of the incircle, achieved by computing the side lengths then finding
        the area and perimeter to use in this Formula: r = area/(perimeter/2) Author: TODO
        """
        side_ab, side_bc, side_ca = self.sides_of_triangle_abc(a, b, c)
        area = self.triangle_area_from_heron_abc(side_ab, side_bc, side_ca)
        perimeter = self.perimeter_of_triangle_abc(side_ab, side_bc, side_ca) / 2
        return area / perimeter

    # def circumcircle_radis_of_triangle_abc_helper(self, ab, bc, ca):
    #     area = self.triangle_area_from_heron_abc(ab, bc, ca)
    #     return (ab*bc*ca) / (4*area)
        
    def circumcircle_radius_of_triangle_abc(self, a, b, c):
        """Computes the radius of the circum-circle, achieved by computing the side lengths then
        gets the area for Formula: r = (ab * bc * ca) / (4 * area) Author: TODO
        """
        side_ab, side_bc, side_ca = self.sides_of_triangle_abc(a, b, c)
        area = self.triangle_area_from_heron_abc(side_ab, side_bc, side_ca)
        return (side_ab * side_bc * side_ca) / (4 * area)

    def incircle_pt_for_triangle_abc_1(self, a, b, c):
        """Get the circle center of an incircle.

        Complexity per call: Time: lots of ops but still O(1), Space O(1)
        Formula: TODO
        Optimization: get sides individually instead of through another call
        """
        radius = self.incircle_radius_of_triangle_abc(a, b, c)
        if self.compare_ab(radius, 0.0) == 0: #  if the radius was 0 we don't have a point
            return False, 0, 0
        side_ab, side_bc, side_ca = self.sides_of_triangle_abc(a, b, c)
        ratio_1 = side_ab/side_ca
        ratio_2 = side_ab/side_bc
        pt_1 = b + (c-b) * (ratio_1/(ratio_1 + 1.0)) 
        pt_2 = a + (c-a) * (ratio_2/(ratio_2 + 1.0))

        if self.is_lines_intersect_ab_to_cd(a, pt_1, b, pt_2):
            intersection_pt = self.pt_lines_intersect_ab_to_cd(a, pt_1, b, pt_2)
            return True, radius, round(intersection_pt, 12)  # can remove the round function
        return False, 0, 0

    def triangle_circle_center_pt_abcd(self, a, b, c, d):
        """A 2 in one method that can get the middle point of both incircle circumcenter.
        Method: TODO

        Complexity per call: Time: lots of ops but still O(1), Space O(1)
        Optimization: paste rotation code instead of function call
        """
        pt_1 = self.rotate_cw_90_wrt_origin(b-a)  # rotation on the vector b-a
        pt_2 = self.rotate_cw_90_wrt_origin(d-c)  # rotation on the vector d-c
        cross_product_1_2 = self.cross_product(pt_1, pt_2)
        # cross_product_2_1 = -cross_product_1_2  # self.cross_product(pt_2, pt_1)
        if self.compare_ab(cross_product_1_2, 0.0) == 0:
            return None
        pt_3 = Pt2d(self.dot_product(a, pt_1), self.dot_product(c, pt_2))
        x = ((pt_3.x * pt_2.y) - (pt_3.y * pt_1.y)) / cross_product_1_2
        y = ((pt_3.x * pt_2.x) - (pt_3.y * pt_1.x)) / -cross_product_1_2  # cross(pt_2, pt_1)
        return round(Pt2d(x, y), 12)

    def angle_bisector_for_triangle_abc(self, a, b, c):
        """Compute the angle bisector point.
        Method: TODO
        """
        dist_ba = self.distance_normalized(b, a)
        dist_ca = self.distance_normalized(c, a)
        ref_pt = (b-a) / dist_ba * dist_ca
        return ref_pt + (c-a) + a

    def perpendicular_bisector_for_triangle_ab(self, a, b):
        """Compute the perpendicular bisector point.
        Method: TODO
        """
        rotated_vector_ba = self.rotate_ccw_90_wrt_origin(b-a)  # code is a ccw turn. check formula
        return rotated_vector_ba + (a+b)/2

    def incircle_pt_for_triangle_abc_2(self, a, b, c):
        """An alternative way to compute incircle. This one uses bisectors
        Method: TODO
        """
        bisector_abc = self.angle_bisector_for_triangle_abc(a, b, c)
        bisector_bca = self.angle_bisector_for_triangle_abc(b, c, a)
        return self.triangle_circle_center_pt_abcd(a, bisector_abc, b, bisector_bca)

    def circumcenter_pt_of_triangle_abc_2(self, a, b, c):
        """An alternative way to compute circumcenter. This one uses bisectors
        Method: TODO
        """
        bisector_ab = self.perpendicular_bisector_for_triangle_ab(a, b)
        bisector_bc = self.perpendicular_bisector_for_triangle_ab(b, c)
        ab2, bc2 = (a+b)/2, (b+c)/2
        return self.triangle_circle_center_pt_abcd(ab2, bisector_ab, bc2, bisector_bc)

    def orthocenter_pt_of_triangle_abc_v2(self, a, b, c):
        """Compute the orthogonal center of triangle abc.Z
        Method: TODO
        """
        return a + b + c - self.circumcenter_pt_of_triangle_abc_2(a, b, c) * 2

    def perimeter_of_polygon_pts(self, pts):
        """Compute summed pairwise perimeter of polygon in CCW ordering."""
        return fsum([self.distance_normalized(a, b) for a, b in pairwise(pts)])
        # return fsum([self.distance_normalized(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])

    def signed_area_of_polygon_pts(self, pts):
        """Compute sum of area of polygon, via shoelace method: half the sum of the pairwise
        cross-products."""
        return fsum([self.cross_product(a, b) for a, b in pairwise(pts)]) / 2
        # return fsum([self.cross_product(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]) / 2

    def area_of_polygon_pts(self, pts):
        """Positive area of polygon using above method."""
        return abs(self.signed_area_of_polygon_pts(pts))

    def is_convex_polygon_pts_no_collinear(self, pts):
        """Determines if polygon is convex, only works when no collinear lines.

        Complexity per call: Time: O(n), Space: O(1)
        """
        if len(pts) > 3:
            pts.append(pts[1])
            end, result = len(pts) - 2, True
            first_turn = self.point_c_rotation_wrt_line_ab(pts[0], pts[1], pts[2])
            for i in range(end):
                if self.point_c_rotation_wrt_line_ab(pts[i], pts[i+1], pts[i+2]) != first_turn:
                    result = False
                    break
            pts.pop()
            return result
        return False

    def is_convex_polygon_pts_has_collinear(self, pts):
        """Determines if polygon is convex, works with collinear but takes more time and space.

        Complexity per call: Time: O(n), Space: O(n)
        """
        if len(pts) > 3:
            pts.append(pts[1])
            end = len(pts) - 2
            rotations = [self.point_c_rotation_wrt_line_ab(pts[i], pts[i+1], pts[i+2])
                         for i in range(end)]
            tally = [0] * 3 # ccw cl cw only 3 types
            for el in rotations:
                tally[el + 1] += 1
            pts.pop()
            lo, hi = min(tally[0], tally[2]), max(tally[0], tally[2])
            return False if lo > 0 or hi == 0 else hi > 0
        return False

    def pt_p_in_polygon_pts_1(self, pts, p):
        """Determine if a point is in a polygon based on the sum of the angles.

        Complexity per call: Time: O(n), Space: O(1)
        """
        if len(pts) > 3:
            angle_sum = 0.0
            # for i in range(len(pts) - 1):  # a = pts[i], b = pts[i+1]
            for a, b in pairwise(pts):
                angle = self.angle_point_c_wrt_line_ab(a, b, p)
                if 1 == self.point_c_rotation_wrt_line_ab(a, b, p):
                    angle_sum += angle
                else:
                    angle_sum -= angle
            return self.compare_ab(abs(angle_sum), pi)
        return False

    def pt_p_in_polygon_pts_2(self, pts, p):
        """Determine if a point is in a polygon via, ray casting.

        Complexity per call: Time: O(n), Space: O(1)
        """
        ans = False
        px, py = p.get_tup()
        xi, yi, xj, yj = 0, 0, pts[0].x, pts[0].y
        for i in range(len(pts)-1):
            xi, yi = xj, yj
            xj, yj = pts[i+1].x, pts[i+1].y
            if (yi <= py < yj or yj <= py < yi) and px < (xi + (xj - xi) * (py - yi) / (yj - yi)):
                ans = not ans
        return ans

    def pt_p_on_polygon_perimeter_pts(self, pts, p):
        """Determine if a point is on the perimeter of a polygon simply via a distance check.

        Complexity per call: Time: O(n), Space: O(1)
        Optimizations: move old_dist and new_dist before loop and only call function on new_dist.
        """
        # for i in range(len(pts) - 1):  # a = pts[i], b = pts[i+1]
        for a, b in pairwise(pts):
            old_dist = self.distance_normalized(a, p)
            new_dist = self.distance_normalized(p, b)
            ij_dist = self.distance_normalized(a, b)
            if self.compare_ab(new_dist+old_dist, ij_dist) == 0:
                return True
        return p in pts

    def pt_p_in_convex_polygon_pts(self, pts, p):
        """For a convex Polygon we are able to search if point is in the polygon faster. TODO

        Complexity per call: Time: O(log n), Space: O(1)
        Optimizations:
        """
        n = len(pts)
        if n == 2:
            distance = self.distance_pt_c_to_line_seg_ab(pts[0], pts[1], p)
            return self.compare_ab(distance, 0.0) == 0
        left, right = 1, n
        while left < right:
            mid = (left + right)/2 + 1
            side = self.point_c_rotation_wrt_line_ab(pts[0], pts[mid], p)
            left, right = (mid, right) if side == 1 else (left, mid-1)
        side = self.point_c_rotation_wrt_line_ab(pts[0], pts[left], p)
        if side == -1 or left == n:
            return False
        side = self.point_c_rotation_wrt_line_ab(pts[left], pts[left + 1] - pts[left], p)
        return side >= 0
    
    # use a set with points if possible checking on the same polygon many times    
    # return 0 for on 1 for in -1 for out
    def pt_p_position_wrt_polygon_pts(self, pts, p):
        """Will determine if a point is in on or outside a polygon.

        Complexity per call: Time: O(n) Convex(log n), Space: O(1)
        Optimizations: use log n version if you need superfast, and it's a convex polygon
        """
        return (0 if self.pt_p_on_polygon_perimeter_pts(pts, p)
                else 1 if self.pt_p_in_polygon_pts_2(pts, p) else -1)

    def centroid_pt_of_convex_polygon(self, pts):
        """Compute the centroid of a convex polygon.

        Complexity per call: Time: O(n), Space: O(1)
        Optimizations:
        """
        ans, n = Pt2d(0, 0), len(pts)
        # for i in range(len(pts) - 1):  # a = pts[i], b = pts[i+1]
        for a, b in pairwise(pts):
            ans = ans + (a + b) * self.cross_product(a, b)
        return ans / (6.0 * self.signed_area_of_polygon_pts(pts))

    def is_polygon_pts_simple_quadratic(self, pts):
        """Brute force method to check if a polygon is simple. check all line pairs

        Complexity per call: Time: O(n^2), Space: O(1)
        Optimizations:
        """
        n = len(pts)
        for i in range(n-1):
            for k in range(i+1, n-1):
                j, l = (i+1) % n, (k+1) % n
                if i == l or j == k:
                    continue
                if self.is_segments_intersect_ab_to_cd(pts[i], pts[j], pts[k], pts[l]):
                    return False
        return True

    def polygon_cut_from_line_ab(self, pts, a, b):
        """Method computes the left side polygon resulting from a cut from the line a-b.
        Method: Walk around the polygon and only take points that return CCW to line ab

        Complexity per call: Time: O(n), Space: O(n)
        Optimizations:
        """
        left_partition = []
        # for i in range(len(pts) - 1):
        for u, v in pairwise(pts):
            rot_1 = self.point_c_rotation_wrt_line_ab(a, b, u)
            rot_2 = self.point_c_rotation_wrt_line_ab(a, b, v)
            if 0 >= rot_1:
                left_partition.append(u)
                if 0 == rot_1:
                    continue
            if rot_1 * rot_2 < 0: # CCW -1, CW 1 so tests if they are opposite ie lines intersect.
                left_partition.append(self.pt_line_seg_intersect_ab_to_cd(u, v, a, b))
        if left_partition and left_partition[0] != left_partition[-1]:
            left_partition.append(left_partition[0])
        return left_partition

    def convex_hull_monotone_chain(self, pts): # needs test
        """Compute convex hull of a list of points via Monotone Chain method. CCW ordering returned.

        Complexity per call: Time: O(nlog n), Space: final O(n), aux O(nlog n)
        Optimizations: can use heapsort for Space: O(n)[for set + heap] or O(1) [if we consume pts],
        Can also optimize out the append and pop with using a stack like index.
        """
        def func(points, cur_hull, min_size):
            for p in points:
                while (len(cur_hull) > min_size
                       and self.point_c_rotation_wrt_line_ab(cur_hull[-2], cur_hull[-1], p) == CW):
                    cur_hull.pop()
                cur_hull.append(p)
            cur_hull.pop()
        unique_points, convex_hull = sorted(set(pts)), []
        if len(unique_points) > 1:
            func(unique_points, convex_hull, 1)
            func(unique_points[::-1], convex_hull, 1 + len(convex_hull))
            return convex_hull
        return unique_points

    def rotating_caliper_of_polygon_pts(self, pts):
        """Computes the max distance of two points in the convex polygon?

        Complexity per call: Time: O(nlog n) unsorted O(n) sorted, Space: O(1)
        Optimizations:
        """
        convex_hull = self.convex_hull_monotone_chain(pts)
        n, t, ans = len(convex_hull) - 1, 1, 0.0
        for i in range(n):
            p_i, p_j = convex_hull[i], convex_hull[(i + 1) % n]
            p = p_j - p_i
            while (t + 1) % n != i:
                if (self.cross_product(p, convex_hull[(t + 1) % n] - p_i) <
                        self.cross_product(p, convex_hull[t] - p_i)):
                    break
                t = (t + 1) % n
            ans = max(ans, self.distance(p_i, convex_hull[t]))
            ans = max(ans, self.distance(p_j, convex_hull[t]))
        return sqrt(ans)

    def closest_pair_helper(self, lo, hi):
        """brute force function, for small range will brute force find the closet pair. O(n^2)"""
        r_closest = (self.distance(self.x_ordering[lo], self.x_ordering[lo + 1]),
                     self.x_ordering[lo], 
                     self.x_ordering[lo+1])
        for i in range(lo, hi):
            for j in range(i+1, hi):
                distance_ij = self.distance(self.x_ordering[i], self.x_ordering[j])
                if self.compare_ab(distance_ij, r_closest) < 0:
                    r_closest = (distance_ij, self.x_ordering[i], self.x_ordering[j])
        return r_closest

    def closest_pair_recursive(self, lo, hi, y_ordering):
        """Recursive part of computing the closest pair. Divide by y recurse then do a special check

        Complexity per call T(n/2) halves each time, T(n/2) halves each call, O(n) at max tho
        Optimizations and notes: If not working use y_part_left and y_part_right again. I did add in
        the optimization of using y_partition 3 times over rather than having 3 separate lists
        """
        n = hi - lo
        if n < 5: # base case we brute force the small set of points
            return self.closest_pair_helper(lo, hi)
        left_len, right_len = lo + n - n//2, lo + n//2
        mid = round((self.x_ordering[left_len].x + self.x_ordering[right_len].x)/2)
        # y_part_left = [el for el in y_ordering if self.compare_ab(el.x, mid) <= 0]
        # y_part_right = [el for el in y_ordering if self.compare_ab(el.x, mid) > 0]
        y_bools = [self.compare_ab(point.x, mid) <= 0 for point in y_ordering]
        y_partition = [point for i, point in enumerate(y_ordering) if y_bools[i]]
        best_left = self.closest_pair_recursive(lo, left_len, y_partition)
        if self.compare_ab(best_left[0], 0.0) == 0:
            return best_left
        y_partition = [point for i, point in enumerate(y_ordering) if not y_bools[i]]
        best_right = self.closest_pair_recursive(left_len, hi, y_partition)
        if self.compare_ab(best_right[0], 0.0) == 0:
            return best_right
        best_pair = best_left if self.compare_ab(best_left[0], best_right[0]) <= 0 else best_right
        y_partition = [point for point in y_ordering
                       if self.compare_ab((point.x-mid) * (point.x-mid), best_pair[0]) < 0]
        y_end = len(y_partition)
        for i in range(y_end):
            for j in range(i+1, y_end):
                dist_ij = y_partition[i].y - y_partition[j].y
                if self.compare_ab(dist_ij * dist_ij, best_pair[0]) > 0:
                    break
                dist_ij = self.distance(y_partition[i], y_partition[j])
                if self.compare_ab(dist_ij, best_pair) < 0:
                    best_pair = (dist_ij, y_partition[i], y_partition[j])
        return best_pair

    def compute_closest_pair(self, pts):
        """Compute the closest pair of points in a set of points. method is divide and conqur

        Complexity per call Time: O(nlog n), Space O(nlog n)
        Optimizations: use c++ if too much memory, haven't found the way to do it without nlog n
        """
        self.x_ordering = sorted(pts, key=lambda point: point.x)
        y_ordering = sorted(pts, key=lambda point: point.y)
        return self.closest_pair_recursive(0, len(pts), y_ordering)

    def delaunay_triangulation_slow(self, pts):
        """A very slow version of  Delaunay Triangulation. Can beat the faster version when n small.

        Complexity per call Time: O(n^4), Space O(n)
        Optimizations: use c++ if too much memory, haven't found the way to do it without nlog n
        """
        n = len(pts)
        ans = []
        z_arr = [el.x ** 2 + el.y ** 2 for el in pts]
        x_arr = [el.x for el in pts]
        y_arr = [el.y for el in pts]
        for i in range(n - 2):
            for j in range(i + 1, n):
                for k in range(i + 1, n):
                    if j == k:
                        continue
                    xn = (y_arr[j] - y_arr[i]) * (z_arr[k] - z_arr[i]) - (y_arr[k] - y_arr[i]) * (z_arr[j] - z_arr[i])
                    yn = (x_arr[k] - x_arr[i]) * (z_arr[j] - z_arr[i]) - (x_arr[j] - x_arr[i]) * (z_arr[k] - z_arr[i])
                    zn = (x_arr[j] - x_arr[i]) * (y_arr[k] - y_arr[i]) - (x_arr[k] - x_arr[i]) * (y_arr[j] - y_arr[i])
                    flag = zn < 0.0
                    for m in range(n):
                        if flag:
                            flag = flag and (self.compare_ab((x_arr[m] - x_arr[i]) * xn +
                                                             (y_arr[m] - y_arr[i]) * yn +
                                                             (z_arr[m] - z_arr[i]) * zn, 0.0) <= 0)
                        else:
                            break
                    if flag:
                        ans.append((pts[i], pts[j], pts[k]))
        return ans

    def pt_left_of_edge(self, pt, edge):
        """A helper function with a name to describe the action. Remove for speedup."""
        return CCW == self.point_c_rotation_wrt_line_ab(pt, edge.origin, edge.dest())

    def pt_right_of_edge(self, pt, edge):
        """A helper function with a name to describe the action. Remove for speedup."""
        return CW == self.point_c_rotation_wrt_line_ab(pt, edge.origin, edge.dest())

    def det3_helper(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):
        """A helper function for determining the angle. Remove for speedup."""
        return (a1 * (b2 * c3 - c2 * b3) -
                a2 * (b1 * c3 - c1 * b3) +
                a3 * (b1 * c2 - c1 * b2))

    def is_in_circle(self, a, b, c, d):
        """Expensive caclution function that determines if """
        a_dot = self.dot_product(a, a)
        b_dot = self.dot_product(b, b)
        c_dot = self.dot_product(c, c)
        d_dot = self.dot_product(d, d)
        det = -self.det3_helper(b.x, b.y, b_dot, c.x, c.y, c_dot, d.x, d.y, d_dot)
        det += self.det3_helper(a.x, a.y, a_dot, c.x, c.y, c_dot, d.x, d.y, d_dot)
        det -= self.det3_helper(a.x, a.y, a_dot, b.x, b.y, b_dot, d.x, d.y, d_dot)
        det += self.det3_helper(a.x, a.y, a_dot, b.x, b.y, b_dot, c.x, c.y, c_dot)
        return det > 0
        # use this if above doesn't work for what ever reason
        # def angle(l, mid, r):
        #     x = self.dot_product(l-mid, r-mid)
        #     y = self.cross_product(l-mid, r-mid)
        #     return atan2(x, y)
        # kek = angle(a, b, c) + angle(c, d, a) - angle(b, c, d) - angle(d, a, b)
        # return self.compare_ab(kek, 0.0) > 0

    def build_triangulation(self, l, r, pts):
        if r - l + 1 == 2:
            res = self.quad_edges.make_edge(pts[l], pts[r])
            return res, res.rev()
        if r - l + 1 == 3:
            edge_a = self.quad_edges.make_edge(pts[l], pts[l + 1])
            edge_b = self.quad_edges.make_edge(pts[l + 1], pts[r])
            self.quad_edges.splice(edge_a.rev(), edge_b)
            sg = self.point_c_rotation_wrt_line_ab(pts[l], pts[l + 1], pts[r])
            if sg == 0:
                return edge_a, edge_b.rev()
            edge_c = self.quad_edges.connect(edge_b, edge_a)
            return (edge_a, edge_b.rev()) if sg == 1 else (edge_c.rev(), edge_c)
        mid = (l + r) // 2
        ldo, ldi = self.build_triangulation(l, mid, pts)
        rdi, rdo = self.build_triangulation(mid + 1, r, pts)
        while True:
            if self.pt_left_of_edge(rdi.origin, ldi):
                ldi = ldi.l_next()
                continue
            if self.pt_right_of_edge(ldi.origin, rdi):
                rdi = rdi.rev().o_next
                continue
            break
        base_edge_l = self.quad_edges.connect(rdi.rev(), ldi)
        if ldi.origin == ldo.origin:
            ldo = base_edge_l.rev()
        if rdi.origin == rdo.origin:
            rdo = base_edge_l
        while True:
            l_cand_edge = base_edge_l.rev().o_next
            if self.pt_right_of_edge(l_cand_edge.dest(), base_edge_l):
                while self.is_in_circle(base_edge_l.dest(), base_edge_l.origin,
                                        l_cand_edge.dest(), l_cand_edge.o_next.dest()):
                    temp_edge = l_cand_edge.o_next
                    self.quad_edges.delete_edge(l_cand_edge)
                    l_cand_edge = temp_edge
            r_cand_edge = base_edge_l.o_prev()
            if self.pt_right_of_edge(r_cand_edge.dest(), base_edge_l):
                while self.is_in_circle(base_edge_l.dest(), base_edge_l.origin,
                                        r_cand_edge.dest(), r_cand_edge.o_prev().dest()):
                    temp_edge = r_cand_edge.o_prev()
                    self.quad_edges.delete_edge(r_cand_edge)
                    r_cand_edge = temp_edge
            l_check = self.pt_right_of_edge(l_cand_edge.dest(), base_edge_l)
            r_check = self.pt_right_of_edge(r_cand_edge.dest(), base_edge_l)
            if (not l_check) and (not r_check):
                break
            if ((not l_check)
                    or (r_check
                        and self.is_in_circle(l_cand_edge.dest(), l_cand_edge.origin,
                                              r_cand_edge.origin, r_cand_edge.dest()))):
                base_edge_l = self.quad_edges.connect(r_cand_edge, base_edge_l.rev())
            else:
                base_edge_l = self.quad_edges.connect(base_edge_l.rev(), l_cand_edge.rev())
        return ldo, rdo
            
    def delaunay_triangulation_fast(self, pts):
        pts.sort()
        result = self.build_triangulation(0, len(pts) - 1, pts)
        edge = result[0]
        edges = [edge]
        while self.point_c_rotation_wrt_line_ab(edge.o_next.dest(), edge.dest(), edge.origin) < CL:
            edge = edge.o_next
        def add_helper():
            cur = edge
            while True:
                cur.used = True
                pts.append(cur.origin)
                edges.append(cur.rev())
                cur = cur.l_next()
                if cur == edge:
                    return
        add_helper()
        pts = []
        kek = 0
        while kek < len(edges):
            edge = edges[kek]
            kek += 1
            if not edge.used:
                add_helper()
        ans = [tuple((sorted([pts[i], pts[i + 1], pts[i + 2]]))) for i in range(0, len(pts), 3)]
        return sorted(list(set(ans)))


####################################################################################################


from itertools import takewhile, pairwise


def pairwise_func(seq):
    it = iter(seq); next(it)
    return zip(iter(seq), it)


# constants can paste into code for speedup
GREATER_EQUAL = -1
GREATER_THAN = 0


class StringAlgorithms:
    def __init__(self):
        self.math_algos = None
        self.text_len = 0
        self.pattern_len = 0
        self.n = 0
        self.prime_p = self.mod_m = 0
        self.text = ''
        self.pattern = ''
        self.back_table = []
        self.suffix_array = []
        self.text_ord = []
        self.pattern_ord = []
        self.longest_common_prefix = []
        self.owner = []
        self.seperator_list = []
        self.hash_powers = []
        self.hash_h_values = []
        self.left_mod_inverse = []

    def kmp_preprocess(self, new_pattern):
        """Preprocess the pattern for KMP. TODO add a bit more to this description ?

        Complexity per call: Time O(m + m), Space: O(m)
        """
        pattern = new_pattern
        pattern_len = len(pattern)  # m = length of pattern
        back_table = [0] * (pattern_len + 1)
        back_table[0], j = -1, -1
        for i, character in enumerate(pattern):
            while j >= 0 and character != pattern[j]:
                j = back_table[j]
            j += 1
            back_table[i + 1] = j
        self.pattern = pattern
        self.pattern_len = pattern_len
        self.back_table = back_table

    def kmp_search_find_indices(self, text_to_search):
        """Search the text for the pattern we preprocessed.

        Complexity per call: Time O(n + m), Space: O(n + m)
        """
        ans, j = [], 0
        for i, character in enumerate(text_to_search):
            while j >= 0 and character != self.pattern[j]:
                j = self.back_table[j]
            j += 1
            if j == self.pattern_len:
                ans.append(1 + i - j)
                j = self.back_table[j]
        return ans

    def suffix_array_counting_sort(self, k, s_array, r_array):
        """Basic count sort for the radix sorting part of suffix arrays.

        Complexity per call. Time: O(n), T(6n), Space: O(n), S(2n)
        Switching to non itertools versions and non enumerating version can speed it up.
        """
        n = self.text_len
        maxi, tmp = max(255, n), 0
        suffix_array_temp, frequency_array = [0] * n, [0] * maxi
        frequency_array[0] = n - (n - k)  # allows us to skip k values, handled in the second loop
        for i in range(k, n):             # here we skip those k iterations
            frequency_array[r_array[i]] += 1
        for i in range(maxi):
            frequency_array[i], tmp = tmp, tmp + frequency_array[i]
        for suffix_i in s_array:
            pos = 0 if suffix_i + k >= n else r_array[suffix_i + k]
            suffix_array_temp[frequency_array[pos]] = suffix_i
            frequency_array[pos] += 1
        for i, value in enumerate(suffix_array_temp):
            s_array[i] = value

    def suffix_array_build_array(self, new_texts):
        """Suffix array construction on a list of texts. n = sum lengths of all the texts.

        Complexity per call: Time: O(nlog n), T(3n log n), Space: O(n), S(6n)
        Optimizations: remove take while, don't use list(map(ord, text)), remove the pairwise
        """
        num_strings = len(new_texts)
        new_text = ''.join([txt + chr(num_strings - i) for i, txt in enumerate(new_texts)])
        self.text_len = new_len = len(new_text)
        suffix_arr, rank_arr = [i for i in range(new_len)], [ord(c) for c in new_text]
        for power in takewhile(lambda x: 2 ** x < new_len, range(32)):  # iterate powers of 2
            k = 2 ** power
            self.suffix_array_counting_sort(k, suffix_arr, rank_arr)
            self.suffix_array_counting_sort(0, suffix_arr, rank_arr)
            rank_array_temp = [0] * new_len
            rank = 0
            for last, curr in pairwise_func(suffix_arr):  # suffix[i] = curr, suffix[i - 1] last
                rank = rank if (rank_arr[curr] == rank_arr[last]
                                and rank_arr[curr + k] == rank_arr[last + k]) else rank + 1
                rank_array_temp[curr] = rank
            rank_arr = rank_array_temp
            if rank_arr[suffix_arr[-1]] == new_len - 1:  # exit loop early optimization
                break
        self.suffix_array, self.text = suffix_arr, new_text
        self.text_ord = [ord(c) for c in new_text]  # optional used in the binary search
        self.seperator_list = [num_strings - i for i in range(len(new_texts))]  # optional owners

    def compute_longest_common_prefix(self):
        """After generating a suffix array you can use that to find the longest common pattern.

        Complexity per call: Time: O(n), T(4n), Space: O(n), S(3n)
        """
        local_suffix_array = self.suffix_array  # optional, avoids expensive load_attr operation
        local_text_len = self.text_len          # ignore for faster implementations, and place it
        local_text_ord = self.text_ord          # directly in the code
        permuted_lcp, phi = [0] * local_text_len, [0] * local_text_len
        phi[0], left = -1, 0
        for last, curr in pairwise_func(local_suffix_array):
            phi[curr] = last
        for i, phi_i in enumerate(phi):
            if phi_i == -1:
                permuted_lcp[i] = 0
                continue
            while (i + left < local_text_len
                   and phi_i + left < local_text_len
                   and local_text_ord[i + left] == local_text_ord[phi_i + left]):
                left += 1
            permuted_lcp[i] = left
            left = 0 if left < 1 else left - 1  # this replaced max(left - 1, 0)
        self.longest_common_prefix = [permuted_lcp[suffix] for suffix in local_suffix_array]

    def suffix_array_compare_from_index(self, offset):
        """C style string compare to compare 0 is equal 1 is greater than -1 is less than.

        Complexity per call: Time: O(k) len of pattern, Space: O(1)
        """
        local_text_ord = self.text_ord        # optional for avoiding expensive load_attr operation
        local_pattern_ord = self.pattern_ord  # ignore for faster implementation of the code
        for i, num_char in enumerate(local_pattern_ord):
            if num_char != local_text_ord[offset + i]:
                return -1 if num_char < local_text_ord[offset + i] else 1
        return 0

    def suffix_array_binary_search(self, lo, hi, comp_val):
        """Standard binary search. comp_val allows us to select how strict we are, > vs >=

        Complexity per call: Time: O(k log n) len of pattern, Space: O(1)
        """
        local_suffix_arr = self.suffix_array
        while lo < hi:
            mid = (lo + hi) // 2
            if self.suffix_array_compare_from_index(local_suffix_arr[mid]) > comp_val:
                hi = mid
            else:
                lo = mid + 1
        return lo, hi

    def suffix_array_string_matching(self, new_pattern):
        """Utilizing the suffix array we can search efficiently for a pattern. gives first and last
        index found for patterns.

        Complexity per call: Time: O(k log n), T(2(k log n)), Space: O(k)
        """
        local_suffix_array = self.suffix_array  # optional avoid expensive load_attr operation
        self.pattern_ord = [ord(c) for c in new_pattern]  # line helps avoid repeated ord calls
        lo, _ = self.suffix_array_binary_search(0, self.text_len - 1, GREATER_EQUAL)
        if self.suffix_array_compare_from_index(local_suffix_array[lo]) != 0:
            return -1, -1
        _, hi = self.suffix_array_binary_search(lo, self.text_len - 1, GREATER_THAN)
        if self.suffix_array_compare_from_index(local_suffix_array[hi]) != 0:
            hi -= 1
        return lo, hi

    def compute_longest_repeated_substring(self):
        """The longest repeated substring is just the longest common pattern. Require lcp to be
        computed already. Returns the first longest repeat pattern, so for other ones implement a
        forloop.

        Complexity per call: Time: O(n), T(2n), Space: O(1)
        for optimization implement the physical forloop itself, however it's still O(n).
        """
        local_lcp = self.longest_common_prefix
        max_lcp = max(local_lcp)
        return max_lcp, local_lcp.index(max_lcp)

    def compute_owners(self):
        """Used to compute the owners of each position in the text. O(n) time and space."""
        local_ord_arr, local_suffix = self.text_ord, self.suffix_array  # optional avoids load_attr
        tmp_owner = [0] * self.text_len
        it = iter(self.seperator_list)
        seperator = next(it)
        for i, ord_value in enumerate(local_ord_arr):
            tmp_owner[i] = seperator
            if ord_value == seperator:
                seperator = next(it, None)
        self.owner = [tmp_owner[suffix_i] for suffix_i in local_suffix]

    def compute_longest_common_substring(self):
        """Computes the longest common substring between two strings. returns index, value pair.

        Complexity per call: Time: O(n), Space: O(1)
        Pre-Requirements: owner, and longest_common_prefix must be built (also suffix array for lcp)
        Variants: LCS pair from k strings, LCS between all k strings.
        """
        local_lcp = self.longest_common_prefix  # optional avoid expensive load_attr operation
        local_owners = self.owner               # can be ignored to code faster
        it = iter(local_lcp)
        max_lcp_index, max_lcp_value = 0, next(it) - 1  # - 1 here since next(it) should return 0
        for i, lcp_value in enumerate(it, 1):
            if lcp_value > max_lcp_value and local_owners[i] != local_owners[i - 1]:
                max_lcp_index, max_lcp_value = i, lcp_value
        return max_lcp_index, max_lcp_value
        
    def compute_rolling_hash(self, new_text):
        """For a given text compute and store the rolling hash. we use the smallest prime lower than
        2^30 since python gets slower after 2^30, p = 131 is a small prime below 256.

        Complexity per call: Time: O(n), T(4n),  Space O(n), midcall S(6n), post call S(4n)
        """
        len_text, p, m = len(new_text), 131, 2**30 - 35  # p is prime m is the smallest prime < 2^30
        h_vals, powers, ord_iter = [0] * len_text, [0] * len_text, map(ord, new_text)
        powers[0], h_vals[0] = 1, 0
        for i in range(1, len_text):
            powers[i] = (powers[i - 1] * p) % m
        for i, power in enumerate(powers):
            h_vals[i] = ((h_vals[i-1] if i != 0 else 0) + (next(ord_iter) * power) % m) % m
        self.text = new_text
        self.hash_powers, self.hash_h_values = powers, h_vals
        self.prime_p, self.mod_m, self.math_algos = p, m, MathAlgorithms()
        self.left_mod_inverse = [pow(power, m-2, m) for power in powers]  # optional

    def hash_fast_log_n(self, left, right):
        """Log n time calculation of rolling hash formula: h[right]-h[left] * mod_inverse(left).

        Complexity per call: Time: O(log mod_m), Space: O(1)
        """
        loc_h_vals = self.hash_h_values  # optional avoids expensive load_attr operation
        ans = loc_h_vals[right]
        if left != 0:
            loc_mod = self.mod_m  # optional avoids expensive load_attr operation
            ans = ((ans - loc_h_vals[left - 1])
                   * pow(self.hash_powers[left], loc_mod-2, loc_mod)) % loc_mod
        return ans

    def hash_fast_constant(self, left, right):
        """Constant time calculation of rolling hash. formula: h[right]-h[left] * mod_inverse[left]

        Complexity per call: Time: O(1), Space: O(1)
        more uses: string matching in n + m or constant if you know that it's a set size
        kattis typo: uses a variant of this were you will expand on the code further.
        """
        loc_h_vals = self.hash_h_values  # optional avoids expensive load_attr operation
        ans = loc_h_vals[right]
        if left != 0:
            ans = ((ans - loc_h_vals[left - 1]) * self.left_mod_inverse[left]) % self.mod_m
        return ans

class Matrix:
    def __init__(self, n, m):
        self.matrix = []
        self.num_rows = n
        self.num_cols = m
        self.matrix = [[0 for _ in range(m)] for _ in range(n)]

    def get_best_sawp_row(self, row, col):
        local_mat, local_rows = self.matrix, self.num_rows
        best, pos = 0.0, -1
        for i in range(row, local_rows):
            if abs(local_mat[i][col]) > best:
                best, pos = abs(local_mat[i][col]), i
        return pos

    def swap_rows(self, row_a, row_b):
        local_mat = self.matrix
        local_mat[row_a], local_mat[row_b] = local_mat[row_b], local_mat[row_a]

    def divide_row(self, row, div):
        local_mat, local_cols = self.matrix, self.num_cols
        for i in range(local_cols):
            local_mat[row][i] /= div

    def row_reduce_helper(self, i, row, val):
        local_mat, local_cols = self.matrix, self.num_cols
        for j in range(local_cols):
            self.matrix[i][j] -= (val * local_mat[row][j])

    def row_reduce(self, row, col, row_begin):
        for i in range(row_begin, self.num_rows):
            if i != row:
                self.row_reduce_helper(i, row, self.matrix[i][col])

    def row_reduce_2(self, row, col, other):
        for i in range(self.num_rows):
            if i != row:
                tmp = self.matrix[i][col]
                self.matrix[i][col] = 0
                self.row_reduce_helper(i, row, tmp)
                other.row_reduce_helper(i, row, tmp)

    def __mul__(self, multiplier):
        product = Matrix(self.num_rows, self.num_rows)
        for k in range(self.num_rows):
            for i in range(self.num_rows):
                if self.matrix[i][k] != 0:
                    for j in range(self.num_rows):
                        product.matrix[i][j] += (self.matrix[i][k] * multiplier.matrix[k][j])
        return product 

    def set_identity(self):
        for i in range(base.num_rows):
            for j in range(base.num_rows):
                self.matrix[i][j] = 1 if i == j else 0

    def fill_matrix(self, new_matrix, a, b):
        for i in range(new_matrix.num_rows):
            for j in range(new_matrix.num_cols):
                self.matrix[i + a][j + b] = new_matrix.matrix[i][j]


    def get_augmented_matrix(self, matrix_b):
        augmented = Matrix(self.num_rows + matrix_b.num_rows, 
                           self.num_cols + matrix_b.num_cols)
        augmented.fill_matrix(self, 0, 0)
        augmented.fill_matrix(matrix_b, 0, self.num_cols)
        return augmented

    def get_determinant_matrix(self):
        det = Matrix(self.num_rows, self.num_cols)
        r = 1
        det.fill_matrix(self, 0, 0)
        for i in range(self.num_rows):
            for j in range(self.num_rows):
                while determinant.matrix[j][i] != 0:
                    ratio = det.matrix[i][i] / det.matrix[j][i]
                    for k in range(i, self.num_rows):
                        det.matrix[i][k] -= (ratio * det.matrix[j][k])
                        det.matrix[i][k], det.matrix[j][k] = det.matrix[j][k], det.matrix[i][k]
                    r = -r
            r = r * det[i][i]
        return r

import dis
M = Matrix(1,1,)
dis.dis(M.swap_rows)










class Matrix_Algorithhms:
    def __init__(self):
        self.matrix_A = []
        self.matrix_b = []
        self.matrix_x = []
        self.num_rows = 0
        self.num_cols = 0

    def matrix_exponentiation(self, base, power):
        result = Matrix(base.num_rows, base.num_rows)
        result.set_identity()
        while power:
            if power%2 == 1:
                result = result * base
            base = base * base
            power //= 2
        return result
    
    def init_matrices(self):
        self.matrix_A = [[0 for _ in range(self.num_rows)] for _ in range(self.num_rows)] 
        self.matrix_b = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)] 
    
    def init_constants(self, n, m):
        self.num_rows = n
        self.num_cols = m

    def init_data(self, n, m):
        self.init_constants(n, m)

    def get_rank_via_reduced_row_echelon(self, aug_Ab):
        rank = 0
        for col in range(aug_Ab.num_cols):
            if rank == aug_Ab.num_rows:
                break
            pos = aug_Ab.get_best_sawp_row(rank, col)
            if pos != -1:
                aug_Ab.swap_rows(pos, rank)
                aug_Ab.divide_row(rank, aug_Ab.matrix[row][col])
                aug_Ab.row_reduce(rank, col, 0)
                rank += 1
        return rank
            
    def gauss_elimination(self, aug_Ab):
        rank = 0
        for col in range(aug_Ab.num_cols):
            if rank == aug_Ab.num_rows:
                break
            pos = aug_Ab.get_best_sawp_row(rank, col)
            if pos != -1:
                aug_Ab.swap_rows(pos, rank)
                aug_Ab.divide_row(rank, aug_Ab.matrix[row][col])
                aug_Ab.row_reduce(rank, col, rank + 1)
                rank += 1
        n = aug_Ab.num_rows
        for i in range(n - 1, -1, -1):
            for j in range(i):
                aug_Ab.matrix[j][n] -= (aug_Ab.matrix[i][n] * aug_Ab.matrix[j][i])
                aug_Ab.matrix[j][i] = 0

    def gauss_jordan_elimination(self, a, b):
        n, m = a.num_rows, b.num_cols
        det = 1.0
        irow, icol, ipivj, ipivk = [0] * n, [0] * n, set(range(n)), set(range(n))
        for i in range(n):
            pj, pk = -1, -1
            for j in ipivj:
                for k in ipivk:
                    if pj == -1 or abs(a.matrix[j][k]) > abs(a.math[pj][pk]):
                        pj, pk = j, k
            ipivj.remove(pk)
            ipivk.remove(pk)
            a.swap_rows(pj, pk)
            b.swap_rows(pj, pk)
            if pj != pk:
                det = -det
            irow[i], icol[i] = pj, pk
            div = a.matrix[pk][pk]
            det /= div
            a.matrix[pk][pk] = 1.0
            a.divide_row(pk, div)
            b.divide_row(pk, div)
            a.row_reduce_2(pk, pk, b)
        for p in range(n - 1, -1, -1):
            if irow[p] != icol[p]:
                for k in range(n):
                    a.matrix[k][irow[p]], a.matrix[k][icol[p]] = a.matrix[k][icol[p]], a.matrix[k][irow[p]]
        return det
            

            
    
        
    
    


