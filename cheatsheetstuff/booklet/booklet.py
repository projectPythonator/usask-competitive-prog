#data structurea
#union find
from sys import setrecursionlimit
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
from math import log2
from collections import deque
from heapq import heappush, heappop, heapify
from sys import setrecursionlimit
setrecursionlimit(100000)

class Graph:
    def __init__(self, v, e, r=None, c=None):
        self.num_edges = e
        self.num_nodes = v
        self.num_rows = r
        self.num_cols = c

        self.adj_list = []
        self.adj_matrix = []
        self.edge_list = []
        self.grid = []

        self.data_to_code = {}
        self.code_to_data = []

    def convert_data_to_code(self, data):
        """Converts data to the form: int u | 0 <= u < |V|, stores (data, u) pair, then return u."""
        if data not in self.data_to_code:
            self.data_to_code[data] = len(self.code_to_data)
            self.code_to_data.append(data) # can be replaced with a count variable if space needed
        return self.data_to_code[data]

    def add_edge_u_v_wt_into_directed_graph(self, u, v, wt=None, data=None):
        """A pick and choose function will convert u, v into index form then add it to the structure
        you choose.
        """
        u = self.convert_data_to_code(u) # omit if u,v is in the form: int u | 0 <= u < |V|
        v = self.convert_data_to_code(v) # omit if u,v is in the form: int u | 0 <= u < |V|

        self.adj_list[u].append((v, wt))    # Adjacency list usage
        self.adj_matrix[u][v] = wt          # Adjacency matrix usage
        self.edge_list.append((wt, u, v))   # Edge list usage
        # the following lines come as a pair-set used in max flow algorithm and are used in tandem.
        self.edge_list.append((v, wt, data))
        self.adj_list[u].append(len(self.edge_list) - 1)

    def add_edge_u_v_wt_into_undirected_graph(self, u, v, wt=None):
        """undirected graph version of the previous function"""
        self.add_edge_u_v_wt_into_undirected_graph(u, v, wt)
        self.add_edge_u_v_wt_into_undirected_graph(v, u, wt)

    def fill_grid_graph(self, new_grid):
        self.num_rows = len(new_grid)
        self.num_cols = len(new_grid[0])
        self.grid = [[self.convert_data_to_code(el) for el in row] for row in new_grid]

INF=2**31
UNVISITED = -1
EXPLORED  = -2
VISITED   = -3
TREE = 0
BIDIRECTIONAL = 1
BACK = 2
FORWARD = 3
class GraphAlgorithms:
    
    def __init__(self, new_graph):
        self.graph = new_graph
        self.dfs_counter = None
        self.dfs_root = None
        self.root_children = None
        self.region_num = None

        self.dir_rc = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.mst_node_set = None
        self.dist = []
        self.visited = None
        self.topo_sort_node_set = None
        self.parent = None
        self.low_values = None
        self.articulation_nodes = None
        self.bridge_edges = None
        self.directed_edge_type = None
        self.component_region = None
        self.decrease_finish_order = None
        self.nodes_on_stack = None
        self.node_state = None
        self.bipartite_colouring = None
        self.last = None

    def flood_fill_via_dfs(self, row, col, old_val, new_val): #needs test
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

    def flood_fill_via_bfs(self, start_row, start_col, old_val, new_val):  #needs test
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

    #will kill the edge list but will save memory
    def min_spanning_tree_via_kruskals_and_heaps(self):  #needs test
        """Computes mst of graph G stored in edge_list, space optimized via heap.

        Complexity: Time: O(|E|log |V|), Space: O(|E|) + Union_Find
        More uses: finding min spanning tree
        Variants: min spanning subgraph and forrest, max spanning tree, 2nd min best spanning tree
        Optimization: We use a heap to make space comp. O(|E|) 
        instead of O(|E|log |E|) when using sort, however edge_list is CONSUMED.
        """
        heapify(self.graph.edge_list)
        ufds = UnionFindDisjointSets(self.graph.num_nodes)
        min_spanning_tree = []
        while self.graph.edge_list and ufds.num_sets > 1:
            wt, u, v = heappop(self.graph.edge_list) # use w, uv = ... for single cord storage
            #v,u = uv%self.num_nodes, uv//self.num_nodes
            if not ufds.is_same_set(u, v):
                min_spanning_tree.append((wt, u, v))
                ufds.union_set(u, v)
        self.mst_node_set = min_spanning_tree
        
    def prims_visit_adj_matrix(self, u, not_visited, mst_best_dist, heap):
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
    
    def prims_visit_adj_list(self, u, not_visited, mst_best_dist, heap): #needs test
        """Find min weight edge in adjacency list implementation of prims.

        Complexity per call: Time: O(|V|log |V|), Space: increase by O(|V|)
        """
        not_visited[u] = False
        for v, wt in self.graph.adj_list[u]:
            if wt <= mst_best_dist[v] and not_visited[v]:
                mst_best_dist[v] = wt
                heappush(heap, (wt, v, u))
    
    def min_spanning_tree_via_prims(self):  #needs test
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
        self.mst_node_set.sort()

    def breadth_first_search_vanilla_template(self, source): #needs test
        """Template for distance based bfs traversal from node source.

        Complexity per call: Time: O(|V| + |E|), Space: O(|V|)
        More uses: connectivity, shortest path on monotone weighted graphs
        Input:
            source: is the node we start from
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

    def topology_sort_via_tarjan_helper(self, u):
        """Recursively explore unvisited graph via dfs.

        Complexity per call: Time: O(|V|), Space: O(|V|) at deepest point
        """
        self.visited[u] = VISITED
        for v in self.graph.adj_list[u]:
            if self.visited[v] == UNVISITED:
                self.topology_sort_via_tarjan_helper(v)
        self.topo_sort_node_set.append(u)
        
    def topology_sort_via_tarjan(self):
        """Compute a topology sort via tarjan method, on adj_list.

        Complexity per call: Time: O(|V| + |E|), Space: O(|V|)
        More Uses: produces a DAG, topology sorted graph, build dependencies
        """
        self.visited = [UNVISITED] * self.graph.num_nodes
        self.topo_sort_node_set = []
        for u in range(self.graph.num_nodes):
            if self.visited[u] == UNVISITED:
                self.topology_sort_via_tarjan_helper(u)
        self.topo_sort_node_set = self.topo_sort_node_set[::-1]

    def topology_sort_via_kahns(self):
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

    def amortized_heap_fix(self, heap):
        """Should we need |V| space this will ensure that while still being O(log|V|)"""
        tmp = [-1] * self.graph.num_nodes
        for wt, v in heap:
            if tmp[v] == -1:
                tmp[v] = wt
        heap = [(wt, v) for v, wt in enumerate(tmp) if wt != -1]
        heapify(heap)

    def single_source_shortest_path_dijkstras(self, source, sink): #needs test
        """It is Dijkstra's pathfinder using heaps.

        Complexity per call: Time: O(|E|log |V|), Space: O(|V|)
        More uses: shortest path on state based graphs
        Input:
            source: can be a single nodes or list of nodes
            sink: the goal node
        """
        distance, parents = [INF] * self.graph.num_nodes, [UNVISITED] * self.graph.num_nodes
        distance[source], parents[source] = 0, source
        heap, limit = [(0, source)], 2**(int(log2(self.graph.num_nods)) + 4)
        while heap:
            # if len(heap) > limit:
            #     self.amortized_heap_fix(heap)
            cur_dist, u = heappop(heap)
            if distance[u] < cur_dist:
                continue
            # if u == sink: return cur_dist #uncomment this line for fast return
            for v, wt in self.graph.adj_list[u]:
                if distance[v] > cur_dist + wt:
                    distance[v] = cur_dist + wt
                    parents[v] = u
                    heappush(heap, (distance[v], v))
        self.dist = distance
        self.parent = parents
    
    def all_pairs_shortest_path_floyd_warshall(self): #needs test
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

    def articulation_point_and_bridge_helper_via_dfs(self, u):
        # need to rego over this and test it *** not as confident as the other code atm since have
        # not really used it to solve a problem
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

    def articulation_points_and_bridges_via_dfs(self):
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

    def cycle_check_on_directed_graph_helper(self, u):
        """Recursion part of the dfs. It is modified to list various types of edges.

        Complexity per call: Time: O(|E|), Space: O(|V|) at deepest call
        More uses: listing edge types: Tree, Bidirectional, Back, Forward/Cross edge. On top of
        listing Explored, Visited, and Unvisited.
        """
        self.visited[u] = EXPLORED
        for v in self.graph.adj_list[u]:
            edge_type = None
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

    def cycle_check_on_directed_graph(self):
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
  
    def strongly_connected_components_of_graph_kosaraju_helper(self, u, pass_one):
        """Pass one explore G and build stack, Pass two mark the SCC regions on transposition of G.

        Complexity per call: Time: O(|E| + |V|), Space: O(|V|)
        """
        self.visited[u] = VISITED
        self.component_region[u] = self.region_num
        neighbours = self.graph.adj_list[u] if pass_one else self.graph.adj_list_trans[u]
        for v in neighbours:
            if self.visited[v] == UNVISITED:
                self.strongly_connected_components_of_graph_kosaraju_helper(v, pass_one)
        if pass_one:
            self.decrease_finish_order.append(u)

    def strongly_connected_components_of_graph_kosaraju(self):
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
    
    def strongly_connected_components_of_graph_tarjans_helper(self, u):
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
                v = self.nodes_on_stack.pop()
                self.visited[v], self.component_region[v] = UNVISITED, self.region_num
                if u == v:
                    break

    def strongly_connected_components_of_graph_tarjans(self):
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

    def bipartite_check_on_graph_helper(self, source, color):
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

    def bipartite_check_on_graph(self):
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


    def max_flow_find_augmenting_path_helper(self, source, sink):
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

    def send_flow_via_augmenting_path(self, source, sink, flow_in):
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

    def send_max_flow_via_dfs(self, u, sink, flow_in):
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

    def max_flow_via_edmonds_karp(self, source, sink):
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

    def max_flow_via_dinic(self, source, sink):
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

    def dfs_bipartite_checker(self):
        pass # find code for this later

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
class Pt2d:
    def __init__(self, x_val, y_val): self.x, self.y = map(float, (x_val, y_val))

    def __add__(self, other): return Pt2d(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Pt2d(self.x - other.x, self.y - other.y)
    def __mul__(self, scale): return Pt2d(self.x * scale, self.y * scale)
    def __truediv__(self, scale): return Pt2d(self.x / scale, self.y / scale)
    def __floordiv__(self, scale): return Pt2d(self.x // scale, self.y // scale)

    def __eq__(self, other): return isclose(self.x, other.x) and isclose(self.y, other.y)
    def __lt__(self, other): return False if self == other else (self.x, self.y) < (other.x, other.y)

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
    def __init__(self):
        self.origin = Pt2d(0, 0)
        self.rot = None
        self.o_next = None
        self.used = False

    def rev(self): return self.rot.rot
    def l_next(self): return self.rot.rev().o_next.rot
    def o_prev(self): return self.rot.o_next.rot
    def dest(self): return self.rev().origin

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
        e3.origin = Pt2d(2 ** 63, 2 ** 63)
        e4.origin = Pt2d(2 ** 63, 2 ** 63)
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
        del edge.rot.rot.rot
        del edge.rot.rot
        del edge.rot
        del edge

    def connect(self, a, b):
        e = self.make_edge(a.dest(), b.origin)
        self.splice(e, a.l_next())
        self.splice(e.rev(), b)
        return e

class GeometryAlgorithms:
    def __init__(self):
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

    def angle_point_c_wrt_line_ab_2d(self, a, b, c):
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

    # projection funcs just returns closes point to obj based on a point c
    def project_pt_c_to_line_ab_2d(self, a, b, c):
        ba, ca = b-a, c-a
        return a + ba*(self.dot_product(ca, ba) / self.dot_product(ba, ba))

    # use compare_ab in return if this isn't good enough
    def project_pt_c_to_line_seg_ab_2d(self, a, b, c):
        ba, ca = b-a, c-a
        u = self.dot_product(ba, ba)
        if self.compare_ab(u, 0.0) == 0:
            return a
        u = self.dot_product(ca, ba) / u
        return a if u < 0.0 else b if u > 1.0 else self.project_pt_c_to_line_ab_2d(a, b, c)

    def distance_pt_c_to_line_ab_2d(self, a, b, c):
        return self.distance_normalized(c, self.project_pt_c_to_line_ab_2d(a, b, c))

    def distance_pt_c_to_line_seg_ab_2d(self, a, b, c):
        return self.distance_normalized(c, self.project_pt_c_to_line_seg_ab_2d(a, b, c))
    
    def is_parallel_lines_ab_and_cd_2d(self, a, b, c, d):
        return self.compare_ab(self.cross_product(b - a, c - d), 0.0) == 0

    def is_collinear_lines_ab_and_cd_2d(self, a, b, c, d):
        return (self.is_parallel_lines_ab_and_cd_2d(a, b, c, d)
        and self.is_parallel_lines_ab_and_cd_2d(b, a, a, c)
        and self.is_parallel_lines_ab_and_cd_2d(d, c, c, a))

    def is_segments_intersect_ab_to_cd_2d(self, a, b, c, d):
        if self.is_collinear_lines_ab_and_cd_2d(a, b, c, d):
            lo, hi = (a, b) if a < b else (b, a)
            return lo <= c <= hi or lo <= d <= hi
        a_val = self.cross_product(d - a, b - a) * self.cross_product(c - a, b - a)
        c_val = self.cross_product(a - c, d - c) * self.cross_product(b - c, d - c)
        return not(a_val>0 or c_val>0)

    def is_lines_intersect_ab_to_cd_2d(self, a, b, c, d):
        return (not self.is_parallel_lines_ab_and_cd_2d(a, b, c, d) or 
                self.is_collinear_lines_ab_and_cd_2d(a, b, c, d))

    def pt_lines_intersect_ab_to_cd_2d(self, a, b, c, d):
        ba, ca, cd = b-a, c-a, c-d
        return a + ba*(self.cross_product(ca, cd) / self.cross_product(ba, cd))

    def pt_line_seg_intersect_ab_to_cd_2d(self, a, b, c, d):
        x, y, cross_prod = c.x-d.x, d.y-c.y, self.cross_product(d, c)
        u = abs(y*a.x + x*a.y + cross_prod)
        v = abs(y*b.x + x*b.y + cross_prod)
        return Pt2d((a.x * v + b.x * u) / (v + u), (a.y * v + b.y * u) / (v + u))

    def is_point_in_circle(self, a, b, r): # use <= if you want points on the circumfrance 
        return self.compare_ab(self.distance_normalized(a, b), r) < 0

    def pt_circle_center_given_pt_abc(self, a, b, c):
        ab, ac = (a+b)/2, (a+c)/2
        ab_rot = ab+self.rotate_cw_90_wrt_origin(a - ab)
        ac_rot = ac+self.rotate_cw_90_wrt_origin(a - ac)
        return self.pt_lines_intersect_ab_to_cd_2d(ab, ab_rot, ac, ac_rot)

    def pts_line_ab_intersects_circle_cr_2d(self, a, b, c, r):
        ba, ac = b-a, a-c
        bb = self.dot_product(ba, ba)
        ab = self.dot_product(ac, ba)
        aa = self.dot_product(ac, ac) - r * r
        dist = ab*ab - bb*aa
        result = self.compare_ab(dist, 0.0)
        if result >= 0:
            first_intersect = c + ac + ba*(-ab + sqrt(dist+EPS))/bb
            second_intersect = c + ac + ba*(-ab - sqrt(dist))/bb
            return (first_intersect) if result == 0 else (first_intersect, second_intersect)
        return None # no intersect 

    def pts_two_circles_intersect_ar1_br1_2d(self, c1, c2, r1, r2):
        center_dist = self.distance_normalized(c1, c2)
        if (self.compare_ab(center_dist, r1+r2) <= 0
            and self.compare_ab(center_dist+min(r1, r2), max(r1, r2)) >= 0):
            x = (center_dist*center_dist - r2*r2 + r1*r1)/(2*center_dist)
            y = sqrt(r1*r1 - x*x)
            v = (b-a)/center_dist
            pt1, pt2 = a + v * x, self.rotate_ccw_90_wrt_origin(v) * y
            return (pt1+pt2) if self.compare_ab(y, 0.0) <= 0 else (pt1+pt2, pt1-pt2)
        return None # no overlap

    def pt_tangent_to_circle_cr_2d(self, c, r, p):
        pc = p-c
        x = self.dot_product(pc, pc)
        dist = x - r*r
        result = self.compare_ab(dist, 0.0)
        if result >= 0:
            dist = dist if result else 0
            q1 = pa * (r*r / x)
            q2 = self.rotate_ccw_90_wrt_origin(pa * (-r * sqrt(dist) / x))
            return [a+q1-q2, a+q1+q2]
        return []

    def tangents_between_2_circles_2d(self, c1, r1, c2, r2):
        r_tangents = []
        if self.compare_ab(r1, r2) == 0:
            c2c1 = c2 - c1
            multiplier = r1/sqrt(self.dot_product(c2c1, c2c1))
            tangent = self.rotate_ccw_90_wrt_origin(c2c1 * multiplier) # need better name
            r_tangents = [(c1+tangent, c2+tangent), (c1-tangent, c2-tangent)]
        else:
            ref_pt = ((c1 * -r2) + (c2 * r1)) / (r1 - r2)
            ps = self.pt_tangent_to_circle_cr_2d(c1, r1, ref_pt)
            qs = self.pt_tangent_to_circle_cr_2d(c2, r2, ref_pt)
            r_tangents = [(ps[i], qs[i]) for i in range(min(len(ps), len(qs)))]
        ref_pt = ((c1 * r2) + (c2 * r1)) / (r1 + r2)
        ps = self.pt_tangent_to_circle_cr_2d(c1, r1, ref_pt)
        qs = self.pt_tangent_to_circle_cr_2d(c2, r2, ref_pt)
        for i in range(min(len(ps), len(qs))):
            r_tangents.append((ps[i], qs[i]))
        return r_tangents

    def sides_of_triangle_abc_2d(self, a, b, c):
        ab = self.distance_normalized(a, b)
        bc = self.distance_normalized(b, c)
        ca = self.distance_normalized(c, a)
        return ab, bc, ca

    def pt_p_in_triangle_abc_2d(self, a, b, c, p):
        return self.point_c_rotation_wrt_line_ab(a, b, p) >= 0 and  \
                self.point_c_rotation_wrt_line_ab(b, c, p) >= 0 and \
                self.point_c_rotation_wrt_line_ab(c, a, p) >= 0

    def perimeter_of_triangle_abc_2d(self, ab, bc, ca):
        return ab + bc + ca

    def triangle_area_bh_2d(self, b, h):
        return b*h/2

    def triangle_area_heron_abc_2d(self, ab, bc, ca):
        s = self.perimeter_of_triangle_abc_2d(ab, bc, ca) / 2
        return sqrt(s * (s-ab) * (s-bc) * (s-ca))

    def triangle_area_cross_product_abc_2d(self, a, b, c):
        ab = self.cross_product(a, b)
        bc = self.cross_product(b, c)
        ca = self.cross_product(c, a)
        return (ab + bc + ca)/2

    def incircle_radis_of_triangle_abc_helper_2d(self, ab, bc, ca):
        area = self.triangle_area_heron_abc_2d(ab, bc, ca)
        perimeter = self.perimeter_of_triangle_abc_2d(ab, bc, ca)/2
        return area/perimeter

    def incircle_radis_of_triangle_abc_2d(self, a, b, c):
        ab, bc, ca = self.sides_of_triangle_abc_2d(a, b, c)
        return self.incircle_radis_of_triangle_abc_helper_2d(ab, bc, ca)

    def circumcircle_radis_of_triangle_abc_helper_2d(self, ab, bc, ca):
        area = self.triangle_area_heron_abc_2d(ab, bc, ca)
        return (ab*bc*ca) / (4*area)
        
    def circumcircle_radis_of_triangle_abc_2d(self, a, b, c):
        ab, bc, ca = self.sides_of_triangle_abc_2d(a, b, c)
        return circumcircle_radis_of_triangle_abc_helper_2d(ab, bc, ca)

    def incircle_pt_for_triangle_abc_2d(self, a, b, c):
        radius = self.incircle_radis_of_triangle_abc_2d(a, b, c)
        if self.compare_ab(radius, 0.0):
            return (False, 0, 0)
        dist_ab = self.distance_normalized(a, b)
        dist_bc = self.distance_normalized(b, c)
        dist_ac = self.distance_normalized(a, c)
        ratio_1 = dist_ab/dist_ac
        ratio_2 = dist_ab/dist_bc
        pt_1 = b + (c-b) * (ratio_1/(ratio_1 + 1.0)) 
        pt_2 = a + (c-a) * (ratio_2/(ratio_2 + 1.0))

        if self.is_lines_intersect_ab_to_cd_2d(a, pt_1, b, pt_2):
            intersection_pt = self.pt_lines_intersect_ab_to_cd_2d(a, pt_1, b, pt_2)
            return (True, radius, round(intersection_pt, 12)) # can remove the round function
        return (False, 0, 0)

    def triangle_circle_center_pt_abcd_2d(self, a, b, c, d):
        ba, dc = b-a, d-c
        pt_1, pt_2 = Pt2d(ba.y, -ba.x), Pt2d(dc.y, -dc.x)
        cross_product_1_2 = self.cross_product(pt_1, pt_2)
        cross_product_2_1 = self.cross_product(pt_2, pt_1)
        if self.compare_ab(cross_product_1_2, 0.0) == 0:
            return None
        pt_3 = Pt2d(self.dot_product(a, pt_1), self.dot_product(c, pt_2))
        x = ((pt_3.x * pt_2.y) - (pt_3.y * pt_1.y)) / cross_product_1_2
        y = ((pt_3.x * pt_2.x) - (pt_3.y * pt_1.x)) / cross_product_2_1
        return Pt2d(x, y)

    def angle_bisector_for_triangle_abc_2d(self, a, b, c):
        dist_ba = self.distance_normalized(b, a)
        dist_ca = self.distance_normalized(c, a)
        ref_pt = (b-a) / dist_ba * dist_ca
        return ref_pt + (c-a) + a

    def perpendicular_bisector_for_triangle_ab_2d(self, a, b):
        ba = b-a
        ba = Pt2d(-ba.y, ba.x)
        return ba + (a+b)/2

    def incircle_pt_of_triangle_abc_v2_2d(self, a, b, c):
        abc = self.angle_bisector_for_triangle_abc_2d(a, b, c)
        bca = self.angle_bisector_for_triangle_abc_2d(b, c, a)
        return self.triangle_circle_center_pt_abcd_2d(a, abc, b, bca)

    def circumcenter_pt_of_triangle_abc_v2_2d(self, a, b, c):
        ab = self.perpendicular_bisector_for_triangle_ab_2d(a, b)
        bc = self.perpendicular_bisector_for_triangle_ab_2d(b, c)
        ab2, bc2 = (a+b)/2, (b+c)/2
        return self.triangle_circle_center_pt_abcd_2d(ab2, ab, bc2, bc)

    def orthocenter_pt_of_triangle_abc_v2_2d(self, a, b, c):
        return a + b + c - 2*self.circumcenter_pt_of_triangle_abc_v2_2d(a, b, c)

    # note these assume counter clockwise ordering of points
    def perimeter_of_polygon_pts_2d(self, pts):
        return fsum([self.distance_normalized(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])

    def signed_area_of_polygon_pts_2d(self, pts):
        return fsum([self.distance_normalized(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])/2

    def area_of_polygon_pts_2d(self, pts):
        return abs(self.signed_area_of_polygon_pts_2d(pts))

    # < is counter clock wise <= includes collinear > for clock wise >= includes collinear
    def is_convex_helper(self, a, b, c):
        return 0 < self.point_c_rotation_wrt_line_ab(a, b, c)

    def is_convex_polygon_pts_2d(self, pts):
        lim = len(pts)
        if lim > 3:
            is_ccw = self.is_convex_helper(pts[0], pts[1], pts[2])
            for i in range(1, n-1):
                a, b, c = pts[i], pts[i+1], pts[i+2 if i+2<lim else 1]
                if base != self.is_convex_helper(a, b, c):
                    return False
            return True 
        return False

    def pt_p_in_polygon_pts_v1_2d(self, pts, p):
        n = len(pts)
        if n > 3:
            angle_sum = 0.0
            for i in range(n-1):
                if 1 == self.point_c_rotation_wrt_line_ab(pts[i], pts[i + 1], p):
                    angle_sum += angle_point_c_wrt_line_ab_2d(pts[i], pts[i+1], p)
                else:
                    angle_sum -= angle_point_c_wrt_line_ab_2d(pts[i], pts[i+1], p)
            return self.compare_ab(abs(angle_sum), pi)
        return -1

    def pt_p_in_polygon_pts_v2_2d(self, pts, p):
        ans = False
        px, py = p.get_tup()
        for i in range(len(pts)-1):
            x1, y1 = pts[i].get_tup()
            x2, y2 = pts[i+1].get_tup()
            lo, hi = y1, y2 if y1 < y2 else y2, y1
            if lo <= py < hi and px < (x1 + (x2-x1) * (py-y1) / (y2-y1)):
                ans = not ans
        return ans

    def pt_p_on_polygon_perimeter_pts_2d(self, pts, p):
            n = len(pts)
            if p in pts:
                return True
            for i in range(n-1):
                dist_ip = self.distance_normalized(pts[i], p)
                dist_pj = self.distance_normalized(p, pts[i + 1])
                dist_ij = self.distance_normalized(pts[i], pts[i + 1])
                if self.compare_ab(dist_ip+dist_pj, dist_ij) == 0:
                    return True
            return False

    def pt_p_in_convex_polygon_pts_2d(self, pts, p):
        n = len(pts)
        if n == 2:
            distance = self.distance_pt_c_to_line_seg_ab_2d(pts[0], pts[1], p)
            return self.compare_ab(distance, 0.0) == 0
        left, right = 1, n
        while left < right:
            mid = (left + rigth)/2 + 1
            side = self.point_c_rotation_wrt_line_ab(pts[0], pts[mid], p)
            left, right = mid, right if side == 1 else left, mid-1
        side = self.point_c_rotation_wrt_line_ab(pts[0], pts[left], p)
        if side == -1 or left == n:
            return False
        side = self.point_c_rotation_wrt_line_ab(pts[left], pts[left + 1] - pts[left], p)
        return side >= 0
    
    # use a set with points if possible checking on the same polygon many times    
    # return 0 for on 1 for in -1 for out
    def pt_p_position_wrt_polygon_pts_2d(self, pts, p):
        return 0 if self.pt_p_on_polygon_perimeter_pts_2d(pts, p) \
                else 1 if self.pt_p_in_polygon_pts_v2_2d(pts, p) else -1

    def centroid_pt_of_convex_polygon_2d(self, pts):
        ans, n = Pt2d(0, 0), len(pts)
        for i in range(n-1):
            ans = ans + (pts[i]+pts[i+1]) * self.cross_product(pts[i], pts[i + 1])
            return ans / (6.0 * self.signed_area_of_polygon_pts_2d(pts))

    def is_polygon_pts_simple_quadratic_2d(self, pts):
        n = len(pts)
        for i in range(n-1):
            for k in range(i+1, n-1):
                j, l = (i+1) % n, (k+1) % n
                if i == l or j == k:
                    continue
                if self.is_segments_intersect_ab_to_cd_2d(pts[i], pts[j], pts[k], pts[l]):
                    return False
        return True

    def polygon_cut(self, pts, a, b):
        ans, n = [], len(pts)
        for i in range(n-1):
            rot_1 = self.point_c_rotation_wrt_line_ab(a, b, pts[i])
            rot_2 = self.point_c_rotation_wrt_line_ab(a, b, pts[i + 1])
            if 1 == rot_1:
                ans.append(pts[i])
            elif 0 == rot_1:
                ans.append(pts[i])
                continue
            if 1 == rot_1 and -1 == rot_2:
                ans.append(self.pt_line_seg_intersect_ab_to_cd_2d(pts[i], pts[i+1], a, b))
        if ans and ans[0] != ans[-1]:
            ans.append(ans[0])
        return ans

    def convex_hull_monotone_chain(self, pts):
        def func(points, r, lim):
            for p in points:
                while (len(r) > lim and
                       self.point_c_rotation_wrt_line_ab(r[-2], r[-1], p) == -1):
                    r.pop()
                r.append(p)
            r.pop()
        ans, convex = sorted(set(pts)), []
        if len(ans) < 2: 
            return ans
        func(ans, convex, 1)
        func(ans[::-1], convex, len(convex)+1)
        return convex
    
    def rotating_caliper_of_polygon_pts_2d(self, pts):
        n, t, ans = len(pts)-1, 0, 0.0
        for i in range(n):
            pi = pts[i]
            pj = pts[i+1] if i+1 <= n else pts[i]
            p = pj-pi
            while (t+1) % n != i:
                cross_1 = self.cross_produc_2d(p, pts[t+1] - pi)
                cross_2 = self.cross_produc_2d(p, pts[t] - pi)
                if self.compare_ab(cross_1, cross_2) == -1:
                    break
                t = (t+1) % n
            ans = max(ans, self.distance_normalized(pi, pts[t]))
            ans = max(ans, self.distance_normalized(pj, pts[t]))
        return ans

    def closest_pair_helper_2d(self, lo, hi):
        r_closest = (self.distance(self.x_ordering[lo], self.x_ordering[lo + 1]),
                     self.x_ordering[lo], 
                     self.x_ordering[lo+1])
        for i in range(lo, hi):
            for j in range(i+1, hi):
                distance_ij = self.distance(self.x_ordering[i],
                                            self.x_ordering[j])
                if self.compare_ab(distance_ij, r_closest):
                    r_closest = (distance_ij, self.x_ordering[i], self.x_ordering[j])
        return r_closest

    def closest_pair_recursive_2d(self, lo, hi, y_ordering):
        n = hi-lo
        if n < 4: # base case 
            return self.closest_pair_helper_2d(lo, hi)
        left_len, right_len = lo + n - n//2, lo + n//2
        mid = round((self.x_ordering[left_len].x + self.x_ordering[right_len].x)/2)
        y_part_left = [el for el in y_ordering if self.compare_ab(el.x, mid) <= 0]
        y_part_right = [el for el in y_ordering if self.compare_ab(el.x, mid) > 0]
        best_left = self.closest_pair_recursive_2d(lo, left_len, y_part_left)
        if self.compare_ab(best_left[0], 0.0) == 0:
            return best_left
        best_right = self.closest_pair_recursive_2d(left_len, hi, y_part_right)
        if self.compare_ab(best_right[0], 0.0) == 0:
            return best_right
        y_part_left = None
        y_part_right = None
        best_pair = best_left if self.compare_ab(best_left[0], best_right[0]) <= 0 else best_right
        y_check = [el for el in y_ordering if self.compare_ab((el.x - mid) * (el.x - mid), best_pair[0]) < 0]
        y_check_len = len(y_check)
        for i in range(y_check_len):
            for j in range(i+1, y_check_len):
                dist_ij = y_check[i].y - y_check[j].y
                if self.compare_ab(dist_ij * dist_ij, best_pair[0]) > 0:
                    break
                dist_ij = self.distance(y_check[i], y_check[j])
                if self.compare_ab(dist_ij, best_pair) < 0:
                    best_pair = (dist_ij, y_check[i], y_check[j])
        return best_pair

    def compute_closest_pair_2d(self, pts):
        self.x_ordering = sorted(pts, key=lambda pt_xy: pt_xy.x)
        y_ordering = sorted(pts, key=lambda pt_xy: pt_xy.y)
        return self.closest_pair_recursive_2d(0, len(pts), y_ordering)

    def delaunay_triangulation_slow(self, pts):
        n = len(pts)
        ans = []
        z = [self.dot_product(el, el) for el in pts]
        x = [el.x for el in pts]
        y = [el.y for el in pts]
        for i in range(n-2):
            for j in range(i + 1, n):
                for k in range(i + 1, n):
                    if j == k: continue
                    xn = (y[j]-y[i])*(z[k]-z[i]) - (y[k]-y[i])*(z[j]-z[i])
                    yn = (x[k]-x[i])*(z[j]-z[i]) - (x[j]-x[i])*(z[k]-z[i])
                    zn = (x[j]-x[i])*(y[k]-y[i]) - (x[k]-x[i])*(y[j]-y[i])
                    flag = zn < 0
                    for m in range(n):
                        if flag:
                            flag = flag and ((x[m]-x[i])*xn + 
                                             (y[m]-y[i])*yn + 
                                             (z[m]-z[i])*zn <= 0)
                        else:
                            break
                    if flag:
                        ans.append((i, j, k))
        return ans

    def pt_left_of_edge_2d(self, pt, edge):
        return 1 == self.point_c_rotation_wrt_line_ab(pt, edge.origin, edge.dest())

    def pt_right_of_edge_2d(self, pt, edge):
        return -1 == self.point_c_rotation_wrt_line_ab(pt, edge.origin, edge.dest())

    def det3_helper(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):
        return (a1 * (b2 * c3 - c2 * b3) - 
                a2 * (b1 * c3 - c1 * b3) + 
                a3 * (b1 * c2 - c1 * b2))

    def is_in_circle(self, a, b, c, d):
        a_dot = self.self.dot_product(a, a)
        b_dot = self.self.dot_product(b, b)
        c_dot = self.self.dot_product(c, c)
        d_dot = self.self.dot_product(d, d)
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
            return (res, res.rev())
        if r - l + 1 == 3:
            edge_a = self.quad_edges.make_edge(pts[l], pts[l + 1])
            edge_b = self.quad_edges.make_edge(pts[l + 1], pts[r])
            self.quad_edges.splce(edge_a.rev(), edge_b)
            sg = self.point_c_rotation_wrt_line_ab(pts[l], pts[l + 1], pts[r])
            if sg == 0:
                return (edge_a, edge_b.rev())
            edge_c = self.quad_edges.connect(edge_b, edge_a)
            return (edge_a, edge_b.rev()) if sg == 1 else (edge_c.rev(), edge_c)
        mid = (l + r) // 2
        ldo, ldi = self.build_triangulation(l, mid, p)
        rdi, rdo = self.build_triangulation(mid + 1, r, p)
        while True:
            if self.pt_left_of_edge_2d(rdi.origin, ldi):
                ldi = ldi.l_next()
                continue
            if self.pt_right_of_edge_2d(ldi.origin, rdi):
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
            if self.right_of(l_cand_edge.dest(), base_edge_l):
                while self.is_in_circle(base_edge_l.dest(), base_edge_l.origin, 
                                        l_cand_edge.dest(), l_cand_edge.o_next.dest()):
                    t = l_cand_edge.o_next
                    self.quad_edges.delete_edge(l_cand_edge)
                    l_cand_edge = t
            r_cand_edge = base_edge_l.o_prev()
            if self.right_of(r_cand_edge.dest(), base_edge_l):
                    while self.is_in_circle(base_edge_l.dest(), base_edge_l.origin, 
                                            r_cand_edge.dest(), r_cand_edge.o_prev().dest()):
                    t = r_cand_edge.o_prev()
                    self.quad_edges.delete_edge(r_cand_edge)
                    r_cand_edge = t
            l_check = self.right_of(l_cand_edge.dest(), base_edge_l)
            r_check = self.right_of(r_cand_edge.dest(), base_edge_l)
            if not l_check and not r_check:
                break
            if (not l_check or 
                    r_check and 
                    self.is_in_circle(l_cand_edge.dest(), l_cand_edge.origin, r_cand_edge.origin, r_cand_edge.dest())):
                base_edge_l = self.quad_edges.connect(r_cand_edge, base_edge_l.rev())
            else:
                base_edge_l = self.quad_edges.connect(base_edge_l.rev(), l_cand_edge.rev())
        return (ldo, rdo)        
            
    def delaunay_triangulation_fast(self, pts):
        pts.sort()
        result = self.build_triangulation(0, len(pts) - 1, p)
        edge = result[0]
        edges = [edge]
        while self.point_c_rotation_wrt_line_ab(edge.o_next.dest(), edge.dest(), edge.origin) < 0:
            edge = edge.o_next
        def add_helper(pts, edge, edges):
            cur = edge
            while True:
                cur.used = True
                pts.append(cur.origin)
                edges.append(cur.rev())
                cur = cur.l_next()
                if cur == edge:
                    return
        add_helper(pts, edge, edges)
        pts = []
        kek = 0
        while kek < len(edges):
            edge = edges[kek]
            kek += 1
            if not edge.used:
                add_helper(pts, edge, edges)
        ans = [(pts[i], pts[i + 1], pts[i + 2]) for i in range(0, len(pts), 3)]
        return ans
        


class String_Algorithms:
    def __init__(self):
        self.n = 0
        self.text = ''

    def init_data(self, new_text):
        self.n = len(new_text):
        self.text = new_text

    def prepare_text_data(self, new_text):
        self.text = new_text
        self.text_len = len(new_text)

    def prepare_pattern_data(self, new_pattern):
        self.pattern = new_pattern
        self.pattern_len = len(new_pattern)

    def prepare_suffix_array_data(self, new_len):
        self.text_len = new_len
        self.counts_len = max(300, self.text_len)
        self.suffix_array = [i for i in range(self.text_len)]
        self.rank_arrary = [ord(self.text[i]) for i in range(self.text_len)]
        self.powers_of_2 = [2**i for i in range(32) if 2**i < self.text_len]

    def prepare_rolling_hash_data(self):
        self.prime_p = 131
        self.mod_m = 10**9 - 7
        self.powers = [0] * self.text_len
        self.h_vals = [0] * self.text_len
        self.math_algos = MathAlgorithms()

    def kmp_preprocess(self, target):
        self.prepare_pattern_data(target)
        self.back_table = [0] * (self.pattern_len + 1)
        self.back_table[0], j = -1, -1
        for i in range(self.pattern_len):
            while j >= 0 and self.pattern[i] != self.pattern[j]:
                j = self.back_table[j]
            j += 1
            self.back_table[i+1] = j

    def kmp_search_find_indices(self):
        ans = []
        j = 0
        for i in range(self.text_len):
            while j >= 0 and self.text[i] != self.pattern[j]:
                j = self.back_table[j]
            j += 1
            if j == self.pattern_len:
                ans.append(1 + i - j)
                j = self.back_table[j]
        return ans

    def suffix_array_preprocess(self, new_text):
        self.prepare_text_data(''.join([new_text, chr(0)*100010]))
        self.epare_suffix_array_data(len(new_text) + 1)

    def suffix_array_counting_sort(self, k):
        suffix_array_temp = [0] * self.text_len
        counts = [0] * self.counts_len
        ind = 0
        counts[0] = self.text_len - (self.text_len - k)
        for i in range(self.text_len - k):
            counts[self.rank_array[i + k]] += 1
        for i in range(self.counts_len):
            counts_i = counts[i]
            counts[i] = ind
            ind += counts_i
        for i in range(self.text_len):
            pos = 0
            if self.suffix_array[i] + k < self.text_len:
                pos = self.rank_array[self.suffix_array[i] + k]
            suffix_array_temp[counts[pos]] = self.suffix_array[i]
            counts[pos] += 1
        self.suffix_array = [el for el in suffix_array_temp]

    def suffix_array_build_array(self):
        for k in self.powers_of_2:
            self.suffix_array_counting_sort(k)
            self.suffix_array_counting_sort(0)
            rank_array_temp = [0] * self.text_len
            rank = 0
            for i in range(1, self.text_len):
                suffix_1 = self.suffix_array[i]
                suffix_2 = self.suffix_array[i - 1]
                r = r if (self.rank_array[suffix_1] == self.rank_array[suffix_2] and 
                          self.rank_array[suffix_1 + k] == self.rank_array[suffix_2 + k] else r + 1)
                rank_array_temp[suffix_1] = r
            self.rank_array = [el for el in rank_array_temp]

    def suffix_array_check_from_ind(self, ind):
        for i in range(self.pattern_len):
            if self.pattern[i] != self.text[ind + i]:
                return 1 if ord(self.text[ind + i]) > ord(self.pattern[i]) else -1
        return 0

    def suffix_array_binary_search(self, lo, hi, comp_val):
        while lo < hi:
            mid = (lo + hi)//2
            lo, hi = lo, mid if self.suffix_array_check_from_ind(self.suffix_array[mid]) > comp_val \
                            else mid + 1, hi
        return lo, hi

    def suffix_array_string_matching(self, new_pattern):
        self.prepare_text_data(new_pattern)
        lo, _ = self.suffix_array_binary_search(0, self.text_len - 1, -1)
        if self.suffix_array_check_from_ind(self.suffix_array[lo]) != 0:
            return (-1, -1)
        _, hi = self.suffix_array_binary_search(lo, self.text_len - 1, 0)
        if self.suffix_array_check_from_ind(self.suffix_array[hi]) != 0:
            hi -= 1
        return (lo, hi)

    def compute_longest_common_prefix(self):
        permuted_lcp = [0] * self.text_len
        phi = [0] * self.text_len
        l = 0
        phi[0] = -1
        for i in range(1, self.text_len):
            phi[self.suffix_array[i]] = self.suffix_array[i - 1]
        for i in range(self.text_len):
            if phi[i] == -1:
                permuted_lcp[i] = 0
                continue
            while (i + l < self.text_len and 
                   phi[i] + l < self.text_len and 
                   self.text[i + l] == self.text[phi[i] + l]):
                l += 1
            permuted_lcp[i] = l
            l = max(l - 1, 0)
        self.longest_common_prefix = [permuted_lcp[el] for el in self.suffix_array]

    def compute_longest_repeated_substring(self):
        ind, max_lcp = 0, -1
        for i in range(1, self.text_len):
            if self.longest_common_prefix[i] > max_lcp:
                ind, max_lcp = i, self.longest_common_prefix[i]
        return (max_lcp, ind)

    def owner(self, ind):
        return 1 if ind < self.text_len - self.pattern_len - 1 else 2

    def compute_longest_common_substring(self):
        ind, max_lcp = 0, -1
        for i in range(1, self.text_len):
            if (self.owner(self.suffix_array[i]) != self.owner(self.suffix_array[i - 1]) and
                self.longest_common_prefix[i] > max_lcp):
                ind, max_lcp = i, self.longest_common_prefix[i]
        return (max_lcp, ind)
        
    def compute_rolling_hash(self, new_text):
        self.prepare_text_data(new_text)
        self.prepare_rolling_hash_data()
        self.powers[0] = 1
        self.h_vals[0] = 0
        for i in range(1, self.text_len):
            self.powers[i] = (self.powers[i - 1] * self.prime_p) % self.mod_m
        for i in range(self.text_len):
            if i != 0:
                self.h_vals[i] = self.h_vals[i - 1]
            self.h_vals[i] = (self.h_vals[i] + (ord(self.text[i]) * self.powers[i]) % self.mod_m) % self.mod_m

    def hash_fast(self, l, r):
        if l == 0:
            return self.h_vals[r]
        ans = ((self.h_vals[r] - self.h_vals[l - 1]) % self.mod_m + self.mod_m) % self.mod_m
        ans = (ans * self.math_algos.mod_inverse(self.powers[l], self.mod_m)) % self.mod_m
        return ans

class Matrix:
    def __init__(self, n, m):
        self.num_rows = n
        self.num_cols = m
        self.mat = [[0 for _ in range(m)] for _ in range(n)]

    def get_best_sawp_row(self, row, col):
        best, pos = 0.0, -1
        for i in range(row, self.num_rows):
            if abs(self.mat[i][col]) > best:
                best, pos = abs(self.mat[i][col]), i
        return pos

    def swap_rows(self, row_a, row_b):
        for i in range(self.num_cols):
            self.mat[row_a][i], self.mat[row_b][i] = self.mat[row_b][i], self.mat[row_a][i]

    def divide_row(self, row, div):
        for i in range(self.num_cols):
            self.mat[row][i] /= div

    def row_reduce_helper(self, i, row, val):
        for j in range(self.num_cols):
            self.mat[i][j] -= (val * self.mat[row][j])

    def row_reduce(self, row, col, row_begin):
        for i in range(row_begin, self.num_rows):
            if i != row:
                self.row_reduce_helper(i, row, self.mat[i][col])

    def row_reduce_2(self, row, col, other):
        for i in range(self.num_rows):
            if i != row:
                tmp = self.mat[i][col]
                self.mat[i][col] = 0
                self.row_reduce_helper(i, row, tmp)
                other.row_reduce_helper(i, row, tmp)

    def __mul__(self, multiplier):
        product = Matrix(self.num_rows, self.num_rows)
        for k in range(self.num_rows):
            for i in range(self.num_rows):
                if self.mat[i][k] != 0:
                    for j in range(self.num_rows):
                        product.mat[i][j] += (self.mat[i][k] * multiplier.mat[k][j])
        return product 

    def set_identity(self):
        for i in range(base.num_rows):
            for j in range(base.num_rows):
                self.mat[i][j] = 1 if i==j else 0

    def fill_matrix(self, new_matrix, a, b):
        for i in range(new_matrix.num_rows):
            for j in range(new_matrix.num_cols):
                self.mat[i + a][j + b] = new_matrix.mat[i][j]


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
                while determinant.mat[j][i] != 0:
                    ratio = det.mat[i][i] / det.mat[j][i]
                    for k in range(i, self.num_rows):
                        det.mat[i][k] -= (ratio * det.mat[j][k])
                        det.mat[i][k], det.mat[j][k] = det.mat[j][k], det.mat[i][k]
                    r = -r
            r = r * det[i][i]
        return r

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
                aug_Ab.divide_row(rank, aug_Ab.mat[row][col])
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
                aug_Ab.divide_row(rank, aug_Ab.mat[row][col])
                aug_Ab.row_reduce(rank, col, rank + 1)
                rank += 1
        n = aug_Ab.num_rows
        for i in range(n - 1, -1, -1):
            for j in range(i):
                aug_Ab.mat[j][n] -= (aug_Ab.mat[i][n] * aug_Ab.mat[j][i])
                aug_Ab.mat[j][i] = 0

    def gauss_jordan_elimination(self, a, b):
        n, m = a.num_rows, b.num_cols
        det = 1.0
        irow, icol, ipivj, ipivk = [0] * n, [0] * n, set(range(n)), set(range(n))
        for i in range(n):
            pj, pk = -1, -1
            for j in ipivj:
                for k in ipivk:
                    if pj == -1 or abs(a.mat[j][k]) > abs(a.math[pj][pk]):
                        pj, pk = j, k
            ipivj.remove(pk)
            ipivk.remove(pk)
            a.swap_rows(pj, pk)
            b.swap_rows(pj, pk)
            if pj != pk:
                det = -det
            irow[i], icol[i] = pj, pk
            div = a.mat[pk][pk]
            det /= div
            a.mat[pk][pk] = 1.0
            a.divide_row(pk, div)
            b.divide_row(pk, div)
            a.row_reduce_2(pk, pk, b)
        for p in range(n - 1, -1, -1):
            if irow[p] != icol[p]:
                for k in range(n):
                    a.mat[k][irow[p]], a.mat[k][icol[p]] = a.mat[k][icol[p]], a.mat[k][irow[p]]
        return det
            

            
    
        
    
    


