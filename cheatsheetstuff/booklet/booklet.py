#data structurea
#union find 
class UnionFind:
    ''' 
    space O(n) --> N*3 or N*2 for now 
    search time α(n) -->  inverse ackerman practically constant 
    insert time O(1) --> 
    '''
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0]*n #optional optimzation 
        self.sizes = [0]*n #optional information
        self.num_sets = n #optional information
        
    def find_set(self, u):
        u_parent = u
        u_children = []
        while u_parent != self.parents[u_parent]:
            u_children.append(u_parent)
            u_parent = self.parents[u_parent]
        for child in u_children:
            self.parents[child] = u_parent
        return u_parent
        
    def is_same_set(self, u, v):
        return self.find_set(u)==self.find_set(v)

    def union_set(self, u, v):
        up = self.find_set(u)
        vp = self.find_set(v)
        if up==vp:
            return

        if self.ranks[up] < self.ranks[vp]:
            self.parents[up] = vp
            self.sizes[vp] = self.sizes[up]
        elif self.ranks[vp] < self.ranks[up]:
            self.parents[vp] = up
            self.sizes[up] = self.sizes[vp]
        else:
            self.parents[vp] = up
            self.ranks[up] += 1
            self.sizes[up] += self.sizes[vp]
        self.num_sets -= 1

    def size_of_u(self, u): #optional information
        return self.sizes[self.find_set(u)]

######################################################################################
#

class GRAPH_ALGOS():
    INF=2**31
    UNVISITED = -1
    EXPLORED  = -2
    VISITED   = -3
    
    def __init__(self, V, E, N=None, M=None):
        self.num_edges = E
        self.num_nodes = V
        self.num_rows = N
        self.num_cols = M

    def init_structures(self): #take what you need and leave the rest
        from collections import deque
        self.adj_list = [{} for _ in range(self.num_nodes)]
        #self.edge_list = [(0,0,0)]*self.num_edges
        #self.matrix = [[0]*self.num_cols for _ in range(self.num_rows)]
    
        #self.queue = deque()
        #self.not_visited = set(list(range(self.num_nodes)))
        #self.visited = [UNVISITED]*self.num_nodes
        #self.stk = []
        #self.heap = []
        #self.dir_rc = [(1,0), (0,1), (-1,0), (0,-1)]
        #self.in_degree = [0]*self.num_nodes
        #self.color = [INF]*self.num_nodes
        #self.low_values = [0]*self.num_nodes
        #self.parent = [-1]*self.num_nodes
    
        #self.dist = [INF]*self.num_nodes
        #self.mst_node_set = []
        #self.topo_sort_node_set = []
        #self.articulation_points = []
    
    def append_edge_list(self, w, u, v):
        self.edge_list.append((w,u,v))
        self.num_edges += 1

    def update_adj_list(self, w, u, v):
        self.adj_list[u][v] = w
    
    def update_edge_list(self, edge, w, u, v):
        self.edge_list[edge] = (w,u,v)
        #uv = u*self.num_nodes + v; self.edge_list[edge] = (w,uv)

    def dfs_flood_fill(self, row, col, old_val, new_val): #needs test
        ans,self.matrix[row][col]=1,new_val
        for row_mod,col_mod in dir_rc:
            if 0<=row<self.num_rows and 0<=col<self.num_cols:
                if self.matrix[row][col]==old_val:
                    ans += self.dfs_flood_fill(row+row_mod, col+col_mod, old_val, new_val)
        return ans

    def dfs_topology_sort_helper(self, u):
        self.visited[u] = VISITED
        for v,w in self.adj_list[u]:
            if self.visited[v]==UNVISITED:
                self.dfs_topology_sort_helper(v)
        self.topo_sort_node_set.append(u)
        
    def dfs_topology_sort(self):
        self.topo_sort_node_set = []
        for u in range(self.num_nodes):
            if self.visited[v]==UNVISITED:
                self.dfs_topology_sort_helper(u)
        self.topo_sort_node_set = self.topo_sort_node_set[::-1]

    def dfs_bipartite_checker(self):
        pass # find code for this later

    def dfs_cycle_checker_helper(self, u):
        self.visited[u] = EXPLORED
        for v in self.adj_list[u]:
            info = '{} to {} is a '.format(u, v)
            if self.visited[v]==UNVISTED:
                print(info+'tree edge')
                self.parent[v] = u
                self.dfs_cycle_checker_helper(v)
            elif self.visited[v]==EXPLORED:
                if v == self.parent[u]:
                    print(info+'bidirectional edge')
                else:
                    print(info+'back edge')
            elif self.visited[v]==VISITED:
                print(info+'forward/crossedge')
        self.visited[u] = VISITED

    def dfs_cycle_checker(self):
        for u in range(self.num_nodes):
            if self.visited[u]==UNVISITED:
                self.dfs_cycle_checker_helper(u)

    def dfs_articulation_point_and_bridge_helper(self, u):
        self.visited[u] = self.dfs_counter
        self.low_values[u] = self.visited[u]
        self.dfs_counter += 1
        for v in self.adj_list[u]:
            if self.visited[v]==UNVISITED:
                self.parent[v] = u
                if u==self.dfs_root:
                    self.root_children += 1
                self.dfs_articulation_point_and_bridge_helper(v)
                if self.low_values[v] >= self.visited[u]:
                    self.articulation_points[u] = 1
                if self.low_values[v] > self.visited[u]:
                    print("bridge?")
                self.low_values[u] = min(self.low_values[u], self.low_values[v])
            elif v != self.parents[u]:
                self.low_values[u] = min(self.low_values[u], self.visited[v])

    def dfs_articulation_point_and_bridge(self):
        self.dfs_counter = 0
        for u in range(self.num_nodes):
            if self.visited[u]==UNVISITED:
                self.dfs_root = u
                self.root_children = 0
                self.dfs_articulation_point_and_bridge_helper(u)
                self.articulation_points[self.dfs_root] = (self.root_children>1)
        for i,u in enumerate(self.articulation_points):
            if u:
                print("vertix {}".format(i))

    def bfs_vanilla(self, start, end): #needs test
        from collections import deque
        self.queue.append(start); self.dist[start] = 0
        while queue:
            u = self.queue.popleft()
            for v in self.adj_list[u]:
                if self.dist[v]>self.dist[u]+1:
                    self.dist[v]=self.dist[u]+1
                    self.queue.append(v)

    def bfs_flood_fill(self, start_row, start_col, old_val, new_val):  #needs test
        self.stk.append(start_row, start_col)
        while self.stk:
            row,col = self.stk.pop()
            if 0<=row<self.num_rows and 0<=col<self.num_cols:
                if self.matrx[row][col]==old_val:
                    self.matrx[row][col] = new_val
                    for row_mod,col_mod in dir_rc:
                        self.stkk.append((row+row_mod, col+col_mod))

    def bfs_kahns_topological_sort(self):
        from heapq import heappush, heappop
        for list_of_u in self.adj_list:
            for v in list_of_u:
                self.in_degree[v] += 1
        for u in range(self.num_nodes):
            if 0==self.in_degree[u]:
                heapppush(self.heap, u)
        while self.heap:
            u = heappop(self.heap)
            for v in adj_list[u]:
                self.in_degree[v] -= 1
                if self.in_degree[v] <= 0:
                    heappush(self.heap, v)

    def bfs_bipartite_check_helper(self, start):
        from collections import deque
        self.queue.clear()
        self.color[start] = 0
        self.queue.append(start)
        is_bipartite = True
        while self.queue and is_bipartite:
            u = self.queue.popleft()
            for v in self.adj_list[u]:
                if self.color[v]!=INF:
                    is_bipartite = False
                    break
                self.color[v] = not self.color[u]

    def bfs_bipartite_check(self):
        for u in self.adj_list:
            if self.color[u] == INF:
                self.bfs_bipartite_check_helper(u)

    def bfs_cycle_checker(self):
        pass #need to get the implimentation 
        
    def sssp_dijkstras_heaps(self, start, end): #needs test
        from heapq import heappush, heappop
        heappush(self.heap, (0, start))
        self.dist[start] = 0
        self.parent[start] = start
        while self.heap:
            cur_dist, u = heappop(self.heap)
            if self.dist[u]<cur_dist:
                continue
            #if u==end: return cur_dist #uncomment this line for fast return
            for v, weight in self.adj_list[u].items():
                if self.dist[v]>cur_dist+weight:
                    self.dist[v] = cur_dist+weight
                    self.parent[v] = u
                    heappush(self.heap, (self.dist[v], v))
        return self.dist[end]

    def apsp_floyd_warshall(self): #needs test
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.matrix[i][j] = min(self.matrix[i][j], self.matrix[i][k]+self.matrix[k][j])

    def apsp_floyd_warshall_neg_cycles(self): #needs test
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for k in range(self.num_nodes):
                    if self.matrix[k][k]<0 and self.matrix[i][k]!=INF and self.matrix[k][j]!=INF:
                        self.matrix[i][j]=-INF

    #will kill the edge list but will save memory
    def mst_kruskals_heaps(self):  #needs test
        from heapq import heapify, heappop
        UF=UnionFind(self.num_nodes)
        heapify(self.edge_list)
        while self.edge_list and  UF.num_sets>1:
            w,u,v = heappop(self.edge_list) #use w, uv = ... for single cord storage
            #v,u = uv%self.num_nodes, uv//self.num_nodes
            if not UF.is_same_set(u,v):
                self.mst_node_set.append((w,u,v))
                UF.union_set(u,v)
        return self.mst_node_set
        
    def mst_prims_process_complete(self, u):  #needs test
        from heapq import heappush
        self.not_visited.remove(u)
        for v in self.not_visited:
            uv_dist = get_dist(u, v)
            if uv_dist<=self.dist[v]:
                self.dist[v] = uv_dist
                heappush(self.heap, (uv_dist, v, u))
    
    def mst_prims_process(self, u): #needs test
        from heapq import heappush
        self.not_visited.remove(u)
        for v, w in self.adj_list[u].items():
            if v in self.not_visited and w<=self.dist[v]:
                self.dist[v] = w
                heappush(self.heap, (w, v, u))
    
    def mst_prims(self):  #needs test
        from heapq import heappop
        self.prims_process(0)
        nodes_taken = 0
        while self.heap and nodes_taken<self.num_nodes:
            w,v,u = heappop(self.heap)
            if v in self.not_visited:
                self.prims_process(v)
                self.mst_node_set.append((w,v,u))
                nodes_taken += 1
        self.mst_node_set.sort()
        return self.mst_node_set

