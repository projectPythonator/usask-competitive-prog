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

INF=2**31
class GRAPH_ALGOS():
    def __init__(self, V, E):
        self.num_edges = E
        self.num_nodes = V

    def init_structures(self): #take what you need and leave the rest
        self.adj_list = [{} for _ in range(self.num_nodes)]
        #self.edge_list = [(0,0,0)]*self.num_edges
    
        #self.queue = deque()
        #self.not_processed = set(list(range(self.num_nodes)))
    
        #self.dist = [INF]*self.num_nodes
        #self.mst_node_set = []
    
    def append_edge_list(self, w, u, v):
        self.edge_list.append((w,u,v))
        self.num_edges += 1

    def update_adj_list(self, w, u, v):
        self.adj_list[u][v] = w
    
    def update_edge_list(self, edge, w, u, v):
        self.edge_list[edge] = (w,u,v)
        #uv = u*self.num_nodes + v; self.edge_list[edge] = (w,uv)

    def bfs_vanilla(self, start, end):
        from collections import deque
        self.queue.append(start); self.dist[start] = 0
        while queue:
            u = self.queue.popleft()
            for v in self.adj_list[u]:
                if self.dist[v]>self.dist[u]+1:
                    self.dist[v]=self.dist[u]+1
                    self.queue.append(v)
    
    #will kill the edge list but will save memory
    def kruskals_heaps_mst(self):
        UF=UnionFind(self.num_nodes)
        heapify(self.edge_list)
        while self.edge_list and  UF.num_sets>1:
            w,u,v = heappop(self.edge_list) #use w, uv = ... for single cord storage
            #v,u = uv%self.num_nodes, uv//self.num_nodes
            if not UF.is_same_set(u,v):
                self.mst_node_set.append((w,u,v))
                UF.union_set(u,v)
        return self.mst_node_set
        
    def prims_process_complete(self, u):
        self.not_processed.remove(u)
        for v in self.not_processed:
            uv_dist = get_dist(u, v)
            if uv_dist<=self.dist[v]:
                self.dist[v] = uv_dist
                heappush(self.edge_list, (uv_dist, v, u))
    
    def prims_process(self, u):
        self.not_processed.remove(u)
        for v, w in self.adj_list[u].items():
            if v in self.not_processed and w<=self.dist[v]:
                self.dist[v] = w
                heappush(self.edge_list, (w, v, u))
    
    def prims_mst(self):
        self.prims_process(0)
        nodes_taken = 0
        while self.edge_list and nodes_taken<self.num_nodes:
            w,v,u = heappop(self.edge_list)
            if v in self.not_processed:
                self.prims_process(v)
                self.mst_node_set.append((w,v,u))
                nodes_taken += 1
        self.mst_node_set.sort()
        return self.mst_node_set

