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
    
    def init_edge_list(self):
        #self.edge_list = [(0,0,0)]*self.num_edges
        self.edge_list = [(0,0)]*self.num_edges
    
    def append_edge(self, w, u, v):
        self.edge_list.append((w,u,v))
        self.num_edges += 1
    
    def update_edge(self, edge, w, u, v):
        uv = u*self.num_nodes + v
        #self.edge_list[edge] = (w,u,v)
        self.edge_list[edge] = (w,uv)
    
    #will kill the edge list but will save memory
    def kruskals_via_heaps(self):
        UF=UnionFind(self.num_nodes)
        heapify(self.edge_list)
        edges_left=0
        mst = []
        while self.edge_list:
            w,uv = heappop(self.edge_list)
            v = uv%self.num_nodes
            uv //= self.num_nodes
            u = uv
            if UF.is_same_set(u,v):
                continue
            mst.append((w,u,v))
            UF.union_set(u,v)
            if UF.amt_of_sets()==1:
                break
        self.edge_left = None
        mst.sort()
        return mst[-1][0]


