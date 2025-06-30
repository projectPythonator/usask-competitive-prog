from sys import stdin as rf
#dijkstras tested shortestpath1, shortestpath2(modified works), flowerytrails(modified works), fulltank(modified)
#APSP tested on allpairspath, kastenlauf(modified), arbitrage

'''
put list of tested stuff in a different file
Algorithms in this class
E=num edges
V=num vertices
Dijkstras SSSP (Dijkstras) O((E+V)LogV) our version okay with negative weights but not negative cycles
FlyodWarshalls (APSP) O(V^3) time O(V^2) space the 4-5 liner, also has many mods that can change it up
flood fill dfs and bfs(ff_dfs, ff_bfs) O(V+E) good for 2d grid fill stuffs just a variation on bfs or dfs
topology_sort_bfs O(V+E) just the bfs modified for topology sorting

mst_prims_faster O((E+V)LogV) is a dijkstra varient of prims that has a decent run time for python(still gets beat out by large dataset tho)
bipartite_colour O(V+E) colours a graph in two colours and tells us if it cant be done
bfs_vanilla O(V+E) does a bfs search on the graph modified to also solve sssp on unweighted graphs
tarjanSCC O(V+E) for finding strongly connected components in a directed graph 
kosarajuSCC O(V+E) alternitve version for finding strongly connected components in a directed graph 
'''

INF=2000000000
class GRAPH_ALGOS():
    def __init__(self):
        self.adj={}
        
    def init_ff(self, n, m, c):
        self.R,self.C=n,m
        self.mat=[[c]*m for _ in range(n)]
    
    def init_mat(self, n):
        self.V=n
        self.mat=[[INF]*n for _ in range(n)]
        for i in range(n): self.mat[i][i]=0
    
    #modify the bottom two as needed
    def init_adj(self, n):
        self.adj={i:{} for i in range(n)}
    
    def add_edge_adj(self, u, v, w):
        self.adj[u][v]=w
        
    def add_edge_mat(self, u, v, w):
        self.mat[u][v]=min(w, self.mat[u][v])
    
    def dijkstras(self, s, t):
        import heapq as h
        self.dst={i:INF for i in self.adj}; self.dst[s]=0
        self.par={i:i for i in self.adj}; self.par[s]=s
        pq=[]; h.heappush(pq, (0,s))
        while pq:
            d,u=h.heappop(pq)
            if d>self.dst[u]: 
                continue
            #if u==t: return d #uncomment this line for fast return
            for v,w in self.adj[u].items():
                if self.dst[v]>d+w:
                    self.dst[v],self.par[v]=d+w,u
                    h.heappush(pq, (d+w, v))
    
    def APSP(self):
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    #if self.matrix[i][k]<INF and self.matrix[k][j]<INF: #use for negative cycles check
                    self.mat[i][j]=min(self.mat[i][j], self.mat[i][k]+self.mat[k][j])
    
    def APSP_neg_cycles(self):
        for i in range(self.V):
            for j in range(self.V):
                for k in range(self.V):
                    if self.mat[k][k]<0 and self.mat[i][k]!=INF and self.mat[k][j]!=INF:
                        self.mat[i][j]=-INF
    
    def ff_dfs(self, r, c, a, b):
        if not 0<=r<self.R and 0<=c<self.C: return 0
        if self.mat[r][c]!=a: return 0
        ans,self.mat[r][c]=1,b
        for rr,cc in drc: ans+=self.ff_dfs(r+rr,c+cc,a,b)
        return ans
    
    def ff_bfs(self, i, j, a, b):
        stk=[(i,j)]
        while stk:
            r,c=stk.pop()
            if not 0<=r<self.R and 0<=c<self.C: continue
            if self.mat[r][c]!=a: continue
            self.mat[r][c]=b
            for rr,cc in drc: stk.append((r+rr,c+cc))
    
    #look up how to start from one source node 
    def topology_sort_bfs(self,s):
        from collections import deque
        q, deg, res=deque(), [0]*len(self.adj), []
        for u in self.adj: 
            for v in u: deg[i]+=1
        for i in self.adj: 
            if deg[i]==0: q.append(i)
        while q:
            u=q.pop(); res.append(u)
            for v in self.adj:
                deg[v]-=1
                if deg[v]==0: q.append(v)
    
    def topolog_sort_dfs(self, s):
        #fill in later 
        pass
    
    #falls to neg weights 
    def mst_prims_faster(self):
        import heapq as h
        n=len(self.adj)
        dst, vis, mst, msc=[INF]*n, [False]*n, [], 0
        pq=[(0,0,0)]
        while pq and len(mst)<n:
            w,u,ov=h.heappop(pq)
            if vis[u]: continue
            vis[u], msc=True, msc+w
            mst.append((u, ov))
            for nv,w in self.adj[u].items():
                if vis[nv] or w>dst[nv]: continue
                dst[nv]=w
                h.heappush(pq, (w,nv,u))
        if len(mst)<n:
            print("Impossible")
        else:
            print(msc)
            ans=[(min(u,v), max(u, v)) for u, v in mst[1:]]
            ans.sort()
            for u, v in ans:
                print(u,v)
    
    def bipartite_helper(self, s):
        from collection import deque
        self.col[s]=0
        if len(self.adj[s])==0: return
        b, q=True, deque(); q.append(s)
        while b and q:
            u=q.popleft()
            for v in self.adj[u]:
                if self.col[v]!=INF:
                    b=False
                    break # can we remove this into a return stat
                self.col[v]=not self.col[u]
    
    def bipartite_colour(self):
        self.col=[INF]*len(self.adj)
        for i in self.adj:
            if self.col[i]==INF:
                self.bipartite_helper(i)
    
    def bfs_vanilla(self, s, t):
        from collections import deque
        dst=[INF]*len(self.adj)
        q, dst[s]=deque(), 0; q.append(s)
        while q:
            u=q.popleft()
            for v in self.adj[u]:
                if dst[v]>dst[u]+1:
                    dst[v]=dst[u]+1
                    q.append(u)
                    
    def tarjan_helper(self, u):
        self.num_cmp+=1
        self.low[u]=self.num[u]=self.num_cmp
        self.stk.append(u); self.vis.add(u)
        for v in self.adj[u]:
            if self.num[v]==INF:
                self.tarjan_helper(v)
                self.low[u]=min(self.low[u], self.low[v])
            elif v in vis:
                self.low[u]=min(self.low[u], self.num[v])
        if self.low[u]==self.num[u]:
            self.SCC.append(set(self.stk))
            self.stk=[]
            #remove vis too?
    
    
    def tarjanSCC(self):
        from sys import setrecursionlimit
        setrecursionlimit(100000)
        self.low={i:INF for i in self.adj}
        self.num={i:INF for i in self.adj}
        self.stk, self.vis, self.SCC=[], set(), []
        self.num_cmp=0
        for v in self.adj:
            if self.num[v]==INF:
                self.tarjan_helper(v)
    
    def kosaraju_helper1(self, u):
        self.seen.add(u)
        for v in self.adj:
            if v not in self.seen:
                self.kosaraju_helper1(v)
        self.stk.append(u)
    
    def kosaraju_helper2(self, u):
        self.seen.add(u)
        self.scc[u]=self.curNum
        for v in self.adjT:
            if v not in self.seen: 
                self.kosaraju_helper2(u)
                
    def kosarajuSCC(self):
        self.seen=set()
        self.stk=[]
        for v in self.adj:
            if v not in self.seen:
                self.kosaraju_helper1(v)
        self.seen=set()
        self.curNum=0
        while self.stk:
            while self.stk and self.stk[-1] in self.seen: self.stk.pop()
            if self.stk:
                self.kosaraju_helper2(self.stk[-1])
                self.curNum+=1
                
    def bellmanford(self):
        self.dst={i:INF for i in self.adj}
        
