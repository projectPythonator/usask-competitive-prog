from sys import stdin as rf
from sys import setrecursionlimit as srl
from collections import deque
import heapq as h
INF=2000000000
class GRAPH_ALGOS(): #F == self use it because its shorter
  def __init__(F): F.G,F.D={},[(1,0),(-1,0),(0,1),(0,-1)]
  def init_adj(F, n): F.AL1={i:{} for i in range(n)}
  def init_ff(F, n,m,c): F.R=n;F.C=m;F.G=[[c]*m for _ in range(n)]
  def init_mat(F, n): #change G to H if G is already used
    F.V=n; F.G=[[INF]*n for _ in range(n)];  
    for i in range(n): F.G[i][i]=0
  
  def add_edge_adj(F, u,v,w): F.AL1[u][v]=w
  def add_edge_mat(F, u,v,w): F.G[u][v]=min(w, F.G[u][v])
  
  def dijkstras1(F, s,t): #remove par if too slow
    f=h.heappush;F.dst={i:INF for i in F.AL1}; F.par={i:i for i in F.AL1}
    F.dst[s]=0; F.par[s]=s; Q=[]; f(Q,(0,s))
    while Q:
      d,u=h.heappop(Q)
      if d>F.dst[u]: continue #if u==t: return for fast returns add below
      for v,w in F.AL1[u].items():
        if F.dst[v]>d+w: F.dst[v],F.par[v],Z=d+w,u,f(Q,(d+w,v))
    
  def dijkstra2(F, s,t):pass
  
  def ff1(F, r,c,a,b): #dfs version
    if (not (0<=r<F.R and 0<=c<F.C)) or F.G[r][c]!=a: return 0
    A,F.G[r][c]=1,b
    for X,Y in F.D: A+=F.ff1(r+X,c+Y,a,b)
    return A
  
  def ff2(F, i,j,a,b): #interative version
    Q=[(i,j)]
    while Q:
      r,c=Q.pop()
      if (not (0<=r<F.R and 0<=c<F.C)) or F.G[r][c]!=a: continue
      F.G[r][c]=b
      for X,Y in F.D: Q.append((r+X,c+Y))
  
  def topo_sort1(F, s):
    f=append;Q,B,A=deque(),[0]*len(F.AL1),[]
    for u in F.AL1:
      for v in u: B[i]+=1
    for i in F.AL1:
      if B[i]==0: Q.f(i)
    while Q:
      u=Q.pop(); A.f(u)
      for v in F.AL1:
        B[v]-=1
        if B[v]==0: Q.f(v)
    return A
  
  def topo_sort2(F, s): pass
  
  def mst_prims1(F): pass
  def mst_prims2(F):
    n=len(F.AL1); dst=[INF]*n; vis=[False]*n; mst=[]; msc=0; Q=[(0,0,0)]
    while Q and len(mst)<n:
      w,u,v=h.heappop(Q)
      if vis[u]: continue
      vis[u]=True; msc+=w; mst.append((u,ov))
      for nv,nw in F.AL1[u].items():
        if vis[nv] or nw>dst[nv]: continue
        dst[nv]=nw; h.heappush(Q,(nw,nv,u))
    if len(mst)<n: print("impossble") #not possible to make tree
    else: #msc is the cost, removes the first sorted point as its 0,0
      print(msc); A=[(u,v) if u<v else (v,u) for u,v in mst]; A.sort(); A=A[1:]
      for u,v in A: print(u,v) 
  
  def bipart_help(F, s):
    F.col[s]=0; f=append
    B=True;Q=deque();Q.f(s)
    while B and Q:
      u=Q.popleft()
      for v in F.AL1:
        if F.col[v]==INF: F.col[v],Z=1-F.col[u],Q.f(v)
        else:
          b=False; break
    return B #false if not bipateite 
  
  def bipartite(F):
    F.col=[INF]*len(F.AL1); A=True
    for i in F.AL1:
      if len(F.AL1[i])==0:F.AL1[i]=0
      else: A=F.bipart_help(i)
      if not A: return False #if we are checking for bipartinies 
    return True
  
  def bfs1(F, s,t): #return dst or make is a class var
    f=append; dst=[INF]*len(F.AL1); Q=deque(); dst[s]=0; Q.f(s)
    while Q:
      u=Q.popleft()
      for v in F.AL1[u]:
        if dst[v]>dst[u]+1: dst[v],Z=dst[u]+1,Q.f(u)
    return dst #unless you make it class var
  
  def tarjan_help(F, u):
    F.nc+=1; F.L[u]=F.N[u]=F.nc; F.STK.append(u); F.vis.add(u)
    for v in F.AL1[u]:
      if F.N[v]==INF:
        F.tarjan_help(v); F.L[u]=min(F.L[u], F.L[v])
      elif v in F.vis: F.L[u]=min(F.L[u], F.N[v])
    if VL[u]==V.N[u]:
      F.SCC.append(set(F.STK));F.STK=[]
  
  def tarjanSCC(F): #set stack size to needed lim
    srl(100000);F.L={i:INF for i in F.AL1};F.N={i:INF for i in F.AL1}
    F.STK=[];F.vis=set();F.SCC=[];F.nc=0
    for v in F.AL1:
      if F.N[v]==INF: F.tarjan_help(v)
  
  def kosaraju_help1(F, u):
    F.S.add(u)
    for v in F.AL1:
      if v not in F.S: F.kosaraju_help1(v)
    F.Q.append(u)
  
  def kosaraju_help2(F, u):
    F.S.add(u); F.SCC[u]=F.CN #add u to an scc here if we need too
    for v in F.AL2:
      if v not in V.S: F.kosaraju_help2(v)

  def kosarajuSCC(F):
    F.S=set(); F.Q=[]; F.CN=0
    for v in F.AL1:
      if v not in F.S: F.kosaraju_help1(v)
    F.S=set()
    while F.Q:
      while F.Q and F.Q[-1] in F.S: F.Q.pop()
      if F.Q:
        F.kosaraju_help2(F.Q[-1]); F.CN+=1
    
    
    
    
    
