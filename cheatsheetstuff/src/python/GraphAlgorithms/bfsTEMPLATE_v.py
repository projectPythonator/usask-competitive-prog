from collections import deque
V=1000 # change to graph size
inf=10**10
adjList=[[] for _ in range(V)]
D=[inf]*V
def bfs(S, T):
    dist[S]=0
    q=deque([S])
    while len(q)>0:
        cur=q.popleft()
        for v in adjList[cur]:
            if D[v]>D[cur]+1:
                D[V]=D[cur]+1
                q.append(cur)

