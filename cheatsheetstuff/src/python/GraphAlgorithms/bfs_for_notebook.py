from collections import deque
V=1000 # change to graph size
inf=10**10
adjList=[[] for _ in range(V)]
D=[inf]*V
def bfs(S, T):
    dist[S]=0 # init 3
    q=deque([S]) # init 4
    # init 5
    while len(q)>0: 
        cur=q.popleft() # init 6
        # init 7
        for v in adjList[cur]:
            if D[v]>D[cur]+1: # init 8
                D[V]=D[cur]+1 # init 9
                # init 10
                q.append(cur)

