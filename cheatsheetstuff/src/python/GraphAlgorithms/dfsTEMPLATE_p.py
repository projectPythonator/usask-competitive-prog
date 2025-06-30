inf=10**9
V=1000 # or max graph size vertex
adjList=[[] for _ in range(V)]
seen=set()
def dfs(S,T):
    seen.add(S)
    for v in adjList[S]:
        if v not in seen:
            dfs(v,T)
