#author agis daniels
V=0
INF=10**10
adj=[]
col=[]
#bipartite coloring a graph # can used to check with a rval 
#template used for id breakingbad, molekule, hoppers(part of it), pubs(modified)
def bipartite(s):
    global col
    col[s]=0
    q=deque()
    q.append(s)
    b=True
    while b and q:
        u=q.popleft()
        for v in adj[u]:
            if col[v]==INF:
                col[v]=1-col[u]
                q.append(v)
            elif col[v]==col[u]:
                b=False
                break
#helper to colour all the nodes can stay here or go in main
def helper():
    for i in range(V):
        if col[i]==INF:
            if len(adj[i])==0:
                col[i]=0
            else:
                bipartite(i) # modify if it needs rval
