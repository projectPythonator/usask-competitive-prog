#tested on following kattis problems given by ids
# brexit(modified), pickupsticks
def topoSortBFS():
    global adj, V, res
    deg=[0]*V
    for u in adj:
        for v in u:
            deg[v]+=1
    q=deque()
    for i in range(V):
        if deg[i]==0:
            q.append(i)
    while len(q)>0:
        u=q.popleft()
        res.append(u)
        for v in adj[u]:
            deg[v]-=1
            if deg[v]==0:
                q.append(v)
