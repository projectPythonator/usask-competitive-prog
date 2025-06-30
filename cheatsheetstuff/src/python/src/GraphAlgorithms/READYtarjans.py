# author agis daniels
# this is the template for tarjans finds scc
# note OFTEN times this exact template doesnt work for python due to
# recursion/stack overflow
# used to solve loopycabdrivers(c++), need more testing
#unless i just forgot some but it should work 
def tarjanSCC(u):
    global low, num, num_cnt, stk, vis
    global UNVIS
    num_cnt+=1
    low[u]=num[u]=num_cnt
    stk.append(u)
    vis[u]=True
    for v in adj[u]:
        if num[v]==UNVIS:
            tarjanSCC(v)
            low[u]=min(low[v],  low[u])
        elif vis[v]:
            low[u]=min(num[v],  low[u])
    if low[u]==num[u]:
        # the stk holds all the vtx
        # of the connected components

#call this to do tarjans on all nodes
def helper():
    vis=[False]*V
    low=[UNVIS]*V
    num=[UNVIS]*V
    num_cnt=0
    for i in range(V):
        if num[i]==UNVIS:
            tarjanSCC(i)
