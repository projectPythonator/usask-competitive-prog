inf=10**9 # 1
V=1000 # 2
adjList=[[] for _ in range(V)] # 3
seen=set() # 4
def dfs(S,T): # 5
    # 6
    seen.add(S) # 7
    for v in adjList[S]: # 8
        if v not in seen: # 9
            # 10
            dfs(v,T) #11
    #12
