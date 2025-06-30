V=1000
edgeList=[]
def kruskals():
    edgeList.sort()
    mst_cost=0
    #UF=UnionFind(V)
    for e in edgeList:
        if True: # change to UF.isSameSet(e[1][1], e[1][0]):
            mst_cost+=e[0]
            #UF.unionSet(e[1][1], e[1][0])

    #print costs
