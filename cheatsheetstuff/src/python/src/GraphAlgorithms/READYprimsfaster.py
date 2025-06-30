#matrix can be a adj list or an adj matrix but needs to work in this manner of hash table
#tested on lostmap, cats, need to translate it to more problems later lots i just used kruskals in c++ ^-^
def mstPrimsfaster():
    global pq, taken, mat, disto, inf
    taken=[False]*V
    disto=[inf]*V
    pq=[(0,0,0)]
    msc=0
    mst=[]
    while pq and len(mst)<V:
        w,v,u=h.heappop(pq)
        if not taken[v]:
            msc+=w
            mst.append((u+1, v+1))
            taken[v]=True
            for nv in mat:
                if nv==v or taken[nv]:
                    continue
                dist=mat[v][nv]
                if dist>disto[nv]:
                    continue
                disto[nv]=dist
                h.heappush(pq, (dist,nv,v))
