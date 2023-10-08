#data structurea
#union find 
class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))
        self.rank  = [0]*n
        self.sizes = [0]*n
        self.num_sets = n
        
    def find_set(self, u):
        u_parent = u
        u_children = []
        while u_parent != self.parents[u_parent]:
            u_children.append(u_parent)
            u_parent = self.parents[u_parent]
        for child in u_children:
            self.parents[c] = u_parent
        return u_parent
        
    def is_same_set(self, u, v):
        return (self.findSet(u)==self.findSet(v))

    def union_set(self, i, j):
        if not self.isSameSet(i, j):
            x = self.findSet(i)
            y = self.findSet(j)
            if self.rank[x]>self.rank[y]:
                self.p[y]=x
            else:
                self.p[x]=y
                if self.rank[x]==self.rank[y]:
                    self.rank[y]+=1
