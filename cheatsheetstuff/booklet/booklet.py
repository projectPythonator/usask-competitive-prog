#data structurea
#union find 
class UnionFind:
    def __init__(self, N):
        self.rank = [0]*N
        self.p =[i for i in range(N)]
    
    def findSet(self, i):
        if self.p[i]==i:
            return i
        else:
            self.p[i] = self.findSet(self.p[i])
            return self.p[i]
    def isSameSet(self, i, j):
        return (self.findSet(i)==self.findSet(j))
    def unionSet(self, i, j):
        if not self.isSameSet(i, j):
            x = self.findSet(i)
            y = self.findSet(j)
            if self.rank[x]>self.rank[y]:
                self.p[y]=x
            else:
                self.p[x]=y
                if self.rank[x]==self.rank[y]:
                    self.rank[y]+=1
