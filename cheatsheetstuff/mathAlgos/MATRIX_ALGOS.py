
import math
from sys import stdin as rf

EPS=1e-10
NUM_SIG=12

class MATRIX_ALGOS:
    def __init__(self, inMat):
        self.N,self.M=len(inMat),len(inMat[0])
        self.mat=[[el for el in row] for row in inMat]
    
    def epscmp0(self, x):
        return -1 if x<-EPS else 1 if x>EPS else 0
    
    def epscmpab(self, a, b):
        return epscmp0(a-b)
    
    def gauss_jordan(self, b):
        n,m=self.N,b.M
        irow,icol,ipiv,det=[0]*n,[0]*n,[0]*n,1
        det=1
        for i in range(n):
            pj=pk=-1
            for j in range(n):
                if ipiv[j]: continue
                for k in range(n):
                    if ipiv[k]: continue
                    if pj==-1 or math.fabs(self.mat[j][k])>math.fabs(self.mat[pj][pk]): pj,pk=j,k
            if self.epscmp0(self.mat[pj][pk])==0:
                print("matrix is singular")
                return None
            ipiv[pk]+=1
            self.mat[pj],self.mat[pk]=self.mat[pk],self.mat[pj]
            b.matrix[pj],b.matrix[pk]=b.matrix[pk],b.matrix[pj]
            if pj!=pk: det*=-1
            irow[i],icol[i]=pj,pk
            
            c,det=1.0/self.mat[pk][pk],det*self.mat[pk][pk]
            self.mat[pk][pk]=1.0
            for p in range(n): self.mat[pk][p]*=c
            for p in range(m): b.matrix[pk][p]*=c
            for p in range(n):
                if p!=pk:
                    c=self.mat[p][pk]
                    self.mat[p][pk]=0
                    for q in range(n): self.mat[p][q]-=(self.mat[pk][q]*c)
                    for q in range(m): b.matrix[p][q]-=(b.matrix[pk][q] * c)
        for p in range(n-1, -1, -1):
            if irow[p]==icol[p]: continue
            for k in range(n): self.mat[k][irow[p]],self.mat[k][icol[p]]=self.mat[k][icol[p]],self.mat[k][irow[p]]
        for i in range(n):
            for j in range(n): self.mat[i][j]=round(self.mat[i][j],NUM_SIG)
            for j in range(m): b.matrix[i][j]=round(b.matrix[i][j], NUM_SIG)
        return round(det,NUM_SIG)
    
def main():
    A=[[1,2,3,4],[1,0,1,0],[5,3,2,4],[6,1,4,6]]
    B=[[1,2],[4,3],[5,6],[8,7]]
    a=MATRIX_ALGOS(A)
    b=MATRIX_ALGOS(B)
    det=a.gauss_jordan(b)
    print(det)
    for row in a.mat:
        print(row)
    print()
    for row in b.mat:
        print(row)
main()
