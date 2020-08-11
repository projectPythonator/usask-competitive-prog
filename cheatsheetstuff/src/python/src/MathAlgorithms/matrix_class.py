
import math
from sys import stdin as rf

EPS=1e-7

class MATRIX_ALGOS:
  def __init__(F): pass
  
  #from mit 2008 cheat sheet
  def gauss_jordan(F, B):
    D,n,m=1,F.n,B.n
    for k in range(n):
      j=k
      for i in range(k+1,n): j=i if abs(F.M[i][j])>abs(F.M[j][k]) else j
      if abs(F.M[j][k])<EPS: print("error Matrix is singular")
      F.M[j],F.M[k]=F.M[k],F.M[j]; B.M[j],B.M[k]=B.M[k],B.M[j]
      if j!=k: D*=-1
      s=F.M[k][k]; D*=s
      for i in range(n): F.M[k][i]/=s
      for i in range(m): B.M[k][i]/=s
      for i in range(n):
        if i==k: continue
        t=F.M[i][k]
        for j in range(n): F.M[i][j]-=t*F.M[k][j]
        for j in range(m): B.M[i][j]-=t*B.M[k][j]
    return D
  #from mit 2008 cheat sheet
  def rref(F):
    n,m,r=F.n,F.m,0
    for c in range(m):
      j=r
      for i in range(r+1,n): j=i if abs(F.M[i][c])>abs(F.M[j][c]) else j
      if abs(F.M[j][c])<EPS: continue
      F.M[j],F.M[r]=F.M[r],F.M[j]; s=F.M[r][c]
      for i in range(m): F.M[r][i]/=s
      for i in range(n):
        if i==r: continue
        t=F.m[i][c]
        for j in range(m): F.M[i][j]-=t*F.M[r][j]
      r+=1
    return r
      
      
      
