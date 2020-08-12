
import math
from sys import stdin as rf

EPS=1e-7

class MATRIX:
  def __init__(F, r,c): F.n,F.m,F.M=r,c,[[0]*c for _ in range(r)]

  # RM=RowMul, RS=RowSub, SR=SwapRows
  # copy paste code back into functions if needed for following functions
  def RM(F,a,b,c):
    for i in range(a): F.M[b][i]*=c
  def RS(F,a,b,c,d):
    for i in range(a): F.M[b][i]-=(c*F.M[d][i])
  def SR(F,a,b): F.M[a],F.M[b]=F.M[b],F.M[a]

  # partial pivoting from mit 2008 cheat sheet
  def gauss_jordan1(F, B):
    D,n,m=1,F.n,B.n
    for k in range(n):
      j=k
      for i in range(k+1,n): j=i if abs(F.M[i][j])>abs(F.M[j][k]) else j
      if abs(F.M[j][k])<EPS: print("error Matrix is singular")
      F.SR(j,k); B.SR(j,k)
      if j!=k: D=-D
      s=F.M[k][k]; D*=s; s=1/s; F.RM(n,k,s); B.RM(m,k,s)
      for i in range(n):
        if i==k: continue
        t=F.M[i][k]; F.RS(n,i,t,k); B.RS(m,i,t,k)
    return D
    
  # full pivot from https://github.com/t3nsor/codebook/blob/master/gaussian.cpp
  def gauss_jordan2(F, B):
    n,m=F.n,B.m; R=[0]*n; C=[0]*n; P=[0]*n; D=1
    for i in range(n):
      J=K=-1
      for j in range(n):
        if P[j]: continue
        for k in range(n):
          if P[k]: continue
          if J==-1 or abs(F.M[j][k])>abs(F.M[J][K]): J,K=j,k
      if abs(F.M[J][K])<EPS: return 0
      P[K]+=1; F.SR(J,K); B.SR(J,K); R[i]=J; C[i]=K
      if J!=K: D=-D # -D==(D*-1) right ?
      c=F.M[K][K]; D*=c; c=1/c; F.M[K][K]=1.0; F.RM(n,K,c); B.RM(m,K,c)
      for j in range(n):
        if j==K: continue
        c=F.M[j][K]; F.M[j][K]=0; F.RS(n,j,c,K); B.RS(m,j,c,K)
    for i in range(n-1, -1, -1):
      if R[i]==C[i]: continue
      for j in range(n): F.M[j][R[i]],F.M[j][C[i]]=F.M[j][C[i]],F.M[j][R[i]]
    return D
      
  #from mit 2008 cheat sheet
  def rref(F):
    n,m,r=F.n,F.m,0
    for c in range(m):
      j=r
      for i in range(r+1,n): j=i if abs(F.M[i][c])>abs(F.M[j][c]) else j
      if abs(F.M[j][c])<EPS: continue
      F.SR(j,r); s=1/F.M[r][c]; F.RM(m,r,s)
      for i in range(n):
        if i==r: continue
        t=F.m[i][c]; F.RS(m,i,t,r)
      r+=1
    return r
    
    #line one is chain mul line two is mat mul?
    def mat_mul(F, A,B,p,q,r):
      C=MATRIX(p,r)
      for i in range(p):
        for k in range(q):
          if A.M[i][k]==0: continue
          for j in range(r):
            #C.M[i][j]+=(A.M[i][k]+B.M[k][j])
            C.M[i][j]+=(A.M[i][k]*B.M[k][j])
      return C
    
    def mat_pow(F, B,p):
      A=MATRIX(B.n,B.m)
      for i in range(B.n): A.M[i][i]=1
      while p:
        if (1&p): A=A.mat_mul(A,B)
        B=B.mat_mul(B,B); p//=2
      return A
      
