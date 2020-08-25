from sys import stdin as rf
import math
#from steven and felix 3rd book
class UFDS1:
  def __init__(F, N): F.ns,F.S,F.R,F.P=N,[1]*N,[0]*N,list(range(N))
  def FS(F, i): F.P[i]=i if F.P[i]==i else F.FS(F.P[i]); return F.P[i]
  def ISS(F, i,j): return (F.FS(i)==F.FS(j))
  def US(F, i,j):
    if not F.ISS(i,j):
      F.ns-=1; x,y=F.FS(i),F.FS(j)
      if F.R[x]>F.R[y]: F.P[y]=x
      else:
        F.P[x]=y
        if F.R[x]==F.R[y]: F.R[y]+=1
  def NS(F): return F.ns

#from steven felix 4th book
class UFDS2:
  def __init__(F, N): F.ns,F.S,F.R,F.P=N,[1]*N,[0]*N,list(range(N))
  def FS(F, i):
    I,C=i,[]; f=C.append
    while I!=F.P[I]: f(I); I=F.P[I]
    for c in C: F.P[c]=I
    return I
  def US(F, a,b):
    A,B=F.FS(a),F.FS(b)
    if A==B: return
    if F.R[A]<F.R[B]: F.P[A]=B; F.S[B]+=F.S[A]
    elif F.R[B]<F.R[A]: F.P[B]=A; F.S[A]+=F.S[B]
    else: F.P[B]=A; F.R[A]+=1; F.S[A]+=F.S[B]
    F.ns-=1
  def SS(F, x): return F.S[F.FS(x)]

#based on steven and felix book 4 implmentation
class FenwickTree:
  def __init__(F, f):
    F.n,F.T=len(f),[0]*(len(f)+1)
    for i in range(1, F.n+1):
      F.T[i]+=f[i-1]
      if i+F.LSB(i)<=F.n: F.T[i+F.LSB(i)]+=F.T[i]
    
  def LSB(F, a): return a&(-a)

  def RSQ(F, a,b):
    if a>1: return F.RSQ(1,b)-F.RSQ(1,a-1)
    s=0
    while b>0: s+=F.T[b]; b-=F.LSB(b)
    return s
  
  def RU(F, i,v):
    while i<=F.n: F.T[i]+=v; i+=F.LSB(i)
  
  def select(F, k):
    p=1;i=0
    while (p*2)<F.n: p*=2
    while p:
      if k>F.T[i+p]: k-=F.T[i+p]; i+=p
      p//=2
    return i+1

#from felix and steven book use for when you need to update ranges
class RUPQ:
  def __init__(F, n): F.ft=FenwickTree([0]*n)
  def RU(F, a,b,v): F.ft.RU(a,v); F.ft.RU(b+1,-v)
  def PQ(F, a): return F.ft.RSQ(1,a)

class RURQ:
  def __init__(F, n): F.ft,F.rt=FenwickTree([0]*n),RUPQ(n)
  def RU(F, a,b,v): F.rt.RU(a,b,v); F.ft.RU(a, v*(a-1)); F.ft.RU(b+1, -v*b)
  def PQ(F, a,b): 
    return F.PQ(1,b)-F.PQ(1,a-1) if a>1 else F.rt.PQ(b)*b-F.ft.RSQ(1,b)

class SEGMENTTREE1:
  def __init__(F,a):
    F.n=len(a); F.T=[0]*(4*F.n); F.Z=[-1]*(4*F.n); F.A=[i for i in a]
    F.B(1,0,F.n-1)
  
  def L(F, a): return 2*a
  def R(F, a): return 2*a+1
  def C(F, a,b): max(a,b) if a==-1 or b==-1 else min(a,b)
  
  def B(F, p,l,r):
    if l==r: F.T[p]=F.A[l]
    else:
      m=(l+r)//2; F.B(F.L(p),l,m); F.B(F.R(p),m+1,r) 
      F.T[p]=F.C(F.T[F.L(p)],F.T[F.R(p)]) 
  
  def P(F, p,l,r):
    if F.Z[p]>-1: 
      F.T[p]=F.Z[p]
      if l!=r: F.Z[F.L(p)]=F.Z[F.R(p)]=F.Z[p]
      else: F.A[l]=F.Z[p]
      F.Z[p]=-1
  
  def RMQ(F, p,l,r,i,j):
    F.P(p,l,r)
    if i>j: return -1
    if l>=i and r<=j: return F.T[p]
    m=(l+r)//2
    return F.C(F.RMQ(F.L(p),l,m,i,min(m,j)),F.RMQ(F.R(p),m+1,r,max(i,m+1),j))
  
  def UPDATE(F, p,l,r,i,j,v):
    F.P(p,l,r)
    if i>j: return
    if l>=i and r<=j: F.Z[p]=v; F.P(p,l,r)
    else:
      m=(l+r)//2
      F.UPDATE(F.L(p),l,m,i,min(m,j),v); F.UPDATE(F.R(p),m+1,r,max(i,m+1),j,v)
      a=F.Z[F.L(p)] if F.Z[F.L(p)]!=-1 else F.T[F.L(p)]
      b=F.Z[F.R(p)] if F.Z[F.R(p)]!=-1 else F.T[F.R(p)]
      F.T[p]=F.T[F.L(p)] if a<=b else F.T[F.R(p)]
  
#currently based on steven and felix version
class SPARSE_TABLE_DS: # renamed shorted after 
  def __init__(F, A):
    n=len(A); N=int(math.log2(n))+1; F.A=A; F.P2=[0]*(N+1); F.L2=[0]*(2**N+1)
    for i in range(N+1): F.P2[i],F.L2[2**i]=2**i,i
    for i in range(2, F.P2[N]):
      if F.L2[i]==0: F.L2[i]=F.L2[i-1]
    F.ST=[[None]*n for _ in range(F.L2[n]+1)]
    for i in range(n): F.ST[0][i]=i
    for i in range(1, N+1):
      a=F.P2[i-1]
      for j in range(n-F.P2[i]+1):
        x,y=F.ST[i-1][j],F.ST[i-1][j+a]
        F.ST[i][j]=x if A[x]<=A[y] else y
  
  def RMQ(F,i,j):
    k=F.L2[j-i+1]; x,y=F.ST[k][i],F.ST[k][j-F.P2[k]+1]
    return x if F.A[x]<=F.A[y] else y
