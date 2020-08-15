from sys import stdin as rf
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
    if a>1: return F.RSQ(1,b)-F.RSQ(1,b-1)
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
