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
