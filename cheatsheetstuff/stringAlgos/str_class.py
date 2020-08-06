


class STR_ALGOS:
  def __init__(F): pass
  
  def prep_KMP(F, s): F.n,F.T=len(s),n
  def kmp_pre(F, s): 
    F.m=len(s); F.P=s; F.B=[0]*(F.m+1); F.B[0]=j=-1
    for i in range(F.m):
      while j>=0 and F.P[i]!=F.P[j]: j=F.B[j]
      j+=1; F.B[i+1]=j
  
  def kmp_find(F):
    A=[]; j=0
    for i in range(F.n):
      while j>=0 and F.T[i]!=F.P[j]: j=F.b[j]
      j+=1
      if j==F.m:
        A.append(1+i-j); j=F.B[j]
    return A #holds the indices that found at
  
  def prep_SA(F, s):
    k=1; F.T=s+chr(0)*(100010); F.n=len(s)+1; F.mx=max(300, F.n)
    F.TSA=[0]*F.n; F.TRA=[0]*F.n; F.RA=list(map(ord, F.T[:F.n]))
    F.SA=list(range(F.n)); F.K=[]
    while k<F.n:
      F.K.append(k); k+=k
  
  def count_sort(F, k):
    c=[0]*F.mx; t=0; c[0]+=(F.n-(F.n-k))
    for i in range(F.n-k): c[F.RA[i+k]]+=1
    for i in range(F.mx): c[i],t=t,t+c[i]
    for i in range(F.n):
      p=F.RA[F.SA[i]+k] if F.SA[i]+k<n else 0
      F.TSA[c[p]]=F.SA[i]; c[p]+=1
    for i, e in enumerate(F.TSA): F.SA[i]=e
  
  def build_SA(F):
    for k in F.K:
      F.count_sort(k); F.count_sort(0); F.TRA[F.SA[0]]=r=0
      for i in range(1, F.n):
        a,b=F.SA[i],F.SA[i-1]
        if not(F.RA[a]==F.RA[b] and F.RA[a+k]==F.RA[b+k]): r+=1
        F.TRA[a]=r
      for i,e in enumerate(F.TRA): F.RA[i]=e
      if F.RA[F.SA[-1]]==F.n-1:
        break
    F.TRA=[];F.SA=[];F.RA=[]
  
  def build_LCP(F): #A=phi,B=PLCP
    F.LCP=[0]*F.n; A=[0]*F.n; B=[0]*F.n; A[0]=-1; L=0
    for i in range(1, F.n): A[F.SA[i]]=F.SA[i-1]
    for i in range(F.n):
      if A[i]==-1:
        B[i]=0; continue
      while F.T[i+L]==F.T[A[i]+L]: L+=1
      B[i]=L; L=max(L-1,0)
    for i in range(F.n): F.LCP[i]=B[F.SA[i]]
  
  def build_LRS(F):
    A,mx=0,-1
    for i in range(1, F.n):
      if F.LCP[i]>mx: A,mx=i,F.LCP[i]
    return (mx,A)
  
  def f(F, I):
    return 1 if I<F.n-F.m-1 else 2
  
  def build_LCS(F):
    A,mx=0,-1
    for i in range(1, F.n):
      if F.f(F.SA[i])!=F.f(F.SA[i-1]) and F.LCP[i]>mx: A,mx=i,F.LCP[i]
    return (mx,A)
  
  def strncmp(F, I):
    for i in range(F.m):
      if F.P[i]!=F.T[I+i]: return ord(F.T[I+i])-ord(F.P[i])
    return 0
  
  def str_match1(F, s):
    F.P=s; F.m=len(s); a,b,c=0,F.n-1,0
    while a<b:
      c=(a+b)//2; r=F.strncmp(F.SA[c])
      if r>=0: b=c
      else: a=c+1
    if F.strncmp(F.SA[a])!=0: return (-1,-1)
    A=a; a,b,c=A,F.n-1,0
    while a<b:
      c=(a+b)//2; r=F.strncmp(F.SA[c])
      if r>0: b=c
      else: a=c+1
    if F.strncmp(F.SA[a])!=0: b-=1
    return (A,b)
    
    
    
