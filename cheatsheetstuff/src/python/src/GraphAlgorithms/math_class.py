import math as M
from sys import stdin as rf

class MATH_ALGOS:
  def __init__(self): pass
  def set_N(self, n): self.N=n
  def is_prime1(self, n):
    if n<4: return n>1
    if n%2==0 or n%3==0: return False
    p=5
    while p*p<=n:
      if n%p==0 or n%(p+2)==0: return False
      p+=6
    return True
  def siv_prime(self, n):
    F=self; F.P,p=[True]*n,[2]
    for i in range(4, n, 2): p[i]=False
    for i in range(3, n, 2):
      if p[i]:
        F.P.append(i)
        for j in range(i*i, n, 2*i): p[j]=False
  
  def set_frm_prime(self): self.UP=set(self.P)
  
  def prim_fact(self, n):
    F=[]
    for p in self.P:
      if p*p>n: break
      if n%p==0:
        while n%p==0:
          n//=p; F.append(p)
    if n>1: F.append(n)
    return F
  
  def is_cmp(self, a,d,n,s):
    if pow(a,d,n)==1: return False
    for i in range(s):
      if pow(a,2**i*d,n)==n-1: return False
    return True
  def is_prime2(self, n, D=16):
    F,S,d,s=self,3215031751,n-1,0
    if n in F.UP: return True
    if any((n%F.P[p]==0) for p in range(50)) or n<2 or n==S:return False
    while not (1&d): d,s=d//2,s+1
    for i,B in enumerate(F.I,2):
      if n<B: return not any(F.is_cmp(F.MP[j],d,n,s) for j in range(i))
    return not any(F.is_cmp(F.P[j],d,n,s) for j in range(D))
  def prep(self):
    F=self; F.siv_prime(1000); F.set_frm_prime(); F.MP=[2,3,5,7,11,13,17]
    F.I=[1373653,25326001,118670087467,2152302898747,3474749660383,341550071728321]
  
  def pow_mod(self, b,e,m):
    x=1
    while e>0:
      b, e, x=b*b%m, e//2, b*x%m if 1&e else x
    return x
  def gcd(self, u,v):
    while v: u,v=v,u%v
    return abs(u)
  def lcm(self, u,v): return u*v//self.gcd(u,v)
  
  #test this its from stanford icpc 2013-14 #givs all solutions ax=b(mod n)
  def mod(self, a,b): return ((a%b)+b)%b
  def extEuc(self, a,b):
    if b==0: return 1,0,a
    x,y,d=self.extEuc(b, a%b); return y, x-y*(a//b), d
  
  #test this its from stanford icpc 2013-14 #givs all solutions ax=b(mod n)
  def mod_lin_eq_sol(self, a,b,n):
    F=self; x,y,d=F.extEuc(a,n)
    if b%d==0:
      x=F.mod(x*(b//d),n); return [F.mod(x+i*(n//d), n) for i in range(d)]
    return []
  
  #test this its from stanford icpc 2013-14 #givs b in ab = 1 (mod n)
  def mod_inv(self, a,n):
    x,y,d=self.extEuc(a,b); return  -1 if d>1 else self.mod(x,n)
  
  #tested on same as chinese_remainder_theorem
  def crt_help(self, x,a,y,b):
    s,t,d=self.extEuc(x,y)
    return (0,-1) if a%d!=b%d else (self.mod(s*b*x+t*a*y,x*y)//d,x*y//d)
  def crt(self, x,a):
    A=(a[0],x[0])
    for i in range(1, len(x)):
      A=self.crt_help(A[1],A[0],x[i],a[i])
      if A[1]==-1: break
    return A
  
  def lin_dioph(self, a,b,c):
    d=self.gcd(a,b)
    if c%d==0:
      x=c//d*self.mod_inv(a//b, b//d); return (x,(c-a*x//b))
    return (-1,-1)
  
  def fib1(self, n): #iterative 
    A=[0]*(n+1); A[1]=1
    for i in range(2, n+1): A[i]=A[i-1]+A[i-2]
    self.fibs=A; return A
  
  def fib2(self, n): #logn rec version
    F=self
    if n==0: return 0
    if n==1 or n==2:
      F.fib[n]=1; return 1
    if n in F.fib: return F.fib[n]
    if n&1:
      k=(n+1)//2; a,b=F.fib2(k),F.fib2(k-1); F.fib[n]=a*a+b*b
    else:
      k=n//2; a,b=F.fib2(k),F.fib2(k-1); F.fib[n]=(2*b+a)*a
    return F.fib[n]
  
  def fib_logn(self, n): self.fib={}; return fib2(n)
  
  def cat1(self, n): #iter verision
    F=self; F.cat=[0]*(n+1); nxt=0; F.cat[0]=1
    for i in range(n): F.cat[i+1]=F.cat[i]*(4*i+2)//(i+2)
  
  def cat2(self, n,p):
    from collections import Counter as g
    F=self; f=F.prim_fact; T,B={},{}; F.siv_prime(int((5*n)**0.5)); A=1
    for i in range(n):
      a=f(4*i+2); D=g(a)
      for k,v in D.items():
        if k not in T: T[k]=v
        else: T[k]+=v
      a=f(i+2); D=g(a)
      for k,v in D.items():
        if k not in B: B[k]=v
        else: B[k]+=v
    for k,v in B.items(): T[k]-=v
    for k,v in T.itmes(): 
      if v>0: A*=pow(k,v,p)
    return A%p
  
  def bin_coe1(self, n,k): #iter version fastish for one O(k) space contsant
    k=min(k,n-k); A=1
    for i in range(k):
      A*=(n-i); A//=(i+1)
    return A
  
  def bin_coe2(self, n,k): #dp verision O(n^2) i think
    F=self; f=F.bin_coe2
    if n==k or k==0: return 1
    if (n,k) not in F.bin: F.bin[(n,k)]=f(n-1,k)+f(n-1,k-1)
    return F.bin[(n,k)]
    
  def prep_bin(self): self.bin={}
  
  def fft(F, A):
    n=len(A); N=n//2
    if n==1: return
    L,R=[None]*(n//2),[None]*(n//2)
    for i in range(N): L[i],R[i]=A[2*i],A[2*i+1]
    F.fft(L); F.fft(R)
    for i in range(N): 
      x=complex(M.cos(2*M.pi*i/n),M.sin(2*M.pi*i/n))
      A[i],A[i+N]=L[i]+x*R[i], L[i]-x*R[i]
    
  def ifft(F, A):
    for i in range(len(A)): A[i]=A[i].conjugate()
    F.fft(A)
    for i in range(len(A)): A[i]=A[i].conjugate()
    for i in range(len(A)): A[i]/=len(A)
  
  def mul_poly(F, a,b):
    n,l=1,len(a)+len(b)-1
    while n<l: n*=2
    A=[complex(0)]*n; B=[complex(0)]*n; C=[complex(0)]*n
    for i in range(len(a)): A[i]=complex(a[i])
    for i in range(len(b)): B[i]=complex(b[i])
    F.fft(A); F.fft(B)
    for i in range(n): C[i]=A[i]*B[i]
    F.ifft(C); R=[round(c.real) for c in C]
    R=R[:l]
    return R
