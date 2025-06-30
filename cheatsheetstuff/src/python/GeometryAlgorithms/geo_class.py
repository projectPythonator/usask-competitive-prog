#code taken from foreverbell, steven felix halim book 3 and 4
from sys import stdin as rf
import math as M
EPS=1e-12
NUM_SIG=2
class pt_xyz:
  def __init__(self, a, b, c): self.x,self.y,self.z=a,b,c
  def __add__(self, b): return pt_xyz(self.x+b.x, self.y+b.y, self.z+b.az)
  def __add__(self, b): return pt_xyz(self.x-b.x, self.y-b.y, self.z-b.z)
  def __mul__(self, c): return pt_xyz(self.x*c, self.y*c, self.z*c)
  def __lt__(self, b): return ((self.x,self.y,self.z)<(b.x,b.y,b.z))
  def __eq__(self, b): return ((self.x,self.y,self.z)==(b.x,b.y,b.z))
  def __str__(self): "{} {} {}".format(self.x, self.y, self.z)
  def __round__(self, n): return pt_xyz(round(self.x,n),round(self.y,n),round(self.z,n))
  def __hash__(self):return hash((self.x,self.y,self.z))

class pt_xy:
  def __init__(self, a,b):self.x,self.y=map(round,[a,b])
  def __add__(self, b): return pt_xy(self.x+b.x, self.y+b.y)
  def __sub__(self, b): return pt_xy(self.x-b.x, self.y-b.y) 
  def __mul__(self, c): return pt_xy(self.x*c, self.y*c)
  def __truediv__(self, c): return pt_xy(self.x/c, self.y/c)
  def __floordiv__(self, c): return pt_xy(self.x//c, self.y//c)
  def __lt__(self, b): return ((self.y, self.x)<(b.y, b.x))
  def __eq__(self, b): return ((self.x,self.y)==(b.x,b.y))
  def __str__(self): return "{} {}".format(self.x, self.y)
  def __round__(self, n): return pt_xy(round(self.x,n),round(self.y,n))
  def __hash__(self):return hash((self.x,self.y))

class GEO_ALGOS:
  def __init__(self): pass
  def epscmp(self, x): return -1 if x<-EPS else 1 if x>EPS else 0
  def c_cmp(self, a,b): return 0 if M.isclose(a, b) else -1 if a<b else 1
  
  def dot(self, a,b):   return a.x*b.x+a.y*b.y
  
  def cross(self, a,b): return a.x*b.y-a.y*b.x
  
  def dst1(self, a, b): return M.hypot((a-b).x, (a-b).y)
  def dst2(self, a, b): return self.dot(a-b,a-b)
  
  def rot_cw90(self, p):  return pt_xy( p.y,-p.x)
  def rot_ccw90(self, p): return pt_xy(-p.y, p.x)
  def rot_ccw(self, p,t): 
    ct,st=M.cos(t),M.sin(t); return pt_xy(p.x*ct-p.y*st, p.x*st+p.y*ct)
  
  #change is_ccw????
  def is_ccw(self, a,b,c): return self.c_cmp(self.cross(b-a,c-a),0)
  def ang_abc(self, a,b,c):
    p,q,f=a-b,c-b,self.dot;return M.acos(f(p,q)/M.sqrt(self.dot(p,p)*f(q,q)))
  
  def pjt_pt_ln(self, a,b,c): p,f=b-a,self.dot; return a+p*f(c-a,p)/f(p,p)
  def pjt_pt_seg(self, a,b,c):
    p,f=b-a,self.dot; r=f(p,p)
    if self.c_cmp(r,0): return a
    r=f(c-a,p)/r
    return a if r<0 else b if r>1 else a+p*r
  
  def dst_pt_ln(self, a,b,c): return self.dst1(c, self.pjt_pt_ln(a,b,c))
  def dst_pt_seg(self, a,b,c): return self.dst1(c, self.pjt_pt_ln_seg(a,b,c))
  
  #following three might have bugs
  def is_ln_par(self, a,b,c,d): return (self.c_cmp(self.cross(b-a,c-a),0)==0)
  def is_ln_col(self, a,b,c,d): 
    f=self.is_ln_par; return (f(a,b,c,d) and f(b,a,a,c) and f(d,c,c,a))
  
  def is_seg_cross(self, a,b,c,d):
    f,g=self.dot,self.cross
    i,j,k,l,m,n=b-a,c-a,c-b,d-a,d-b,d-c
    if self.is_ln_col(a,b,c,d):
      for p in (a,b):
        for q in (c,d):
          if self.c_cmp(self.dst2(p,q),0)==0: return True
      return not(f(j,k)>0 and f(l,m)>0 and f(k,m)>0)
    return not((g(l,i)*g(j,i)>0) or (g(a-c,n)*g(b-c,n)>0))
    
  def is_ln_cross(self, a,b,c,d): 
    return (not self.is_ln_par(a,b,c,d) or self.is_ln_col(a,b,c,d))
    
  def pt_ln_cross(self, a,b,c,d): 
    p,q,f=b-a,c-d,self.cross; return a+p*f(c-a,q)/f(p,q)
  
  def pt_seg_cross(self, a,b,c,d):
    y,x,p,f=d.y-c.y,c.x-d.x,self.cross(d,c),M.fab
    u,v=f(y*a.x+x*a.y+p),f(y*b.x+x*b.y+p)
    return pt_xy((a.x*v+b.x*u)/(v+u),(a.y*v+b.y*u)/(v+u))
  
  def crl_has_pt(self, a,b,r): return (self.dst1(a,b)<r)
  
  def crl_ln_cross(self, a,b,c,r): # points with line cross circle
    p,q,f,g=b-a,a-c,self.dot,M.sqrt; A,B,C=(p,p),f(q,p),f(q,q)-r*r; D=B*B-A*C
    if -1==self.c_cmp(D,0): return None
    r=c+q+p*(-B+g(D+EPS))/A; return (r) if not(D>EPS) else (r,c+q+p*(-B-f(D))/A)
  
  def crl_crl_cross(self, a,b,r,R):
    d=self.dst1(a, b)
    if d>r+R or d+min(r,R)<max(r,R): return None
    x,v=(d*d-R*r+r*r)/(2*d),(b-a)/d; y=M.sqrt(r*r-x*x)
    i,j=a+v*x,self.rot_ccw90(v)*y; return (i+j) if y<EPS else (i+j,i-j)

  def crl_tan_pt(self, a,r,p):
    P,R=p-a,r**2; x=self.dot(P,P); D=x-R; C=self.c_cmp(D,0)
    if C==-1: return []
    if C==0: D=0
    A,B=P*(R/x),self.rot_ccw90(P*(-r*M.sqrt(D)/x)); return [a+A-B,a+A+B]
  
  def crl_crl_tan(self, a,b,r,R):
    f,g,h,V=self.crl_tan_pt,min,len,[]
    if 0==self.c_cmp(r,R):
      A=b-a; A=self.rot_ccw90(A*(r/M.sqrt(self.dot(A,A))));
      V=[(a+A,b+A),(a-A,b-A)]
    else:
      p=(a*-R+b*r)/(r-R); P,Q=f(a,r,p),f(b,R,p)
      V=[(P[i],Q[i]) for i in range(g(h(P),h(Q)))]
    p=(a*R+b*r)/(r+R); P,Q=f(a,r,p),f(b,R,p)
    for i in range(g(h(P),h(Q))): V.append((P[i],Q[i]))
    return V
  
  def tri_has_pt(self, a,b,c,p): #expand if too slow to statments
    f=self.is_ccw; return min((f(a,b,p),f(b,c,p),f(c,a,p)))>=0
  
  def tri_perm(self, a,b,c): return a+b+c
  def tri_area1(self, b,h): return b*h/2
  def tri_area2(self, a,b,c):
    s=self.tri_per(a,b,c)/2; return M.sqrt(s*(s-a)*(s-b)*(s-c))
  
  def tri_area3(self, a,b,c): f=self.cross; return (f(a,b)+f(b,c)+f(c,a))/2
  def tri_inc_help(self, a,b,c): 
    F=self;return F.tri_area2(a,b,c)/(F.tri_perm(a,b,c)/2)
  
  def tri_incir_rad(self, a,b,c):
    f=self.dst1; return self.tri_inc_help(f(a,b),f(b,c),f(c,a))
  
  def tri_cic_help(self, a,b,c): return a*b*c/(4*self.tri_area2(a,b,c))
  def tri_cic_rad(self, a,b,c):
    f=self.dst1; return self.tri_cic_help(f(a,b),f(b,c),f(c,a))
  
  def tri_cir_cen(self, a,b,c,d):
    F=self; f,g=F.cross,F.dot; p,q=b-a,d-c; p,q=pt_xy(p.y,-p.x),pt_xy(q.y,-q.x)
    A,B=f(p,q),f(q,p)
    if F.c_cmp(A,0)==0: return None
    r=pt_xy(g(a,p),g(c,q));return pt_xy((r.x*q.y-r.y*p.y)/A,(r.x*q.x-r.y*p.x)/B)

  #bisectors
  def tri_ang_bisct(self, a,b,c): #set to be vars if bugs
    f,g=M.sqrt,self.dst2; return (b-a)/f(g(b,a))*f(g(c,a))+(c-a)+a
  def tri_perp_bisct(self, a,b): return self.rot_ccw90(b-a)+(a+b)/2
  
  #triangle center for incircle circum and ortho
  def tri_incen(self, a,b,c):
    F=self; f=F.tri_ang_bisct; return F.tri_cir_cen(a,f(a,b,c),b,f(b,c,a))
  def tri_cic(self, a,b,c):
    F=self; f=F.tri_perp_bisct; 
    return F.tri_cir_cen((a+b)/2, f(a,b,c), (b+c)/2, f(b,c,a))
  def tri_ortho(self, a,b,c): return a+b+c-self.tri_cic(a,b,c)*2
  
  def poly_perm(self, P):
    return M.fsum((self.dst1(P[i],P[i+1]) for i in range(len(P)-1)))
  def poly_area1(self, P):
    return M.fsum((self.cross(P[i],P[i+1]) for i in range(len(P)-1)))/2
  def poly_area2(self, P): return abs(self.poly_area1(P))
  
  def poly_cnvx(self, P): #my sub for steven book
    f,e,s=self.is_ccw,len(P),1
    if e<4: return False
    t1=f(P[0],P[1],P[2])
    for i in range(s, e-2):
      if f(P[i],P[i+1],P[i+2])!=t1: return False
    return (f(P[-2],P[0],P[1])==t1)
  
  def poly_has_pt(self, P, p):#my submition for steven book 4
    if len(P)<4: return -1
    F=self; f,n,s=F.dst1,len(P),0.0
    for i in range(n-1): # on poly
      a,b=P[i],P[i+1]
      if abs(f(a,p)+f(p,b)-f(a,b))<EPS:return 0
    for i in range(n-1): # in poly
      A,B=P[i],P[i+1]; a=F.ang_abc(A,p,B); s+=a if F.is_ccw(p,A,B)<0 else -a
    return 1 if abs(s)>M.pi else -1
  
  def poly_centroid(self, P):
    F=self; A=pt_xy(0,0)
    for i in range(len(P)-1): 
      a,b=P[i],P[i+1];A=A+(a+b)*F.cross(a,b)
    return A/(6.0*F.poly_area1(P))
  
  def poly_cut(self, P,a,b):
    F=self;A,f=[],F.is_ccw; g=A.append
    for i in range(len(P)-1):
      p,q=P[i],P[i+1]; s,t=f(a,b,p),f(a,b,q)
      if 1==s: g(p)
      elif 0==s:
        g(p); continue
      if 1==s and -1==t: g(F.pt_seg_cross(p,q,a,b))
    if A and A[0]!=A[-1]: g(A[0])
    return A

  #https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
  def convex_hull1(self, P):
    g,p,H=len,sorted(set(P)),[]
    def f(B):
      for q in p:
        while g(H)>B and self.is_ccw(H[-2],H[-1],q)<0: H.pop()
        H.append(q)
      H.pop()
    if g(p)<=1: return p
    f(1); p=p[::-1]; f(g(H)+1); return H
  
  def poly_rot_caliper(self, P):
    F=self; f,g=F.cross,F.dst1; n,t,A=len(P)-1,0,0.0
    for i in range(n):
      a,b=P[i],P[(i+1)%n]; p=b-a
      while (t+1)%n!=i:
        if F.c_cmp(f(p, P[t+1]-a)-f(p, P[t]-a),0)<0: break
        t=(t+1)%n
      A=max(A,g(a,P[t])); A=max(A,g(b,P[t]));
    return A
  
  def cp_help(self, a,b):
    F=self; p,q=F.X[a],F.X[a+1]; A=(F.dst2(p,q),p,q)
    for i in range(a,b):
      for j in range(i+1,b):
        p,q=F.X[i],F.X[j]; d=F.dst2(p,q)
        if d<A[0]: A=(d,p,q)
    return A
  
  def cp_dq(self, a,b,Y):
    F=self; n,h=b-a,F.cp_dq
    if n<4: return F.cp_help(a,b)
    L,R,l,r=a+n-n//2, a+n//2,[],[]; m=round((F.X[L].x+F.X[R].x)/2)
    for p in Y:
      r.append(p) if m<p.x else l.append(p)
    dl=h(a,L,l)
    if dl[0]==0: return dl
    dr=h(L,b,r)
    if dr[0]==0: return dr
    d=dl if dl[0]<dr[0] else dr; A=[p for p in Y if d[0]>(p.x-m)**2]
    for i in range(len(A)):
      for j in range(i+1, len(A)):
        I,J=A[i],A[j]
        if (I.y-J.y)**2>=d[0]: break
        D=F.dst2(I,J)
        if D<d[0]:d=(D,I,J)
    return d
  
  def cp_solve(self, P):
    self.X=sorted(P, key=lambda pt_xy: pt_xy.x)
    Y=sorted(P, key=lambda pt_xy: pt_xy.y)
    return self.cp_dq(0, len(P), Y)
  
  def set_por(self, p): self.por=p
  def ang_cmp(self, a,b): 
    A=self.is_ccw(a[0],self.por,b[0])
    return A if A!=0 else self.dst1(self.por,a[0])-self.dst1(self.por,b[0])

def my_cmp(f):
  class K:
    def __init__(self, o): self.o=o
    def __lt__(self, b): return f(self.o, b.o)<0
  return K
    
class GEO_ALGOS_3D:
  def __init__(self): pass
  def dot3d(self, a,b): return a.x*b.x+a.y*b.y+a.z*b.z
  def cross3d(self, a,b): 
    return pt_xyz(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x)
  def vol_3d(self, a,b,c,d): F=self; return F.dot3d(d-a, F.cross3d(b-a,c-a))/6
  def area_3d(self, a,b,c): F=self; p=F.cross(b-a,c-a); return F.dst3d(p,p)/2
  def dst3d(self, a,b): return M.sqrt(self.dot3d(a-b,a-b))
  
  def pt_dst_pln_3d(self, p,a,b,c,d): 
    return abs(p.x*a+p.y*b+p.z*c+d)/M.sqrt(a*a+b*b+c*c)
  def pln_dst_pln_par_3d(self, a,b,c,d,D): return abs(d-D)/M.sqrt(a*a+b*b+c*c)
  
  def pjt_pt_ln_3d(self, a,b,c,t): #line 0, seg 1, ray 2, 1st pt is ray endpoint
    F=self;A=F.dst3d(b,c);
    if A==0: return b
    else:
      u=((a.x-b.x)*(c.x-b.x)+(a.y-b.y)*(c.y-b.y)+(a.z-b.z)*(c.z-b.z))/A
      P=(c-b)*u+b
      if t>0 and u<0: P=b
      if t==1 and u>1: P=c
      return P
  
  def pt_dst_ln_3d(self, a,b,c,t):F=self;return F.dst3d(a, F.pjt_pt_ln(a,b,c,t)) 
