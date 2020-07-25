
import math
from sys import stdin as rf

#euclidean_distance (floats)tested on cursethedarkness, sibice, imperfectgps
#dist_pt_seg tested on goatrope
#segments_intersect tested on countingtriangles
#convex_hull_monotone_chain tested on convexhull2, convexhull, dartscoring, robotprotection
#polygon_area tested on convexpolygonarea
#polygon_signed_area tested on polygonarea
#polygon_where_pt tested on pointinpolygon for in on out of polygon
        

EPS=1e-12
NUM_SIG=9
#todo make sure we know the math behind the functions it loosk like 
# a lot of matrix operation stuffs 
#uncomment the comment the commented lines for floating point cords
class pt_xy:
    def __init__(self, new_x, new_y):self.x,self.y=map(int,[new_x,new_y])
    #def __init__(self, new_x=float(0), new_y=float(0)): self.x,self.y=round(new_x, NUM_SIG), round(new_y, NUM_SIG)
    
    def set_pt_xy(self, n_pt_xy): self.x,self.y=n_pt_xy.x,n_pt_xy.y
    def display(self): print(self.x,self.y)
        
    def __add__(self, b): return pt_xy(self.x+b.x, self.y+b.y)
    def __sub__(self, b): return pt_xy(self.x-b.x, self.y-b.y) 
    def __mul__(self, c): return pt_xy(self.x*c, self.y*c)
    def __truediv__(self, c): return pt_xy(self.x/c, self.y/c) #does this even make sense in a int based point
    def __floordiv__(self, c): return pt_xy(self.x//c, self.y//c)
    

    def __lt__(self, b): return (self.x<b.x) if self.x!=b.x else (self.y<b.y)
    def __eq__(self, b): return (self.x==b.x) and (self.y==b.y)
    #def __lt__(self, b): return (self.x<b.x) if math.fabs(self.x-b.x)>EPS else (self.y<b.y)
    #def __eq__(self, b): return (abs(self.x-b.x)<EPS) and (abs(self.y-b.y)<EPS)
    
    def __str__(self): return "{} {}".format(self.x, self.y)

    
    def __str__(self): return "(x={},y={})".format(self.x, self.y)
    
    def __round__(self, n): return pt_xy(round(self.x,n),round(self.y,n))
    def __hash__(self):return hash((self.x,self.y))

class DLL:
    def __init__(self, v):
        self.LIS={}
        self.head=None
        self.Nam=v
    
    def __len__(self): return len(self.LIS)
    
    def rem_el(self, el):
        if el not in self.LIS:
            print("error {} not in {} when deleting".format(el, self.Nam))
            return
        p,s=self.LIS[el] #pred, succ
        if el==self.head: self.head=s
        del self.LIS[el]
        leno=len(self.LIS)
        if leno==0: return
        elif leno==1: self.LIS[p]=(None, None)
        elif leno==2: self.LIS[p], self.LIS[s]=(s, s), (p, p)
        else:
            self.LIS[p]=(self.LIS[p][0], s)
            self.LIS[s]=(self.LIS[s][0], p)
            
    def add_first(self, a):
        self.head=a
        self.LIS[a]=(None, None)
    
    #add a after b
    def add_after(self, a, b):
        leno=len(self.LIS)
        if leno==0: self.add_first(b); return; #or print an error in our algo
        elif leno==1: self.LIS[a], self.LIS[b]=(b, b), (a, a)
        else:
            c=self.LIS[b][1]
            bb,cc=self.LIS[b], self.LIS[c]
            self.LIS[a]=(b,c)
            self.LIS[b], self.LIS[c]=(bb[0], a), (a, cc[1])
    
    #add a before b
    def add_after(self, a, b):
        leno=len(self.LIS)
        if b==self.head: self.head=a
        if leno==0: self.add_first(b); return; #or print an error in our algo
        elif leno==1: self.LIS[a], self.LIS[b]=(b, b), (a, a)
        else:
            c=self.LIS[b][0]
            bb,cc=self.LIS[b], self.LIS[c]
            self.LIS[a]=(c,b)
            self.LIS[b], self.LIS[c]=(a, bb[1]), (cc[0], a)

class ADJDLL:
    def __init__(self, n):
        self.adj={i: DLL(i) for i in range(n)}
        
    def PRED(self, a, b): return self.adj[a][b][0]
    def SUCC(self, a, b): return self.adj[a][b][1]
    
    def INSERT(self, L, R, LL, RR):#left is before right is after 
        self.adj[L].add_after(R, RR)
        self.adj[R].add_before(L, LL)
    
    def DELETE(self, L, R):
        self.adj[L].rem_el(R)
        self.adj[R].rem_el(L)


class GEO_ALGOS:
    def __init__(self):
        pass
    
    def epscmp(self, x):
        return -1 if x<-EPS else 1 if x>EPS else 0
    
    def c_cmp(self, a, b): return 0 if math.isclose(a, b) else -1 if a<b else 1
    
    def euclidean_distance(self, a, b): return math.hypot((a-b).x, (a-b).y)
    
    #TODO look up and see how well the z cord translates into these functions
    def dot_product(self, a, b): return a.x*b.x+a.y*b.y #move to point class
    def cross_product(self, a, b): return a.x*b.y-a.y*b.x
    
    def dist2(self, a, b): return self.dot_product(a-b,a-b)
    
    def rotate_ccw90(self, p): return pt_xy(-p.y, p.x)
    def rotate_cw90(self, p):  return pt_xy( p.y,-p.x)
    def rotate_ccw(self, p, t): 
        return pt_xy(p.x*math.cos(t)-p.y*math.sin(t), p.x*math.sin(t)+p.y*math.cos(t))
    
    def point_rotation_wrt_line(self, a, b, c):
        return self.c_cmp(self.cross_product(b-a,c-a),0)
    
    def angle_abc(self, a, b, c): #MARKED
        ab,cb=a-b,c-b
        return math.acos(self.dot_product(ab,cb)/math.sqrt(self.dot_product(ab,ab)*self.dot_product(cb,cb)))
    
    def project_pt_line(self, a, b, c): 
        return a+(b-a)*self.dot_product(c-a,b-a)/self.dot_product(b-a, b-a)
    
    def project_pt_line_segment(self, a, b, c):
        r=self.dot_product(b-a,b-a)
        if abs(r)<EPS: return a
        r=self.dot_product(c-a,b-a)/r
        if r<0: return a
        elif r>1: return b
        else: return a+(b-a)*r
    
    def dist_pt_segment(self, a, b, c):
        return self.euclidean_distance(c, self.project_pt_line_segment(a,b,c))
        
    def dist_pt_line(self, a, b, c): #cross product version to fomula sheeet
        return self.euclidean_distance(c, self.project_pt_line(a,b,c))
    
    def is_lines_parallel(self, a, b, c, d):
        return (self.epscmp(self.cross_product(b-a, c-d))==0)
        #return (abs(self.cross_product(b-a, c-d))<EPS)
        
    def is_lines_collinear(self, a, b, c, d): #copy paste code if its too slow this is safety from typing errors
        return (self.is_lines_parallel(a,b,c,d) 
        and self.is_lines_parallel(b,a,a,c) 
        and self.is_lines_parallel(d,c,c,a))
    
    #return if collinear to get a unique intersection or check outside the funtion before hand
    def is_segments_intersect(self, a, b, c, d):
        if self.is_lines_collinear(a, b, c, d):
            if self.dist2(a, c)<EPS or self.dist2(a, d)<EPS or self.dist2(b, c)<EPS or self.dist2(b, d)<EPS:
                return True
            return not(self.dot_product(c-a, c-b)>0 and self.dot_product(d-a, d-b)>0 and self.dot_product(c-b, d-b)>0)
        return not((self.cross_product(d-a,b-a)*self.cross_product(c-a,b-a)>0)
        or (self.cross_product(a-c,d-c)*self.cross_product(b-c,d-c)>0))
    
    #move not outside the expr to get unique checker
    def is_lines_intersect(self, a, b, c, d):
        return (not self.is_lines_parallel(a,b,c,d) or self.is_lines_collinear(a,b,c,d))
    
    #this assumes you checked if unique intersections
    #doubles as segment intersections if youve check that they actually intersect
    def pt_lines_intersect(self, a, b, c, d):
        b,c,d=b-a,c-a,c-d
        return a+b*self.cross_product(c,d)/self.cross_product(b,d)
    
    def pt_line_segment_intersect(self, a, b, c, d):
        y, x, cp=d.y-c.y, c.x-d.x, self.cross_product(d,c)
        u=math.fab(y*a.x+x*a.y+cp)
        v=math.fab(y*b.x+x*b.y+cp)
        return pt_xy((a.x*v+b.x*u)/(v+u),(a.y*v+b.y*u)/(v+u))
    
    #add the case for lines collinear
    def circle_center_given_abc(self, a, b, c):
        b,c=(a+b)/2,(a+c)/2
        return self.pt_lines_intersect(b, b+self.rotate_cw90(a-b), c, c+self.rotate_cw90(a-c))
    
    #mod for integer return later 
    def circle_has_point(self, a, b, r): return (self.euclidean_distance(a, b)<r)
    
    def circle_line_intersection(self, a, b, c, r):
        b,a=b-a,a-c
        A,B,C=self.dot_product(b,b), self.dot_product(a,b), self.dot_product(a,a)-r*r
        D=B*B-A*C
        if D<-EPS: return None
        rVal=c+a+b*(-B+math.sqrt(D+EPS))/A
        return (rVal) if not(D>EPS) else (rVal, c+a+b*(-B-math.sqrt(D))/A)
        
    def circle_circle_intersection(self, a, b, r1, r2):
        d=self.euclidean_distance(a, b)
        if d>r1+r2 or d+min(r1, r2)<max(r1,r2): return None
        x=(d*d-r2*r2+r1*r1)/(2*d)
        y=math.sqrt(r1*r1-x*x)
        v=(b-a)/d
        i,j=a+v*x,self.rotate_ccw90(v)*y
        return (i+j) if not(y>0) else (i+j, i-j)

    def circle_tangent_point(self, a, r, p):
        pa=p-a; x=self.dot_product(pa, pa)
        dst=x-r**2; res=self.c_cmp(dst, 0)
        if res==-1: return []
        if res==0: dst=0
        q1,q2=pa*(r**2/x),self.rotate_ccw90(pa*(-r*math.sqrt(dst)/x))
        return [a+q1-q2,a+q1+q2]
    
    def circle_circle_tangents(self, c1, r1, c2, r2):
        rTans=[]
        if self.c_cmp(r1,r2)==0:
            t=c2-c1
            t=self.rotate_ccw90(t*(r1/math.sqrt(self.dot_product(t,t))))
            rTans=[(c1+t,c2+t),(c1-t,c2-t)]
        else:
            p=(c1*-r2+c2*r1)/(r1-r2)
            ps,qs=self.circle_tangent_point(c1,r1,p),self.circle_tangent_point(c2,r2,p)
            rTans=[(ps[i],qs[i]) for i in range(min(len(ps),len(qs)))]
        p=(c1*r2+c2*r1)/(r1+r2)
        ps,qs=self.circle_tangent_point(c1,r1,p),self.circle_tangent_point(c2,r2,p)
        for i in range(min(len(ps),len(qs))):
            rTans.append((ps[i], qs[i]))
        return rTans


    def triangle_has_pt(self, a, b, c, p):
        t1=(self.point_rotation_wrt_line(a,b,p)>=0)
        t2=(self.point_rotation_wrt_line(b,c,p)>=0)
        t3=(self.point_rotation_wrt_line(c,a,p)>=0)
        return t1 and t2 and t3

    
    def triangle_area_2bh(self, b, h):
        return b*h/2
    
    def triangle_area_heron(self, ab, bc, ca):
        s=self.triangle_perimeter(ab, bc, ca)*0.5
        return math.sqrt(s*(s-ab)*(s-bc)*(s-ca))
        
    def triangle_perimeter(self, ab, bc, ca):
        return ab+bc+ca
    
    def triangle_area_verts(self, a, b, c):
        return (self.cross_product(a,b)+self.cross_product(b,c)+self.cross_product(c,a))/2
        
    def incircle_helper(self, ab, bc, ca):
        return self.triangle_area_heron(ab, bc, ca)/(self.triangle_perimeter(ab, bc, ca)*0.5)
    
    def triangle_incircle_radius(self, a, b, c):
        ab=self.euclidean_distance(a,b)
        bc=self.euclidean_distance(b,c)
        ca=self.euclidean_distance(c,a)
        return self.incircle_helper(ab,bc,ca)
    
    def circumcircle_helper(self, ab, bc, ca):
        return ab*bc*ca/(4*self.triangle_area_heron(ab,bc,ca))
        
    def triangle_circumcircle_radius(self, a, b, c):
        ab=self.euclidean_distance(a,b)
        bc=self.euclidean_distance(b,c)
        ca=self.euclidean_distance(c,a)
        return self.circumcircle_helper(ab,bc,ca)
    
    # def triangle_incircle(self, a, b, c):
    #     r=self.triangle_incircle_radius(a,b,c)
    #     if abs(r)<EPS: return (False,0,0)
    #     ratio=self.euclidean_distance(a,b)/self.euclidean_distance(a,c)
    #     p1=b+(c-b)*(ratio/(1.0+ratio))
        
    #     ratio=self.euclidean_distance(a,b)/self.euclidean_distance(b,c)
    #     p2=a+(c-a)*(ratio/(1.0+ratio))
    #     if self.is_lines_intersect(a,p1,b,p2): return (True, r, round(self.pt_lines_intersect(a,p1,b,p2),12))
    #     else: return  (False,0,0)
    
    #center of triangle circle stuff 
    
    def triangle_circle_center(self, a, b, c, d):
        p1,p2=b-a,d-c
        p1=pt_xy(p1.y,-p1.x)
        p2=pt_xy(p2.y,-p2.x)
        c12,c21=self.cross_product(p1,p2),self.cross_product(p2,p1)
        if self.epscmp(c12)==0: return None # doesnt exist? colined <=> a1*c2-a2*c1=0 && b1*c2-b2*c1=0
        p3=pt_xy(self.dot_product(a,p1), self.dot_product(c,p2))
        return pt_xy((p3.x*p2.y-p3.y*p1.y)/c12,(p3.x*p2.x-p3.y*p1.x)/c21)
        
    
    def triangle_angle_bisector(self, a, b, c):
        val=(b-a)/math.sqrt(self.dist2(b,a))*math.sqrt(self.dist2(c,a))
        return val+(c-a)+a
    
    def triangle_perpendicular_bisector(self, a, b):
        v=b-a
        v.x,v.y=-v.y,v.x
        return v+(a+b)/2
        
    def triangle_incenter(self, a, b, c):
        v1=self.triangle_angle_bisector(a,b,c)
        v2=self.triangle_angle_bisector(b,c,a)
        return self.triangle_circle_center(a, v1, b, v2)
    
    def triangle_circumcenter(self, a, b, c):
        v1=self.triangle_perpendicular_bisector(a,b)
        v2=self.triangle_perpendicular_bisector(b,c)
        return self.triangle_circle_center((a+b)/2, v1, (b+c)/2, v2)
    
    def triangle_orthocenter(self, a, b, c):
        return a+b+c-self.triangle_circumcenter(a,b,c)*2
    
    #note these are based on counter clowck wise ordering 
    #check for improvements on integers
    def polygon_perimeter(self, ps):#MARKED
        return math.fsum([self.euclidean_distance(ps[i], ps[i+1]) for i in range(len(ps)-1)])
    
    def polygon_signed_area(self, ps): #MARKED
        return math.fsum([self.cross_product(ps[i], ps[i+1]) for i in range(len(ps)-1)])/2
    
    def polygon_area(self, ps): #MARKED
        return abs(self.polygon_signed_area(ps))
    
    #notes for this function (subject to change)
    #test it obviously 
    #isLeft is for counter clockwise swapping rot=-1 goes for clock wise ordering
    #not modified for colinear points yet
    def polygon_convex(self, ps): #MARKED
        lim,rot=len(ps),1
        if lim<4: return False
        isLeft=(1==self.point_rotation_wrt_line(ps[0],ps[1],ps[2]))
        for i in range(rot, lim-1):
            if isLeft!=(rot==self.point_rotation_wrt_line(ps[i],ps[i+1],ps[i+1 if i+2<lim else 1])):
                return False
        return True
    
    # #doesnt work when point is on boundry and also dont understand it will look at it later
    # #maybe change to not fsum but i think fsum good idea 
    # def polygon_contains_pt(self, ps, p): #MARKED
    #     if len(ps)<4: return 0
    #     tmp=[self.angle_abc(ps[i],p,ps[i+1]) for i in range(len(ps)-1)]
    #     soa=math.fsum([el if 1==self.point_rotation_wrt_line(ps[i],p,ps[i+1]) else -el for i,el in enumerate(tmp)])
    #     return (0==self.epscmp(math.fabs(soa)-round(math.pi,13)*2.0))
    
    def polygon_contains_pt(self, ps, p):
        ans=False
        for i in range(len(ps)-1):
            if (ps[i].y<=p.y and p.y<ps[i+1].y or ps[i+1].y<=p.y and p.y<ps[i].y) and \
            (p.x<ps[i].x + (ps[i+1].x-ps[i].x)*(p.y-ps[i].y)/(ps[i+1].y-ps[i].y)):
                ans=not ans
        return ans
        
    def polygon_has_pt(self, ps, p):
        for i in range(len(ps)-1):
            if self.epscmp(self.dist_pt_segment(ps[i],ps[i+1],p))==0:
                return True
        return False   
    
    def polygon_where_pt(self, ps, p):
        a,b=self.polygon_contains_pt(ps, p), self.polygon_has_pt(ps, p)
        return 0 if b else 1 if a else -1

    
    def polygon_centroid(self, ps):
        ans=pt_xy(0,0)
        for i in range(len(ps)-1):
            ans=ans+(ps[i]+ps[i+1])*self.cross_product(ps[i],ps[i+1])
        return ans/(6.0*self.polygon_signed_area(ps))
    
    #look up do see if we can do better than n^2
    def polygon_is_simple(self, ps):
        for i in range(len(ps)-1):
            for k in range(i+1,len(ps)-1):
                j,l=(i+1)%len(ps),(k+1)%len(ps)
                if i==l or j==k: continue
                if self.is_segments_intersect(ps[i],ps[j],ps[k],ps[l]):
                    return False
        return True
    
    def polygon_cut(self, ps, a, b):
        ans=[]
        for i in range(len(ps)-1):
            r1,r2=self.point_rotation_wrt_line(a,b,ps[i]), self.point_rotation_wrt_line(a,b,ps[i+1])
            if 1==r1: ans.append(ps[i])
            elif 0==r1: 
                ans.append(ps[i])
                continue
            if 1==r1 and -1==r2:
                ans.append(self.pt_line_segment_intersect(ps[i], ps[i+1], a, b))
        if ans and ans[0]!=ans[-1]:
            ans.append(ans[0])
        return ans
    
    #https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    def convex_hull_monotone_chain(self, ps):
        def f(pts, r, lim):
            for p in pts:
                while len(r)>lim and self.cross_product(r[-1]-r[-2], p-r[-2])<0:r.pop()
                r.append(p)
            r.pop()
        ans,r=sorted(set(ps)),[]
        if len(ans)<=1: return ans
        f(ans, r, 1);f(ans[::-1], r, len(r)+1);return r
    
    def polygon_rotating_caliper(self, ps):
        n,t,ans=len(ps)-1,0,0.0
        for i in range(n):
            pi=ps[i]
            pii=ps[i+1] if i+1<=n else ps[0]
            p=pii-pi
            while (t+1)%n!=i:
                if self.c_cmp(self.cross_product(p, ps[t+1]-pi)-self.cross_product(p, ps[t]-pi),0)<0:
                    break
                t=(t+1)%n
            #ans=max(ans, self.triangle_area_verts(pi, pii, ps[t]))
            ans=max(ans, self.euclidean_distance(pi, ps[t]))
            ans=max(ans, self.euclidean_distance(pii, ps[t]))
        return ans
    
    def bf_helper(self, x1, x2):
        ans=(self.dist2(self.X[x1], self.X[x1+1]), self.X[x1], self.X[x1+1])
        for i in range(x1, x2):
            for j in range(i+1, x2):
                dt=self.dist2(self.X[i], self.X[j])
                if dt<ans[0]:
                    ans=(dt, self.X[i], self.X[j])
        return ans

    def closest_pair(self, x1, x2, Y):
        n=x2-x1
        if n<=3: return self.bf_helper(x1, x2)
        l_siz, r_siz=x1+n-n//2, x1+n//2
        mid=round((self.X[l_siz].x+self.X[r_siz].x)/2)
        YL, YR=[],[]
        rapp,lapp=YR.append,YL.append
        for py in Y:
            if mid<py.x: rapp(py)
            else:        lapp(py)
        dl=self.closest_pair(x1, l_siz, YL)
        if dl[0]==0: return dl
        dr=self.closest_pair(l_siz, x2, YR)
        if dr[0]==0: return dr
        da=dl if dl[0]<dr[0] else dr
        YP=[p for p in Y if da[0]>(p.x-mid)**2]
        for i in range(len(YP)):
            for j in range(i+1, len(YP)):
                if (YP[i].y-YP[j].y)**2>=da[0]: break
                dt=self.dist2(YP[i],YP[j])
                if dt<da[0]:
                    da=(dt,YP[i], YP[j])
        return da

    def compute_closest_pair(self, pts):
        self.X=sorted(pts, key=lambda pt_xy: pt_xy.x)
        Y=sorted(pts, key=lambda pt_xy: pt_xy.y)
        return self.closest_pair(0,len(pts), Y)
    
    #might be a way to merge the two lower ones
    def dt_find_uct(self, lb, le, rb, re):
        X,Y=le,rb
        ZT=self.CHL[self.CHLM[X]+1]
        ZR, ZL=self.CHR[self.CHRM[Y]+1], self.DT.PRED(X, ZT)
        while True: #check online later to see if it needs to be like this or if it needs to be something else 
            if self.point_rotation_wrt_line(self.V[X], self.V[Y], self.V[ZR])<0:
                Y,ZR=ZR,self.DT.SUCC(ZR, Y)
            else:
                if self.point_rotation_wrt_line(self.V[X], self.V[Y], self.V[ZL])<0:
                    X,ZL=ZL,self.DT.PRED(ZL, X)
                else:
                    return (X,Y)
        return None #error
    
    def dt_find_lct(self, lb, le, rb, re):
        X,Y=le,rb
        ZT=self.CHL[self.CHLM[X]+1]
        ZR, ZL=self.CHR[self.CHRM[Y]+1], self.DT.PRED(X, ZT)
        while True: #check online later to see if it needs to be like this or if it needs to be something else 
            if self.point_rotation_wrt_line(self.V[X], self.V[Y], self.V[ZR])<0:
                Y,ZR=ZR,self.DT.SUCC(ZR, Y)
            else:
                if self.point_rotation_wrt_line(self.V[X], self.V[Y], self.V[ZL])<0:
                    X,ZL=ZL,self.DT.PRED(ZL, X)
                else:
                    return (X,Y)
        return None #error
    
    def dt_divide(self, l, r):
        n=r-l
        if n<4: 
            #base case a triangle or less
            ch=self.convex_hull_monotone_chain(self.V[l:r])
            #add the triangulation start here
            return ch
        nl,nr=l+n-n//2, l+n//2
        #get the left and right convex hulls recursively maybe change that storage method
        #so its not in a object var idk just want to get it done
        self.CHL=self.dt_divide(l, nl) 
        self.CHR=self.dt_divide(nl, r)
        self.CHLM={el:i for el,i in enumerate(self.CHL)}
        self.CHRM={el:i for el,i in enumerate(self.CHR)}
        BT=self.dt_find_lct(l, nl-1, nl, r)
        UT=self.dt_find_uct(l, nl-1, nl, r)
        L,R=BT[0], BT[1]
        LP,RP=self.CHL[self.CHLM[L]+1], self.CHR[self.CHRM[R]+1]
        while BT!=UT: # make this look correct
            a=b=False
            self.DT.INSERT(L,R,LP,RP)
            R1=self.DT.PRED(R,L)
            #if r1 left of line from L to R
            if self.point_rotation_wrt_line(self.V[L], self.V[R], self.V[R1])>0:
                R2=self.DT.PRED(R,R1)
                while True: #put test as qtest
                    self.DT.DELETE(R,R1)
                    R1=R2
                    R2=self.DT.PRED(R,R1)
            else: a=True
            L1=self.DT.SUCC(L,R)
            #if l1 left of line from R to L
            if self.point_rotation_wrt_line(self.V[R], self.V[L], self.V[L1])<0:
                L2=self.DT.SUCC(L,L1)
                while True: #put cond as qtest
                    self.DT.DELETE(L, L1)
                    L1=L2
                    L2=self.DT.SUCC(L,L1)
            else: b=True
            if a: LP,L=L,L1
            else:
                if b: RP,R=R,R1
                else:
                    if True: #put qtest here
                        RP,R=R,R1
                    else:
                        LP,L=L,L1
        return self.CHL #merge them into this ?
        
        
        
    
    def compute_delaunay_triangles(self, ps):
        self.DT=ADJDLL(len(ps))
        self.V=sorted(ps)
        ans=self.dt_divide(0, len(ps))
        #triangulation in the DT 

def main():
    obj=GEO_ALGOS()
    tri=[None]*3
    for i in range(3):
        x,y=map(int, rf.readline().split())
        tri[i]=pt_xy(x,y)
    tri.append(tri[0])
    n=int(rf.readline().strip())
    amt=0
    for i in range(n):
        x,y=map(int, rf.readline().split())
        if 0<=obj.polygon_where_pt(tri, pt_xy(x,y)):
            amt+=1
    print("{:.1f}\n{}".format(obj.triangle_area_verts(tri[0],tri[1],tri[2]),amt))
main()
