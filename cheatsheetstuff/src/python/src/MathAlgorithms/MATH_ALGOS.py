#is_prime tested on primes2
#lcm/gcd tested on temperatureconfusion, primes2, dasblinkenlights, jackpot, prsteni, smallestmultiple, jughard
#extendedEuclid tested on candydistribution, modulararithmetic
#
#fib_logn tested on porpoises, anti11
#sieve_primes tested on
#is_prime_mrpt tested on primereduction, primes2
#catalan_n tested on catalan, catalansquare
#catalan_n_mod_p tested on fiat
#binomialCoefficient tested on lockedtreasure, election
#binomialCoefficientdp election
#chinese_remainder_theorem tested on chineseremainder, heliocentric, generalchineseremainder

#note all things marked "#test this its from stanford icpc 2013-14"
#are taken from that cheat sheet and translated to python
#taken because the system is small in code will test and see if it makes the cut
#for our own cheat sheet
class MATH_ALGOS:
    def __init__(self):
        self.n=None
    
    def set_N(self, val):
        self.N=val
    
    #prime related section
    def is_prime_triv(self, n):
        if n<=3:
            return n>1
        elif n%2==0 or n%3==0:
            return False
        p=5
        while p*p<=n:
            if n%p==0 or n%(p+2)==0:
                return False
            p+=6
        return True
    
    def sieve_primes(self, n):
        pb=[True]*n
        self.ps=[2]
        for i in range(4, n, 2):
            pb[i]=False
        for i in range(3, n, 2):
            if pb[i]:
                self.ps.append(i)
                for j in range(i*i, n, 2*i):
                    pb[j]=False
    
    def gen_set_primes(self):
        self.pss=set(self.ps)
    
    #mods at the bottom
    def pFactorize(self, n):
        facts=[]
        for p in self.ps:
            if p*p>n:
                break
            if n%p==0:
                while n%p==0:
                    n//=p
                    facts.append(p)
        if n>1:
            facts.append(n)
        return facts
    
    def is_comp(self, a, d, n, s):
        if pow(a, d, n)==1:
            return False
        for i in range(s):
            if pow(a, 2**i*d, n)==n-1:
                return False
        return True 
    
    def is_prime_mrpt(self, n, pfhn=16):
        if n in self.pss:
            return True
        if any((n%self.ps[p]==0) for p in range(50)) or n<2 or n==3215031751:
            return False
        d,s=n-1,0
        while not (1&d):
            d,s=d//2,s+1
        for i,bound in enumerate(self.inds,2):
            if n<bound:
                return not any(self.is_comp(self.mrps[j], d, n, s) for j in range(i))
        return not any(self.is_comp(self.ps[j], d, n, s) for j in range(pfhn))
    
    def prep_mrpt(self):
        self.sieve_primes(1000) #comment out if different size needed
        self.gen_set_primes() #comment out if already have bigger size
        self.inds=[1373653, 25326001, 118670087467, 2152302898747, 3474749660383, 341550071728321]
        self.mrps=[2, 3, 5, 7, 11, 13, 17]
    
    #modular section
    def power_mod(self, b, e, m):
        x=1
        while e>0:
            b, e, x=b*b%m, e//2, b*x%m if 1&e else x
        return x
    
    def gcd(self, u, v):
        while v:
            u,v=v,u%v
        return abs(u)

    def lcm(self, u, v):
        return u*v//self.gcd(u,v)
    
    #test this its from stanford icpc 2013-14
    #givs all solutions ax=b(mod n)
    def mod(self, a, b):
        return ((a%b)+b)%b #remove this later??
    
    def extendedEuclid(self, a, b):
        if b==0:
            return 1,0,a
        x,y,d=self.extendedEuclid(b, a%b)
        return y, x-y*(a//b), d
    
    #test this its from stanford icpc 2013-14
    #givs all solutions ax=b(mod n)
    def modular_linear_equation_solver(self, a, b, n):
        x,y,d=self.extendedEuclid(a, n)
        if b%d==0:
            x=self.mod(x*(b//d), n)
            return [self.mod(x+i*(n//d), n) for i in range(d)]
        return []
    
    #test this its from stanford icpc 2013-14
    #givs b in ab = 1 (mod n)
    def mod_inverse(self, a, n):
        x,y,d=self.extendedEuclid(a, n)
        return -1 if d>1 else self.mod(x, n)
    
    #test this its from stanford icpc 2013-14
    #tested on same as chinese_remainder_theorem
    def crt_helper(self, x, a, y, b):
        s,t,d=self.extendedEuclid(x, y)
        return (0,-1) if a%d != b%d else (self.mod(s*b*x+t*a*y,x*y)//d,x*y//d)
    
    #test this its from stanford icpc 2013-14
    #tested 
    def chinese_remainder_theorem(self, x, a):
        ans=(a[0], x[0])
        for i in range(1, len(x)):
            ans=self.crt_helper(ans[1], ans[0], x[i], a[i])
            if ans[1]==-1:
                break
        return ans
    
    #test this its from stanford icpc 2013-14
    #computes x and y in ax+by=c failure x=y=-1
    def linear_diophantine(self, a, b, c):
        d=self.gcd(a,b)
        if c%d==0:
            x=c//d * self.mod_inverse(a//b, b//d)
            return (x,(c-a*x//b))
        return (-1,-1)
    
    #combinatorics and sequences
    def fib_iter(self, n):
        self.fibs=[0]*(n+1)
        self.fibs[1]=1
        for i in range(2, n+1):
            self.fibs[i]=self.fibs[i-1]+self.fibs[i-2]
        return self.fibs[n]
    
    def fib_logn_rec(self, n):
        if n==0:
            return 0
        if n==1 or n==2:
            self.fibs[n]=1
            return 1
        if n in self.fibs:
            return self.fibs[n]
        
        if n&1:
            k=(n+1)//2
            fk1=self.fib_logn_rec(k)
            fk2=self.fib_logn_rec(k-1)
            self.fibs[n]=fk1*fk1+fk2*fk2
        else:
            k=n//2
            fk1=self.fib_logn_rec(k)
            fk2=self.fib_logn_rec(k-1)
            self.fibs[n]=(2*fk2+fk1)*fk1
        return self.fibs[n]
        
    def fib_logn(self, n):
        self.fibs={}
        return self.fib_logn_rec(n)
    
    #mod with an array for repeated use    
    def catalan_n(self, n):
        self.cat=[0]*(n+1)
        nxt=0
        self.cat[nxt]=1
        for i in range(n):
            self.cat[i+1]=self.cat[i]*(4*i+2)//(i+2)
    
    def catalan_n_mod_p(self, n, p):
        from collections import Counter
        self.sieve_primes(int((5*n)**0.5))
        TPF={}
        BPF={}
        for i in range(n):
            a=self.pFactorize(4*i+2)
            pfs=Counter(a)
            for k,v in pfs.items():
                if k not in TPF:
                    TPF[k]=v
                else:
                    TPF[k]+=v
            a=self.pFactorize(i+2)
            pfs=Counter(a)
            for k,v in pfs.items():
                if k not in BPF:
                    BPF[k]=v
                else:
                    BPF[k]+=v
        for k,v in BPF.items():
            TPF[k]-=v
        ans=1
        for k,v in TPF.items():
            if v>0:
                ans*=pow(k,v,p)
        return ans%p

    #O(k) versuon to get (n,k) space O(1)
    def binomialCoefficient(self,n,k):
        k=min(k, n-k)
        res=1
        for i in range(k):
            res*=(n-i)
            res//=(i+1)
        return res
    
    #O(n^2) i think its n^2 gets memozing 
    def binomialCoefficientdp(self, n, k):
        if n==k or k==0:
            return 1
        if (n,k) not in self.binomial:
            self.binomial[(n,k)]=self.binomialCoefficientdp(n-1,k)+self.binomialCoefficientdp(n-1,k-1)
        return self.binomial[(n,k)]
    
    def binomialCoefficientPrep(self):
        self.binomial={}

    #matrix stuff
        
        
#mods for other functions: deal with three or 4 lines: facts(l1), append both inside and lst if(l2)
#last line of while loor(l3), line after while loop(l4) 
#only modded lines will show up
#numPF(self, n): return ans
#l1:ans=0
#l2:empty
#l3:ans+=1
#sumPF(self, n): return ans
#l1:ans=0
#l2:empty
#l3:ans+=p
#numDiffPF(self, n): return ans
#l1:ans=0
#l2:ans+=1
#numDivs(self, n): return ans
#l1: ans=1
#l2: power=0
#l3: power+=1
#l4: ans+=(power+1)
#sumDivs(self, n): return ans
#l1: ans=1
#l2: power=0
#l3: power+=1
#l4: ans+=(pow(p, power+1)-1)//(p-1)
#phi(self,n) return ans
#l1: ans=n
#l2: ans-=(ans//p)

obj=MATH_ALGOS()

