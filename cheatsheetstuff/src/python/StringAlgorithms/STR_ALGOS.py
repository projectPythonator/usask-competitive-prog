#kmp works on id stringmatching, irepeatmyself, 
#suffix sort tested on suffixsorting
#longest common prefix(also means suffix array works) repeatedsubstrings, dvaput, substrings

class STR_ALGOS:
    def __init__(self):
        self.n=0
        self.T=''
    
    def prepKMP(self, s):
        self.n=len(s)
        self.T=s
    
    def kmpPreprocess(self, s):
        self.m=len(s)
        self.P=s
        self.B=[0]*(self.m+1) #add plus one if no work
        self.B[0]=j=-1
        for i in range(self.m):
            while j>=0 and self.P[i] != self.P[j]:
                j=self.B[j]
            j+=1
            self.B[i+1]=j
    
    def kmpSearch(self):
        ans=[]
        j=0
        for i in range(self.n):
            while j>=0 and self.T[i] != self.P[j]:
                j=self.B[j]
            j+=1
            if j==self.m:
                ans.append(1+i-j)
                j=self.B[j]
        return ans
    
    def prepSA(self, s):
        k=1
        self.T=s+chr(0)*(100010)
        self.n=len(s)+1
        self.c=[0]*max(300,self.n)
        self.tempSA=[0]*self.n
        self.tempRA=[0]*self.n
        self.RA=[ord(C) for C in self.T[:self.n]]
        self.SA=[i for i in range(self.n)]
        self.K=[]
        while k<self.n:
            self.K.append(k)
            k*=2

    def countingSort(self, k):
        maxi=max(300,self.n)
        tmp=0
        for i in range(maxi):
            self.c[i]=0
        self.c[0]+=(self.n-(self.n-k))
        for i in range(self.n-k):
            self.c[self.RA[i+k]]+=1
        for i in range(maxi):
            t=self.c[i]
            self.c[i]=tmp
            tmp+=t
        for i in range(self.n):
            pos=0
            if self.SA[i]+k<self.n:
                pos=self.RA[self.SA[i]+k]
            self.tempSA[self.c[pos]]=self.SA[i]
            self.c[pos]+=1
        for i, el in enumerate(self.tempSA):
            self.SA[i]=el

    def constructSA(self):
        for k in self.K:
            self.countingSort(k)
            self.countingSort(0)
            self.tempRA[self.SA[0]],r=0,0
            for i in range(1, self.n):
                s1=self.SA[i]
                s2=self.SA[i-1]
                if not (self.RA[s1] == self.RA[s2] and self.RA[s1+k] == self.RA[s2+k]):
                    r+=1
                self.tempRA[s1]=r
            for i,el in enumerate(self.tempRA):
                self.RA[i]=el
            if self.RA[self.SA[-1]]==self.n-1:
                break
        self.tempRA=[]
        self.tempSA=[]
        self.RA=[]
    
    def computeLCP(self):
        self.LCP=[0]*self.n
        self.Phi=[0]*self.n
        self.PLCP=[0]*self.n
        self.Phi[0]=-1
        L=0
        for i in range(1, self.n):
            self.Phi[self.SA[i]]=self.SA[i-1]
        for i in range(self.n):
            if self.Phi[i]==-1:
                self.PLCP[i]=0
                continue
            while self.T[i+L]==self.T[self.Phi[i]+L]:
                L+=1
            self.PLCP[i]=L
            L=max(L-1,0)
        for i in range(self.n):
            self.LCP[i]=self.PLCP[self.SA[i]]
    
    def strncmp(self, ind):
        for i in range(self.m):
            if self.P[i]!=self.T[ind+i]:
                return ord(self.T[ind+i])-ord(self.P[i])
        return 0
    
    def stringMatching(self, s):
        self.P=s
        self.m=len(s)
        lo,hi,mid,b=0,self.n-1,0,len(s)
        while lo<hi:
            mid=(lo+hi)//2
            res=self.strncmp(self.SA[mid])
            if res>=0:
                hi=mid
            else:
                lo=mid+1
        if self.strncmp(self.SA[lo])!=0:
            return (-1,-1)
        alo=lo
        lo,hi,mid=alo,self.n-1,0
        while lo<hi:
            mid=(lo+hi)//2
            res=self.strncmp(self.SA[mid])
            if res>0:
                hi=mid
            else:
                lo=mid+1
        if self.strncmp(self.SA[hi])!=0:
            hi-=1
        return (alo, hi)

    def LRS(self):
        ind,maxLCP=0,-1
        for i in range(1, self.n):
            if self.LCP[i]>maxLCP:
                ind,maxLCP=i,self.LCP[i]
        return (maxLCP, ind)
    
    def owner(self, ind):
        return 1 if ind<self.n-self.m-1 else 2
    
    def LCS(self):
        ind,maxLCP=0,-1
        for i in range(1, self.n):
            if owner(self.SA[i]) != owner(self.SA[i-1]) and self.LCP[i]>maxLCP:
                ind,maxLCP=i,self.LCP[i]
        return (maxLCP, ind)
