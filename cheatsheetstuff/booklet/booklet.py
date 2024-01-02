#data structurea
#union find 
class UnionFind:
    ''' 
    space O(n) --> N*3 or N*2 for now 
    search time α(n) -->  inverse ackerman practically constant 
    insert time O(1) --> 
    '''
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0]*n #optional optimzation 
        self.sizes = [0]*n #optional information
        self.num_sets = n #optional information
        
    def find_set(self, u):
        u_parent = u
        u_children = []
        while u_parent != self.parents[u_parent]:
            u_children.append(u_parent)
            u_parent = self.parents[u_parent]
        for child in u_children:
            self.parents[child] = u_parent
        return u_parent
        
    def is_same_set(self, u, v):
        return self.find_set(u)==self.find_set(v)

    def union_set(self, u, v): # BUG sizes not working needs a fix probably just revert back to book implementtion from steven and felix
        up = self.find_set(u)
        vp = self.find_set(v)
        if up==vp:
            return

        if self.ranks[up] < self.ranks[vp]:
            self.parents[up] = vp
            self.sizes[vp] = self.sizes[up]
        elif self.ranks[vp] < self.ranks[up]:
            self.parents[vp] = up
            self.sizes[up] = self.sizes[vp]
        else:
            self.parents[vp] = up
            self.ranks[up] += 1
            self.sizes[up] += self.sizes[vp]
        self.num_sets -= 1

    def size_of_u(self, u): #optional information
        return self.sizes[self.find_set(u)]

######################################################################################
#

class GRAPH_ALGOS():
    INF=2**31
    UNVISITED = -1
    EXPLORED  = -2
    VISITED   = -3
    
    def __init__(self, V, E, N=None, M=None):
        self.num_edges = E
        self.num_nodes = V
        self.num_rows = N
        self.num_cols = M

    def init_structures(self): #take what you need and leave the rest
        from collections import deque
        self.adj_list = [{} for _ in range(self.num_nodes)]
        #self.edge_list = [(0,0,0)]*self.num_edges
        #self.matrix = [[0]*self.num_cols for _ in range(self.num_rows)]
    
        #self.queue = deque()
        #self.not_visited = set(list(range(self.num_nodes)))
        #self.visited = [UNVISITED]*self.num_nodes
        #self.stk = []
        #self.heap = []
        #self.dir_rc = [(1,0), (0,1), (-1,0), (0,-1)]
        #self.in_degree = [0]*self.num_nodes
        #self.color = [INF]*self.num_nodes
        #self.low_values = [0]*self.num_nodes
        #self.parent = [-1]*self.num_nodes
    
        #self.dist = [INF]*self.num_nodes
        #self.mst_node_set = []
        #self.topo_sort_node_set = []
        #self.articulation_points = []
    
    def append_edge_list(self, w, u, v):
        self.edge_list.append((w,u,v))
        self.num_edges += 1

    def update_adj_list(self, w, u, v):
        self.adj_list[u][v] = w
    
    def update_edge_list(self, edge, w, u, v):
        self.edge_list[edge] = (w,u,v)
        #uv = u*self.num_nodes + v; self.edge_list[edge] = (w,uv)

    def dfs_flood_fill(self, row, col, old_val, new_val): #needs test
        ans,self.matrix[row][col]=1,new_val
        for row_mod,col_mod in dir_rc:
            if 0<=row<self.num_rows and 0<=col<self.num_cols:
                if self.matrix[row][col]==old_val:
                    ans += self.dfs_flood_fill(row+row_mod, col+col_mod, old_val, new_val)
        return ans

    def dfs_topology_sort_helper(self, u):
        self.visited[u] = VISITED
        for v,w in self.adj_list[u]:
            if self.visited[v]==UNVISITED:
                self.dfs_topology_sort_helper(v)
        self.topo_sort_node_set.append(u)
        
    def dfs_topology_sort(self):
        self.topo_sort_node_set = []
        for u in range(self.num_nodes):
            if self.visited[v]==UNVISITED:
                self.dfs_topology_sort_helper(u)
        self.topo_sort_node_set = self.topo_sort_node_set[::-1]

    def dfs_bipartite_checker(self):
        pass # find code for this later

    def dfs_cycle_checker_helper(self, u):
        self.visited[u] = EXPLORED
        for v in self.adj_list[u]:
            info = '{} to {} is a '.format(u, v)
            if self.visited[v]==UNVISTED:
                print(info+'tree edge')
                self.parent[v] = u
                self.dfs_cycle_checker_helper(v)
            elif self.visited[v]==EXPLORED:
                if v == self.parent[u]:
                    print(info+'bidirectional edge')
                else:
                    print(info+'back edge')
            elif self.visited[v]==VISITED:
                print(info+'forward/crossedge')
        self.visited[u] = VISITED

    def dfs_cycle_checker(self):
        for u in range(self.num_nodes):
            if self.visited[u]==UNVISITED:
                self.dfs_cycle_checker_helper(u)
    # dfs_articulation_point_and_bridge_helper tested in advent of code for  removing 3 nodes from a graph on removing the last node 
    # seems to work fine not sure if hidden bugs still here 
    def dfs_articulation_point_and_bridge_helper(self, u): # need to rego over this and test it *** not as confident as the other code atm since have not really used it to solve a problem
        self.visited[u] = self.dfs_counter
        self.low_values[u] = self.visited[u]
        self.dfs_counter += 1
        for v in self.adj_list[u]:
            if self.visited[v]==UNVISITED:
                self.parent[v] = u
                if u==self.dfs_root:
                    self.root_children += 1
                self.dfs_articulation_point_and_bridge_helper(v)
                if self.low_values[v] >= self.visited[u]:
                    self.articulation_points[u] = 1
                if self.low_values[v] > self.visited[u]:
                    print("bridge?")
                self.low_values[u] = min(self.low_values[u], self.low_values[v])
            elif v != self.parents[u]:
                self.low_values[u] = min(self.low_values[u], self.visited[v])

    def dfs_articulation_point_and_bridge(self):
        self.dfs_counter = 0
        for u in range(self.num_nodes):
            if self.visited[u]==UNVISITED:
                self.dfs_root = u
                self.root_children = 0
                self.dfs_articulation_point_and_bridge_helper(u)
                self.articulation_points[self.dfs_root] = (self.root_children>1)
        for i,u in enumerate(self.articulation_points):
            if u:
                print("vertix {}".format(i))

    def dfs_scc_kosaraju_pass1(self, u):
        self.not_visited.remmove(u)
        for v in self.adj_list[u]:
            if v  in self.not_visited:
                self.dfs_scc_kosaraju_pass1(v)
        self.stk.append(u)

    def dfs_scc_kosaraju_pass2(self, u):
        self.not_visited.remove(u)
        self.scc[u] = self.cur_num
        for v in self.adj_list_trans:
            if v in self.not_visited:
                self.dfs_scc_kosaraju_pass2(v)

    def dfs_scc_kosaraju(self):
        for u in self.adj_list:
            if v not in self.not_seen:
                self.dfs_scc_kosaraju_pass1(u)
        self.not_visited = set(list(self.adj_list))
        self.cur_num = 0
        while self.stk:
            while self.stk and self.stk[-1] not in self.not_visited: 
                self.stk.pop()
            if self.stk:
                self.dfs_scc_kosaraju_pass2(self.stk[-1])
                self.cur_num += 1
    
    def dfs_scc_tarjans_helper(self, u):
        self.num_cmp += 1
        self.low_values[u]=self.visited[u]=self.num_cmp
        self.stk.append(u)
        self.not_visited.add(u)
        for v in self.adj_list[u]:
            if self.visited[v]==INF:
                self.dfs_scc_tarjans_helper(v)
                self.low_values[u] = min(self.low_values[u], self.low_values[v])
            elif self.v in self.not_visited:
                self.low_values[u] = min(self.low_values[u], self.visited[v])
        if self.low_values[u]==self.visited[u]:
            self.scc.append(set(self.stk))
            self.stk=[]

    def dfs_scc_tarjans(self):
        from sys import setrecursionlimit
        setrecursionlimit(100000) 
        for u in self.adj_list:
            if self.num[u]==INF:
                self.dfs_scc_tarjans_helper(u)
        pass

    def bfs_vanilla(self, start, end): #needs test
        from collections import deque
        self.queue.append(start); self.dist[start] = 0
        while queue:
            u = self.queue.popleft()
            for v in self.adj_list[u]:
                if self.dist[v]>self.dist[u]+1:
                    self.dist[v]=self.dist[u]+1
                    self.queue.append(v)

    def bfs_flood_fill(self, start_row, start_col, old_val, new_val):  #needs test
        self.stk.append(start_row, start_col)
        while self.stk:
            row,col = self.stk.pop()
            if 0<=row<self.num_rows and 0<=col<self.num_cols:
                if self.matrx[row][col]==old_val:
                    self.matrx[row][col] = new_val
                    for row_mod,col_mod in dir_rc:
                        self.stkk.append((row+row_mod, col+col_mod))

    def bfs_kahns_topological_sort(self):
        from heapq import heappush, heappop
        for list_of_u in self.adj_list:
            for v in list_of_u:
                self.in_degree[v] += 1
        for u in range(self.num_nodes):
            if 0==self.in_degree[u]:
                heapppush(self.heap, u)
        while self.heap:
            u = heappop(self.heap)
            for v in adj_list[u]:
                self.in_degree[v] -= 1
                if self.in_degree[v] <= 0:
                    heappush(self.heap, v)

    def bfs_bipartite_check_helper(self, start):
        from collections import deque
        self.queue.clear()
        self.color[start] = 0
        self.queue.append(start)
        is_bipartite = True
        while self.queue and is_bipartite:
            u = self.queue.popleft()
            for v in self.adj_list[u]:
                if self.color[v]!=INF:
                    is_bipartite = False
                    break
                self.color[v] = not self.color[u]

    def bfs_bipartite_check(self):
        for u in self.adj_list:
            if self.color[u] == INF:
                self.bfs_bipartite_check_helper(u)

    def bfs_cycle_checker(self):
        pass #need to get the implimentation 
        
    def sssp_dijkstras_heaps(self, start, end): #needs test
        from heapq import heappush, heappop
        heappush(self.heap, (0, start))
        self.dist[start] = 0
        self.parent[start] = start
        while self.heap:
            cur_dist, u = heappop(self.heap)
            if self.dist[u]<cur_dist:
                continue
            #if u==end: return cur_dist #uncomment this line for fast return
            for v, weight in self.adj_list[u].items():
                if self.dist[v]>cur_dist+weight:
                    self.dist[v] = cur_dist+weight
                    self.parent[v] = u
                    heappush(self.heap, (self.dist[v], v))
        return self.dist[end]

    def apsp_floyd_warshall(self): #needs test
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.matrix[i][j] = min(self.matrix[i][j], self.matrix[i][k]+self.matrix[k][j])

    def apsp_floyd_warshall_neg_cycles(self): #needs test
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for k in range(self.num_nodes):
                    if self.matrix[k][k]<0 and self.matrix[i][k]!=INF and self.matrix[k][j]!=INF:
                        self.matrix[i][j]=-INF

    #will kill the edge list but will save memory
    def mst_kruskals_heaps(self):  #needs test
        from heapq import heapify, heappop
        UF=UnionFind(self.num_nodes)
        heapify(self.edge_list)
        while self.edge_list and  UF.num_sets>1:
            w,u,v = heappop(self.edge_list) #use w, uv = ... for single cord storage
            #v,u = uv%self.num_nodes, uv//self.num_nodes
            if not UF.is_same_set(u,v):
                self.mst_node_set.append((w,u,v))
                UF.union_set(u,v)
        return self.mst_node_set
        
    def mst_prims_process_complete(self, u):  #needs test
        from heapq import heappush
        self.not_visited.remove(u)
        for v in self.not_visited:
            uv_dist = get_dist(u, v)
            if uv_dist<=self.dist[v]:
                self.dist[v] = uv_dist
                heappush(self.heap, (uv_dist, v, u))
    
    def mst_prims_process(self, u): #needs test
        from heapq import heappush
        self.not_visited.remove(u)
        for v, w in self.adj_list[u].items():
            if v in self.not_visited and w<=self.dist[v]:
                self.dist[v] = w
                heappush(self.heap, (w, v, u))
    
    def mst_prims(self):  #needs test
        from heapq import heappop
        self.prims_process(0)
        nodes_taken = 0
        while self.heap and nodes_taken<self.num_nodes:
            w,v,u = heappop(self.heap)
            if v in self.not_visited:
                self.prims_process(v)
                self.mst_node_set.append((w,v,u))
                nodes_taken += 1
        self.mst_node_set.sort()
        return self.mst_node_set



class MATH_ALGOS:
    def __init__(self):
        self.n=None
        self.primes_sieve = []
        self.primes_list = []
        self.primes_set = set()
        self.prime_factors = []
        self.mrpt_known_bounds = []
        self.mrpt_known_tests = []
        self.fibonacci_list = []
        self.fibonacci_dict = {}

    def init_data(self, n): #call before using other functions, make a reset if needed to reset per case
        self.primes_sieve = [True] * n
        self.primes_list = []
        self.primes_set = set()
        self.prime_factors = []
        
    def is_prime_triv(self, n):
        if n <= 3:
            return n > 1
        elif n%2 == 0 or n%3 == 0:
            return False
        p=5
        while p*p <= n:
            if n%p == 0 or n%(p+2) == 0:
                return False
            p+=6
        return True
        
    def sieve_primes(self, n):
        self.primes_list = [2]
        for i in range(4, n, 2):
            self.primes_sieve[i] = False
        for i in range(3, n, 2):
            if self.primes_sieve[i]:
                self.primes_list.append(i)
                for j in range(i*i, n, 2*i):
                    self.primes_sieve[j] = False
    
    def gen_set_primes(self):
        self.primes_set=set(self.primes_list)

    def prime_factorize(self, n):
        self.prime_factors = [] # clobbers the previous prime factors so save before calling this func
        for p in self.primes_list:
            if p*p > n:
                break
            if n%p == 0:
                while n%p == 0:
                    n//=p
                    self.prime_factors.append(p)
        if n > 1:
            self.prime_factors.append(n)

   def is_composite(self, a, d, n, s):
        if pow(a, d, n)==1:
            return False
        for i in range(s):
            if pow(a, 2**i * d, n)==n-1:
                return False
        return True 
    
    def is_prime_mrpt(self, n, precision_for_huge_n=16):
        if n in self.primes_set:
            return True
        if any((n%self.primes_list[p] == 0) for p in range(50)) or n < 2 or n == 3215031751:
            return False
        d, s = n-1, 0
        while not d % 2:
            d, s = d//2, s+1
        for i, bound in enumerate(self.mrpt_known_bounds):
            if n < bound:
                return not any(self.is_composite(self.mrpt_known_tests[j], d, n, s) for j in range(i))
        return not any(self.is_composite(self.primes_list[j], d, n, s) for j in range(precision_for_huge_n))
    
    def prep_mrpt(self):
        self.mrpt_known_bounds = [1373653, 25326001, 118670087467, 2152302898747, 3474749660383, 341550071728321]
        self.mrpt_known_tests = [2, 3, 5, 7, 11, 13, 17]
        self.sieve_primes(1000) #comment out if different size needed
        self.gen_set_primes() #comment out if already have bigger size

    #test this against stevens
    def extended_euclid(self, a, b):
        if 0 == b:
            return 1, 0, a
        x, y, d = self.extended_euclid(b, a%b)
        return y, x-y*(a//b), d

    def extended_euclid_2(self, a, b):
        xx, yy = 0, 1
        x, y = 1, 0
        while b != 0:
            q = a//b
            a, b = b, a%b
            x, xx = xx, x-q*xx
            y, yy = yy, y-q*yy
        return a, x, y

    # use in c++ and java
    # use ((a % n) + n) % n for getting proper mod of negative value 
    # use (a + b) % --> ((a % n) + (b % n)) % n for operations sub out + for * and - 
    def mod(self, a, n): #needs test
        return ((a % n) + n) % n

    def modular_linear_equation_solver(self, a, b, n): #needs test
        x, y, d = self.extended_euclid(a, n)
        if 0 == b % d:
            x = self.mod(x*(b//d), n)
            return [self.mod(x+i*(n//d), n) for i in range(d)]
        return []

    def mod_inverse(self, a, n): #needs test
        x, y, d = self.extended_euclid(a, n)
        return -1 if d > 1 else self.mod(x, n)

    # stanford icpc 2013-14
    def crt_helper(self, x, a, y, b): #needs test
        s, t, d = self.extended_euclid(x, y)
        return (0, -1) if a%d != b%d else (self.mod(s*b*x + t*a*y, x*y)//d, x*y//d)
    
    # from stanford icpc 2013-14
    def chinese_remainder_theorem(self, x, a):
        ans = (a[0], x[0])
        for i in range(1, len(x)):
            ans = self.crt_helper(ans[1], ans[0], x[i], a[i])
            if -1 == ans[1]:
                break
        return ans
    
    #test this its from stanford icpc 2013-14
    #computes x and y in ax+by=c failure x=y=-1
    def linear_diophantine(self, a, b, c):
        d = math.gcd(a,b)
        if c%d == 0:
            x = c//d * self.mod_inverse(a//b, b//d)
            return (x, (c - a*x//b))
        return (-1, -1)

    def fibonacci_n_iter(self, n):
        self.fibonacci_list = [0] * (n+1)
        self.fibonacci_list[1] = 1
        for i in range(2, n+1):
            self.fibonacci_list[i] = self.fibonacci_list[i-1] + self.fibonacci_list[i-2]
        return self.fibonacci_list[n]

    def fibonacci_n_dp_helper(self, n):
        if n == 0:
            return 0
        if n < 3:
            self.fibonacci_dict[n] = 1
            return 1
        if n in self.fibonacci_dict:
            return self.fibonacci_dict[n]
        if n%2 == 1:
            k = (n+1)//2
            fib_1 = self.fibonacci_n_dp_log_n(k)
            fib_2 = self.fibonacci_n_dp_log_n(k-1)
            self.fibonacci_dict[n] = fib_1*fib_1 + fib_2*fib_2
        else:
            k = n//2
            fib_1 = self.fibonacci_n_dp_log_n(k)
            fib_2 = self.fibonacci_n_dp_log_n(k-1)
            self.fibonacci_dict[n] = (2*fib_2 + fib_1)*fib_1
        return self.fibonacci_dict[n]

    def fibonacci_n_dp(self, n):
        self.fibonacci_dict = {}
        return self.fibonacci_n_dp_helper(n)
    
    #this needs testing 
    def generate_catalan_n(self, n):
        self.catalan = [0] * (n+1)
        self.catalan[0] = 1
        for i in range(n-1):
            self.catalan[i+1] = self.catalan[i]*(4*i+2)//(i+2)

    def generate_catalan_n_mod_inverse(self, n, p):
        self.catalan = [0] * (n+1)
        self.catalan[0] = 1
        for i in range(n-1):
            self.catalan[i+1] = ((4*i+2)%p * self.catalan[i]%p * pow(i+1, p-2, p)) % p

    


    def catalan_n_mod_p_helper(self, table, val):
        self.prime_factorize(val)
        factor_tally = Counter(self.prime_factors)
        for k, v in factor_tally.items():
            if k not in table:
                table[k] = 0
            table[k] += v

    def catalan_n_mod_p(self, n, p):
        # from collections import counter needs to be imported
        self.sieve_primes(int((5*n)**0.5))
        tpf = {}
        bpf = {}
        for i in range(n):
            self.catalan_n_mod_p_helper(tpf, 4*i+2)
            self.catalan_n_mod_p_helper(bpf, i+2)
        for k, v in bpf.items():
            tpf[k] -= v
        ans = 1
        for k, v in tpf.items():
            if v > 0:
                ans *= pow(k, v, p)
        return ans % p

    def binomial_coefficient(self, n, k):
        k = min(k, n-k)
        res = 1
        for i in range(k):
            res *= (n-i)
            res //= (i+1)
        return res
            
    def binomial_coefficient_dp(self, n, k):
        if n == k or 0 == k:
            return 1
        if (n, k) not in self.binomial:
            self.binomial[(n, k)] = self.binomial_coefficient_dp(n-1, k) + self.binomial_coefficient_dp(n-1, k-1)
        return self.binomial[(n, k)]

from math import isclose, dist, sin, cos, acos, sqrt
# remember to sub stuff out for integer ops when you want only integers 
# for ints need to change init, eq and 
class pt_xy:
    def __init__(self, x_val, y_val): 
        self.x, self.y = map(float, [x_val, y_val])

    def __add__(self, other): return pt_xy(self.x+other.x, self.y+other.y)
    def __sub__(self, other): return pt_xy(self.x-other.x, self.y-other.y)
    def __mul__(self, scale): return pt_xy(self.x*scale, self.y*scale)
    def __truediv__(self, scale): return pt_xy(self.x/scale, self.y/scale)
    def __floordiv__(self, scale): return pt_xy(self.x//scale, self.y//scale)

    def __eq__(self, other): return isclose(self.x, other.x) and isclose(self.y, other.y)
    def __lt__(self, other): return False if self == other else (self.x, self.y) < (other.x, other.y)

    def __str__(self): return "{} {}".format(self.x, self.y)
    def __str__(self): return "(x = {:20}, y = {:20})".format(self.x, self.y)
    def __round__(self, n): return pt_xy(round(self.x, n), round(self.y, n))
    def __hash__(self): return hash((self.x, self.y))

    def get_tup(self): return (self.x, self.y)


class pt_xyz:
    def __init__(self, x_val, y_val, z_val): 
        self.x, self.y, self.z = map(float, [x_val, y_val, z_val])

    def __add__(self, other):
        return pt_xyz(self.x+other.x, self.y+other.y, self.z+other.z)
    def __sub__(self, other): 
        return pt_xyz(self.x-other.x, self.y-other.y, self.z-other.z)
    def __mul__(self, scale): 
        return pt_xyz(self.x*scale, self.y*scale, self.z*scale)
    def __truediv__(self, scale): 
        return pt_xyz(self.x/scale, self.y/scale, self.z/scale)
    def __floordiv__(self, scale): 
        return pt_xyz(self.x//scale, self.y//scale, self.z//scale)

    def __eq__(self, other): 
        return isclose(self.x, other.x) and isclose(self.y, other.y) and isclose(self.z, other.z)
    def __lt__(self, other):
        return False if self == other else (self.x, self.y, self.z) < (other.x, other.y, other.y)

    def __str__(self): 
        return "{} {} {}".format(self.x, self.y, self.z)
    def __str__(self): 
        return "(x = {:20}, y = {:20}), z = {:20})".format(self.x, self.y, self.z)
    def __round__(self, n): 
        return pt_xyz(round(self.x, n), round(self.y, n), round(self.z, n)
    def __hash__(self):
        return hash((self.x, self.y, self.z))

def Geometry_Algorithms:
    def __init__(self):
        pass
    # replacing epscmp and c_cmp for epscmp simply use compare_ab(a, 0) or math.isclose(a, 0)
    def compare_ab(self, a, b): return 0 if isclose(a, b) else -1 if a<b else 1

    def dot_product_2d(self, a, b): return a.x*b.x + a.y*b.y
    def cross_product_2d(self, a, b): return a.x*b.y - a.y*b.x

    def distance_normalized_2d(self, a, b): return math.dist(a.get_tup(), b.get_tup())
    def distance_2d(self, a, b): return self.dot_product_2d(a-b, a-b)

    def rotate_cw_90_wrt_origin_2d(self, pt): return pt_xy(pt.y, -pt.x)
    def rotate_ccw_90_wrt_origin_2d(self, pt): return pt_xy(-pt.y, pt.x)
    def rotate_ccw_rad_wrt_origin_2d(self, pt, rad):
        return pt_xy(pt.x*cos(rad) - pt.y*sin(rad), 
                     pt.x*sin(rad) + pt.y*cos(rad))

    # 0 if colinear else 1 if counter clock wise (ccw) else -1 if clockwise (cw) 
    def point_c_rotation_wrt_line_ab_2d(self, a, b, c):
        return self.compare_ab(self.cross_product_2d(b-a, c-a), 0.0)

    def angle_point_c_wrt_line_ab_2d(self, a, b, c): # possibly doesn't work for some (probably overflow)
        ab, cb = a-b, c-b
        abcb = self.dot_product_2d(ab, cb)
        abab = self.dot_product_2d(ab, ab)
        cbcb = self.dot_product_2d(cb, cb)
        # return acos(abcb/sqrt(abab*cbcb))
        return acos(abcb/(sqrt(abab)*sqrt(cbcb)))

    # projection funcs just returns closes point to obj based on a point c
    def project_pt_c_to_line_ab_2d(self, a, b, c):
        ba, ca = b-a, c-a
        return a + ba*(self.dot_product_2d(ca, ba)/self.dot_product_2d(ba, ba))

    # use compare_ab in return if this isn't good enough
    def project_pt_c_to_line_seg_ab_2d(self, a, b, c):
        ba, ca = b-a, c-a
        u = self.dot_product_2d(ba, ba)
        if self.compare_ab(u, 0.0) == 0:
            return a
        u = self.dot_product_2d(ca, ba)/u
        return a if u < 0.0 else b if u > 1.0 else self.project_pt_c_to_line_ab_2d(a, b, c)

    def distance_pt_c_to_line_ab_2d(self, a, b, c):
        return self.distance_normalized_2d(c, self.project_pt_c_to_line_ab_2d(a, b, c))

    def distance_pt_c_to_line_seg_ab_2d(self, a, b, c):
        return self.distance_normalized_2d(c, self.project_pt_c_to_line_seg_ab_2d(a, b, c))
    
    def is_parallel_lines_ab_and_cd_2d(self, a, b, c, d):
        return self.compare_ab(self.cross_product_2d(b-a, c-d), 0.0) == 0

    def is_collinear_lines_ab_and_cd_2d(self, a, b, c, d):
        return (self.is_parallel_lines_ab_and_cd_2d(a, b, c, d)
        and self.is_parallel_lines_ab_and_cd_2d(b, a, a, c)
        and self.is_parallel_lines_ab_and_cd_2d(d, c, c, a))

    def is_segments_intersect_ab_to_cd_2d(self, a, b, c, d):
        if self.is_collinear_lines_ab_and_cd_2d(a, b, c, d):
            lo, hi = a, b if a < b else b, a
            return lo <= c <= hi or lo <= d <= hi
        a_val = self.cross_product_2d(d-a, b-a)*self.cross_product_2d(c-a, b-a)
        c_val = self.cross_product_2d(a-c, d-c)*self.cross_product_2d(b-c, d-c)
        return not(a_val>0 or c_val>0)

    def is_lines_intersect_ab_to_cd_2d(self, a, b, c, d):
        return (not self.is_parallel_lines_ab_and_cd_2d(a, b, c, d) or 
                self.is_collinear_lines_ab_and_cd_2d(a, b, c, d))

    def pt_lines_intersect_ab_to_cd_2d(self, a, b, c, d):
        ba, ca, cd = b-a, c-a, c-d
        return a + ba*(self.cross_product_2d(ca, cd)/self.cross_product_2d(ba, cd))

    def pt_line_seg_intersect_ab_to_cd_2d(self, a, b, c, d):
        x, y, cross_prod = c.x-d.x, d.y-c.y, self.cross_product_2d(d, c)
        u = abs(y*a.x + x*a.y + cross_prod)
        v = abs(y*b.x + x*b.y + cross_prod)
        return pt_xy((a.x*v + b.x*u)/(v + u), (a.y*v + b.y*u)/(v + u))

    def is_point_in_circle(self, a, b, r): # use <= if you want points on the circumfrance 
        return self.compare_ab(self.distance_normalized_2d(a, b), r) < 0

    def pt_circle_center_given_pt_abc(self, a, b, c):
        ab, ac = (a+b)/2, (a+c)/2
        ab_rot, ac_rot = ab+self.rotate_cw90(a-ab), ac+self.rotate_cw90(a-ac)
        return self.pt_lines_intersect_ab_to_cd_2d(ab, ab_rot, ac, ac_rot)


            
    
        
    
    


