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
        self.sizes = [1]*n #optional information
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
            self.sizes[vp] += self.sizes[up]
        elif self.ranks[vp] < self.ranks[up]:
            self.parents[vp] = up
            self.sizes[up] += self.sizes[vp]
        else:
            self.parents[vp] = up
            self.ranks[up] += 1
            self.sizes[up] += self.sizes[vp]
        self.num_sets -= 1

    def size_of_u(self, u): #optional information
        return self.sizes[self.find_set(u)]

######################################################################################
#
from collections import deque
from heapq import heappush, heappop, heapreplace

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

    def max_flow_bfs(self, source, sink):
        self.dist = [-1] * self.num_nodes
        self.path = [[-1, -1] for _ in range(self.num_nodes)]
        self.queue = deque([source])
        self.dist[s] = 0
        while self.queue:
            u = self.queue.pop_left()
            if u == sink:
                break
            for idx in self.adj_list[u]:
                v, cap, flow = self.edge_list[idx]
                if cap - flow > 0 and self.dist[v] == -1:
                    self.dist[v] = self.dist[u] + 1
                    self.queue.append(v)
                    self.path[v] = [u, idx]
        return self.dist[sink] != -1

    def max_flow_dfs(self, u, sink, flow_in):
        if u == sink or flow_in == 0:
            return flow_in
        for i in range(self.last[u], len(self.adj_list[u])):
            self.last[u] = i
            v, edge_cap, edge_flow = self.edge_list[self.adj_list[u][i]]
            if self.dist[v] != self.dist[u] + 1:
                continue
            pushed_flow = self.max_flow_dfs(v, sink, min(flow_in, edge_cap - edge_flow))
            if pushed_flow != 0:
                flow_in += pushed_flow
                self.edge_list[self.adj_list[u][i]][2] = edge_flow
                self.edge_list[self.adj_list[u][i] ^ 1][2] -= pushed_flow
                return pushed_flow
        return 0

    def max_flow_send_one_flow(self, source, sink, flow_in):
        if source == sink:
            return flow_in
        u, edge_ind = self.path[sink]
        _, edge_cap, edge_flow = self.edge_list[edge_ind]
        pushed_flow = self.max_flow_send_one_flow(sink, u, min(flow_in, edge_cap - edge_flow))
        self.edge_list[edge_ind][2] = edge_flow + pushed_flow
        self.edge_list[edge_ind ^ 1][2] -= pushed_flow
        return pushed_flow

    def max_flow_add_edge(self, u, v, capacity, directed):
        if u == v:
            return
        self.edge_list.append([v, capacity, 0])
        self.adj_list[u].append(len(self.edge_list) - 1)
        self.edge_list.append([u, 0 if directed else capacity, 0])
        self.adj_list[v].append(len(self.edge_list) - 1)

    def edmonds_karp(self, source, sink):
        max_flow = 0
        while self.max_flow_bfs(source, sink):
            flow = self.max_flow_send_one_flow(source, sink, inf)
            if flow == 0:
                break
            max_flow += flow
        return max_flow

    def dinic(self, source, sink):
        max_flow = 0
        while self.max_flow_bfs(source, sink):
            self.last = [0] * self.num_nodes
            flow = self.max_flow_dfs(source, sink, inf)
            while flow != 0:
                max_flow += flow
                flow = self.max_flow_dfs(source, sink, inf)
        return max_flow
        

class Math_Algorithms:
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
        return -1 if d > 1 else (x + n) % n

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

from math import isclose, dist, sin, cos, acos, sqrt, fsum, pi, tau, atan2
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

class Quad_Edge:
    def __init__(self):
        self.origin = pt_xy(0, 0)
        self.rot = None
        self.o_next = None
        self.used = False

    def rev(self): return self.rot.rot
    def l_next(self): return self.rot.rev().o_next.rot
    def o_prev(self): return self.rot.o_next.rot
    def dest(self): return self.rev().origin

class Quad_edge_data_structure:
    def __init__(self):
        pass

    def make_edge(self, in_pt, out_pt):
        e1 = Quad_Edge()
        e2 = Quad_Edge()
        e3 = Quad_Edge()
        e4 = Quad_Edge()
        e1.origin = in_pt
        e2.origin = out_pt
        e3.origin = pt_xy(2**63, 2**63)
        e4.origin = pt_xy(2**63, 2**63)
        e1.rot = e3
        e2.rot = e4
        e3.rot = e2
        e4.rot = e1
        e1.o_next = e1
        e2.o_next = e2
        e3.o_next = e4
        e4.o_next = e3
        return e1

    def splice(self, a, b):
        a.o_next.rot.o_next, b.o_next.rot.o_next =  b.o_next.rot.o_next, a.o_next.rot.o_next
        a.o_next, b.o_next = b.o_next, a.o_next

    def delete_edge(self, edge):
        self.splice(edge, edge.o_prev())
        self.splice(edge.rev(), edge.rev().o_prev())
        del edge.rev().rot
        del edge.rev()
        del edge.rot
        del edge

    def connect(self, a, b):
        e = self.make_edge(a.dest(), b.origin)
        self.splice(e, a.l_next())
        self.splice(e.rev(), b)
        return e

class Geometry_Algorithms:
    def __init__(self):
        self.quad_edges = Quad_Edge_Data_Structure()
        
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
        ab_rot = ab+self.rotate_cw_90_wrt_origin_2d(a-ab)
        ac_rot = ac+self.rotate_cw_90_wrt_origin_2d(a-ac)
        return self.pt_lines_intersect_ab_to_cd_2d(ab, ab_rot, ac, ac_rot)

    def pts_line_ab_intersects_circle_cr_2d(self, a, b, c, r):
        ba, ac = b-a, a-c
        bb = self.dot_product_2d(ba, ba)
        ab = self.dot_product_2d(ac, ba)
        aa = self.dot_product_2d(ac, ac)-r*r
        dist = ab*ab - bb*aa
        result = self.compare_ab(dist, 0.0)
        if result >= 0:
            first_intersect = c + ac + ba*(-ab + sqrt(dist+EPS))/bb
            second_intersect = c + ac + ba*(-ab - sqrt(dist))/bb
            return (first_intersect) if result == 0 else (first_intersect, second_intersect)
        return None # no intersect 

    def pts_two_circles_intersect_ar1_br1_2d(self, c1, c2, r1, r2):
        center_dist = self.distance_normalized_2d(c1, c2)
        if self.compare_ab(center_dist, r1+r2) <= 0 \
        and self.compare_ab(center_dist+min(r1, r2), max(r1, r2)) >= 0:
            x = (center_dist*center_dist - r2*r2 + r1*r1)/(2*center_dist)
            y = sqrt(r1*r1 - x*x)
            v = (b-a)/center_dist
            pt1, pt2 = a + v*x, self.rotate_ccw_90_wrt_origin_2d(v)*y
            return (pt1+pt2) if self.compare_ab(y, 0.0) <= 0 else (pt1+pt2, pt1-pt2)
        return None # no overlap

    def pt_tangent_to_circle_cr_2d(self, c, r, p):
        pc = p-c
        x = self.dot_product_2d(pc, pc)
        dist = x - r*r
        result = self.compare_ab(dist, 0.0)
        if result >= 0:
            dist = dist if result else 0
            q1 = pa * (r*r / x)
            q2 = self.rotate_ccw_90_wrt_origin_2d(pa * (-r * sqrt(dist)/x))
            return [a+q1-q2, a+q1+q2]
        return []

    def tangents_between_2_circles_2d(self, c1, r1, c2, r2):
        r_tangents = []
        if self.compare_ab(r1, r2) == 0:
            c2c1 = c2 - c1
            multiplier = r1/sqrt(self.dot_product_2d(c2c1, c2c1))
            tangent = self.rotate_ccw_90_wrt_origin_2d(c2c1 * multiplier) # need better name
            r_tangents = [(c1+tangent, c2+tangent), (c1-tangent, c2-tangent)]
        else:
            ref_pt = ((c1 * -r2) + (c2 * r1)) / (r1 - r2)
            ps = self.pt_tangent_to_circle_cr_2d(c1, r1, ref_pt)
            qs = self.pt_tangent_to_circle_cr_2d(c2, r2, ref_pt)
            r_tangents = [(ps[i], qs[i]) for i in range(min(len(ps), len(qs)))]
        ref_pt = ((c1 * r2) + (c2 * r1)) / (r1 + r2)
        ps = self.pt_tangent_to_circle_cr_2d(c1, r1, ref_pt)
        qs = self.pt_tangent_to_circle_cr_2d(c2, r2, ref_pt)
        for i in range(min(len(ps), len(qs))):
            r_tangents.append((ps[i], qs[i]))
        return r_tangents

    def sides_of_triangle_abc_2d(self, a, b, c):
        ab = self.distance_normalized_2d(a, b)
        bc = self.distance_normalized_2d(b, c)
        ca = self.distance_normalized_2d(c, a)
        return ab, bc, ca

    def pt_p_in_triangle_abc_2d(self, a, b, c, p):
        return self.point_c_rotation_wrt_line_ab_2d(a, b, p) >= 0 and  \
                self.point_c_rotation_wrt_line_ab_2d(b, c, p) >= 0 and \
                self.point_c_rotation_wrt_line_ab_2d(c, a, p) >= 0

    def perimeter_of_triangle_abc_2d(self, ab, bc, ca):
        return ab + bc + ca

    def triangle_area_bh_2d(self, b, h):
        return b*h/2

    def triangle_area_heron_abc_2d(self, ab, bc, ca):
        s = self.perimeter_of_triangle_abc_2d(ab, bc, ca) / 2
        return sqrt(s * (s-ab) * (s-bc) * (s-ca))

    def triangle_area_cross_product_abc_2d(self, a, b, c):
        ab = self.cross_product_2d(a, b)
        bc = self.cross_product_2d(b, c)
        ca = self.cross_product_2d(c, a)
        return (ab + bc + ca)/2

    def incircle_radis_of_triangle_abc_helper_2d(self, ab, bc, ca):
        area = self.triangle_area_heron_abc_2d(ab, bc, ca)
        perimeter = self.perimeter_of_triangle_abc_2d(ab, bc, ca)/2
        return area/perimeter

    def incircle_radis_of_triangle_abc_2d(self, a, b, c):
        ab, bc, ca = self.sides_of_triangle_abc_2d(a, b, c)
        return self.incircle_radis_of_triangle_abc_helper_2d(ab, bc, ca)

    def circumcircle_radis_of_triangle_abc_helper_2d(self, ab, bc, ca):
        area = self.triangle_area_heron_abc_2d(ab, bc, ca)
        return (ab*bc*ca) / (4*area)
        
    def circumcircle_radis_of_triangle_abc_2d(self, a, b, c):
        ab, bc, ca = self.sides_of_triangle_abc_2d(a, b, c)
        return circumcircle_radis_of_triangle_abc_helper_2d(ab, bc, ca)

    def incircle_pt_for_triangle_abc_2d(self, a, b, c):
        radius = self.incircle_radis_of_triangle_abc_2d(a, b, c)
        if self.compare_ab(radius, 0.0):
            return (False, 0, 0)
        dist_ab = self.distance_normalized_2d(a, b)
        dist_bc = self.distance_normalized_2d(b, c)
        dist_ac = self.distance_normalized_2d(a, c)
        ratio_1 = dist_ab/dist_ac
        ratio_2 = dist_ab/dist_bc
        pt_1 = b + (c-b) * (ratio_1/(ratio_1 + 1.0)) 
        pt_2 = a + (c-a) * (ratio_2/(ratio_2 + 1.0))

        if self.is_lines_intersect_ab_to_cd_2d(a, pt_1, b, pt_2):
            intersection_pt = self.pt_lines_intersect_ab_to_cd_2d(a, pt_1, b, pt_2)
            return (True, radius, round(intersection_pt, 12)) # can remove the round function
        return (False, 0, 0)

    def triangle_circle_center_pt_abcd_2d(self, a, b, c, d):
        ba, dc = b-a, d-c
        pt_1, pt_2 = pt_xy(ba.y, -ba.x), pt_xy(dc.y, -dc.x)
        cross_product_1_2 = self.cross_product_2d(pt_1, pt_2)
        cross_product_2_1 = self.cross_product_2d(pt_2, pt_1)
        if self.compare_ab(cross_product_1_2, 0.0) == 0:
            return None
        pt_3 = pt_xy(self.dot_product_2d(a, pt_1), self.dot_product_2d(c, pt_2))
        x = ((pt_3.x * pt_2.y) - (pt_3.y * pt_1.y)) / cross_product_1_2
        y = ((pt_3.x * pt_2.x) - (pt_3.y * pt_1.x)) / cross_product_2_1
        return pt_xy(x, y) 

    def angle_bisector_for_triangle_abc_2d(self, a, b, c):
        dist_ba = self.distance_normalized_2d(b, a)
        dist_ca = self.distance_normalized_2d(c, a)
        ref_pt = (b-a) / dist_ba * dist_ca
        return ref_pt + (c-a) + a

    def perpendicular_bisector_for_triangle_ab_2d(self, a, b):
        ba = b-a
        ba = pt_xy(-ba.y, ba.x)
        return ba + (a+b)/2

    def incircle_pt_of_triangle_abc_v2_2d(self, a, b, c):
        abc = self.angle_bisector_for_triangle_abc_2d(a, b, c)
        bca = self.angle_bisector_for_triangle_abc_2d(b, c, a)
        return self.triangle_circle_center_pt_abcd_2d(a, abc, b, bca)

    def circumcenter_pt_of_triangle_abc_v2_2d(self, a, b, c):
        ab = self.perpendicular_bisector_for_triangle_ab_2d(a, b)
        bc = self.perpendicular_bisector_for_triangle_ab_2d(b, c)
        ab2, bc2 = (a+b)/2, (b+c)/2
        return self.triangle_circle_center_pt_abcd_2d(ab2, ab, bc2, bc)

    def orthocenter_pt_of_triangle_abc_v2_2d(self, a, b, c):
        return a + b + c - 2*self.circumcenter_pt_of_triangle_abc_v2_2d(a, b, c)

    # note these assume counter clockwise ordering of points
    def perimeter_of_polygon_pts_2d(self, pts):
        return fsum([self.distance_normalized_2d(pts[i], pts[i+1]) for i in range(len(pts)-1)])

    def signed_area_of_polygon_pts_2d(self, pts):
        return fsum([self.distance_normalized_2d(pts[i], pts[i+1]) for i in range(len(pts)-1)])/2

    def area_of_polygon_pts_2d(self, pts):
        return abs(self.signed_area_of_polygon_pts_2d(pts))

    # < is counter clock wise <= includes collinear > for clock wise >= includes collinear
    def is_convex_helper(self, a, b, c):
        return 0 < self.point_c_rotation_wrt_line_ab_2d(a, b, c)

    def is_convex_polygon_pts_2d(self, pts):
        lim = len(pts)
        if lim > 3:
            is_ccw = self.is_convex_helper(pts[0], pts[1], pts[2])
            for i in range(1, n-1):
                a, b, c = pts[i], pts[i+1], pts[i+2 if i+2<lim else 1]
                if base != self.is_convex_helper(a, b, c):
                    return False
            return True 
        return False

    def pt_p_in_polygon_pts_v1_2d(self, pts, p):
        n = len(pts)
        if n > 3:
            angle_sum = 0.0
            for i in range(n-1):
                if 1 == self.point_c_rotation_wrt_line_ab_2d(pts[i], pts[i+1], p):
                    angle_sum += angle_point_c_wrt_line_ab_2d(pts[i], pts[i+1], p)
                else:
                    angle_sum -= angle_point_c_wrt_line_ab_2d(pts[i], pts[i+1], p)
            return self.compare_ab(abs(angle_sum), pi)
        return -1

    def pt_p_in_polygon_pts_v2_2d(self, pts, p):
        ans = False
        px, py = p.get_tup()
        for i in range(len(pts)-1):
            x1, y1 = pts[i].get_tup()
            x2, y2 = pts[i+1].get_tup()
            lo, hi = y1, y2 if y1 < y2 else y2, y1
            if lo <= py < hi and px < (x1 + (x2-x1) * (py-y1) / (y2-y1)):
                ans = not ans
        return ans

    def pt_p_on_polygon_perimeter_pts_2d(self, pts, p):
            n = len(pts)
            if p in pts:
                return True
            for i in range(n-1):
                dist_ip = self.distance_normalized_2d(pts[i], p)
                dist_pj = self.distance_normalized_2d(p, pts[i+1])
                dist_ij = self.distance_normalized_2d(pts[i], pts[i+1])
                if self.compare_ab(dist_ip+dist_pj, dist_ij) == 0:
                    return True
            return False

    def pt_p_in_convex_polygon_pts_2d(self, pts, p):
        n = len(pts)
        if n == 2:
            distance = self.distance_pt_c_to_line_seg_ab_2d(pts[0], pts[1], p)
            return self.compare_ab(distance, 0.0) == 0
        left, right = 1, n
        while left < right:
            mid = (left + rigth)/2 + 1
            side = self.point_c_rotation_wrt_line_ab_2d(pts[0], pts[mid], p)
            left, right = mid, right if side == 1 else left, mid-1
        side = self.point_c_rotation_wrt_line_ab_2d(pts[0], pts[left], p)
        if side == -1 or left == n:
            return False
        side = self.point_c_rotation_wrt_line_ab_2d(pts[left], pts[left+1] - pts[left], p)
        return side >= 0
    
    # use a set with points if possible checking on the same polygon many times    
    # return 0 for on 1 for in -1 for out
    def pt_p_position_wrt_polygon_pts_2d(self, pts, p):
        return 0 if self.pt_p_on_polygon_perimeter_pts_2d(pts, p) \
                else 1 if self.pt_p_in_polygon_pts_v2_2d(pts, p) else -1

    def centroid_pt_of_convex_polygon_2d(self, pts):
        ans, n = pt_xy(0, 0), len(pts)
        for i in range(n-1):
            ans = ans + (pts[i]+pts[i+1]) * self.cross_product_2d(pts[i], pts[i+1])
            return ans / (6.0 * self.signed_area_of_polygon_pts_2d(pts))

    def is_polygon_pts_simple_quadratic_2d(self, pts):
        n = len(pts)
        for i in range(n-1):
            for k in range(i+1, n-1):
                j, l = (i+1) % n, (k+1) % n
                if i == l or j == k:
                    continue
                if self.is_segments_intersect_ab_to_cd_2d(pts[i], pts[j], pts[k], pts[l]):
                    return False
        return True

    def polygon_cut(self, pts, a, b):
        ans, n = [], len(pts)
        for i in range(n-1):
            rot_1 = self.point_c_rotation_wrt_line_ab_2d(a, b, pts[i])
            rot_2 = self.point_c_rotation_wrt_line_ab_2d(a, b, pts[i+1])
            if 1 == rot_1:
                ans.append(pts[i])
            elif 0 == rot_1:
                ans.append(pts[i])
                continue
            if 1 == rot_1 and -1 == rot_2:
                ans.append(self.pt_line_seg_intersect_ab_to_cd_2d(pts[i], pts[i+1], a, b))
        if ans and ans[0] != ans[-1]:
            ans.append(ans[0])
        return ans

    def convex_hull_monotone_chain(self, pts):
        def func(points, r, lim):
            for p in points:
                while len(r) > lim and \
                    self.point_c_rotation_wrt_line_ab_2d(r[-2], r[-1], p)) == -1:
                    r.pop()
                r.append(p)
            r.pop()
        ans, convex = sorted(set(pts)), []
        if len(ans) < 2: 
            return ans
        func(ans, convex, 1)
        func(ans[::-1], convex, len(r)+1)
        return r
    
    def rotating_caliper_of_polygon_pts_2d(self, pts):
        n, t, ans = len(pts)-1, 0, 0.0
        for i in range(n):
            pi = pts[i]
            pj = pts[i+1] if i+1 <= n else pts[i]
            p = pj-pi
            while (t+1) % n != i:
                cross_1 = self.cross_produc_2d(p, pts[t+1] - pi)
                cross_2 = self.cross_produc_2d(p, pts[t] - pi)
                if self.compare_ab(cross_1, cross_2) == -1:
                    break
                t = (t+1) % n
            ans = max(ans, self.distance_normalized_2d(pi, pts[t]))
            ans = max(ans, self.distance_normalized_2d(pj, pts[t]))
        return ans

    def closest_pair_helper_2d(self, lo, hi):
        r_closest = (self.distance_2d(self.x_ordering[lo], self.x_ordering[lo+1]),
                     self.x_ordering[lo], 
                     self.x_ordering[lo+1])
        for i in range(lo, hi):
            for j in range(i+1, hi):
                distance_ij = self.distance_2d(self.x_ordering[i], 
                                               self.x_ordering[j])
                if self.compare_ab(distance_ij, r_closest):
                    r_closest = (distance_ij, self.x_ordering[i], self.x_ordering[j])
        return r_closest

    def closest_pair_recursive_2d(self, lo, hi, y_ordering):
        n = hi-lo
        if n < 4: # base case 
            return self.closest_pair_helper_2d(lo, hi)
        left_len, right_len = lo + n - n//2, lo + n//2
        mid = round((self.x_ordering[left_len].x + self.x_ordering[right_len].x)/2)
        y_part_left = [el for el in y_ordering if self.compare_ab(el.x, mid) <= 0]
        y_part_right = [el for el in y_ordering if self.compare_ab(el.x, mid) > 0]
        best_left = self.closest_pair_recursive_2d(lo, left_len, y_part_left)
        if self.compare_ab(best_left[0], 0.0) == 0:
            return best_left
        best_right = self.closest_pair_recursive_2d(left_len, hi, y_part_right)
        if self.compare_ab(best_right[0], 0.0) == 0:
            return best_right
        y_part_left = None
        y_part_right = None
        best_pair = best_left if self.compare_ab(best_left[0], best_right[0]) <= 0 else best_right
        y_check = [el for el in y_ordering if self.compare_ab((el.x - mid) * (el.x - mid), best_pair[0]) < 0]
        y_check_len = len(y_check)
        for i in range(y_check_len):
            for j in range(i+1, y_check_len):
                dist_ij = y_check[i].y - y_check[j].y
                if self.compare_ab(dist_ij * dist_ij, best_pair[0]) > 0:
                    break
                dist_ij = self.distance_2d(y_check[i], y_check[j])
                if self.compare_ab(dist_ij, best_pair) < 0:
                    best_pair = (dist_ij, y_check[i], y_check[j])
        return best_pair

    def compute_closest_pair_2d(self, pts):
        self.x_ordering = sorted(pts, key=lambda pt_xy: pt_xy.x)
        y_ordering = sorted(pts, key=lambda pt_xy: pt_xy.y)
        return self.closest_pair_recursive_2d(0, len(pts), y_ordering)

    def delaunay_triangulation_slow(self, pts):
        n = len(pts)
        ans = []
        z = [self.dot_product_2d(el, el) for el in pts]
        x = [el.x for el in pts]
        y = [el.y for el in pts]
        for i in range(n-2):
            for j in range(i + 1, n):
                for k in range(i + 1, n):
                    if j == k: continue
                    xn = (y[j]-y[i])*(z[k]-z[i]) - (y[k]-y[i])*(z[j]-z[i])
                    yn = (x[k]-x[i])*(z[j]-z[i]) - (x[j]-x[i])*(z[k]-z[i])
                    zn = (x[j]-x[i])*(y[k]-y[i]) - (x[k]-x[i])*(y[j]-y[i])
                    flag = zn < 0
                    for m in range(n):
                        if flag:
                            flag = flag and ((x[m]-x[i])*xn + 
                                             (y[m]-y[i])*yn + 
                                             (z[m]-z[i])*zn <= 0)
                        else:
                            break
                    if flag:
                        ans.append((i, j, k))
        return ans

    def pt_left_of_edge_2d(self, pt, edge):
        return 1 == self.point_c_rotation_wrt_line_ab_2d(pt, edge.origin, edge.dest())

    def pt_right_of_edge_2d(self, pt, edge):
        return -1 == self.point_c_rotation_wrt_line_ab_2d(pt, edge.origin, edge.dest())

    def det3_helper(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):
        return (a1 * (b2 * c3 - c2 * b3) - 
                a2 * (b1 * c3 - c1 * b3) + 
                a3 * (b1 * c2 - c1 * b2))

    def is_in_circle(self, a, b, c, d):
        a_dot = self.self.dot_product_2d(a, a)
        b_dot = self.self.dot_product_2d(b, b)
        c_dot = self.self.dot_product_2d(c, c)
        d_dot = self.self.dot_product_2d(d, d)
        det = -self.det3_helper(b.x, b.y, b_dot, c.x, c.y, c_dot, d.x, d.y, d_dot)
        det += self.det3_helper(a.x, a.y, a_dot, c.x, c.y, c_dot, d.x, d.y, d_dot)
        det -= self.det3_helper(a.x, a.y, a_dot, b.x, b.y, b_dot, d.x, d.y, d_dot)
        det += self.det3_helper(a.x, a.y, a_dot, b.x, b.y, b_dot, c.x, c.y, c_dot)
        return det > 0
        # use this if above doesn't work for what ever reason
        # def angle(l, mid, r):
        #     x = self.dot_product_2d(l-mid, r-mid)
        #     y = self.cross_product_2d(l-mid, r-mid)
        #     return atan2(x, y)
        # kek = angle(a, b, c) + angle(c, d, a) - angle(b, c, d) - angle(d, a, b)
        # return self.compare_ab(kek, 0.0) > 0

    def build_triangulation(l, r, pts):
        if r - l + 1 == 2:
            res = self.quad_edges.make_edge(pts[l], pts[r])
            return (res, res.rev())
        if r - l + 1 == 3:
            edge_a = self.quad_edges.make_edge(pts[l], pts[l + 1])
            edge_b = self.quad_edges.make_edge(pts[l + 1], pts[r])
            self.quad_edges.splce(edge_a.rev(), edge_b)
            sg = self.point_c_rotation_wrt_line_ab_2d(pts[l], pts[l + 1], pts[r])
            if sg == 0:
                return (edge_a, edge_b.rev())
            edge_c = self.quad_edges.connect(edge_b, edge_a)
            return (edge_a, edge_b.rev()) if sg == 1 else (edge_c.rev(), edge_c)
        mid = (l + r) // 2
        ldo, ldi = self.build_triangulation(l, mid, p)
        rdi, rdo = self.build_triangulation(mid + 1, r, p)
        while True:
            if self.pt_left_of_edge_2d(rdi.origin, ldi):
                ldi = ldi.l_next()
                continue
            if self.pt_right_of_edge_2d(ldi.origin, rdi):
                rdi = rdi.rev().o_next
                continue
            break
        base_edge_l = self.quad_edges.connect(rdi.rev(), ldi)
        if ldi.origin == ldo.origin:
            ldo = base_edge_l.rev()
        if rdi.origin == rlo.origin:
            rdo = base_edge_l
        while True:
            l_cand_edge = base_edge_l.rev().o_next
            if self.right_of(l_cand_edge.dest(), base_edge_l):
                while self.is_in_circle(base_edge_l.dest(), base_edge_l.origin, 
                                        l_cand_edge.dest(), l_cand_edge.o_next.dest()):
                    t = l_cand_edge.o_next
                    self.quad_edges.delete_edge(l_cand_edge)
                    l_cand_edge = t
            r_cand_edge = base_edge_l.o_prev()
            if self.right_of(r_cand_edge.dest(), base_edge_l):
                    while self.is_in_circle(base_edge_l.dest(), base_edge_l.origin, 
                                            r_cand_edge.dest(), r_cand_edge.o_prev().dest()):
                    t = r_cand_edge.o_prev()
                    self.quad_edges.delete_edge(r_cand_edge)
                    r_cand_edge = t
            l_check = self.right_of(l_cand_edge.dest(), base_edge_l)
            r_check = self.right_of(r_cand_edge.dest(), base_edge_l)
            if not l_check and not r_check:
                break
            if (not l_check or 
                    r_check and 
                    self.is_in_circle(l_cand_edge.dest(), l_cand_edge.origin, r_cand_edge.origin, r_cand_edge.dest())):
                base_edge_l = self.quad_edges.connect(r_cand_edge, base_edge_l.rev())
            else:
                base_edge_l = self.quad_edges.connect(base_edge_l.rev(), l_cand_edge.rev())
        return (ldo, rdo)        
            
    def delaunay_triangulation_fast(self, pts):
        pts.sort()
        result = self.build_triangulation(0, len(pts) - 1, p)
        edge = result[0]
        edges = [edge]
        while self.point_c_rotation_wrt_line_ab_2d(edge.o_next.dest(), edge.dest(), edge.origin) < 0:
            edge = edge.o_next
        def add_helper(pts, edge, edges):
            cur = edge
            while True:
                cur.used = True
                pts.append(cur.origin)
                edges.append(cur.rev())
                cur = cur.l_next()
                if cur == edge:
                    return
        add_helper(pts, edge, edges)
        pts = []
        kek = 0
        while kek < len(edges):
            edge = edges[kek]
            kek += 1
            if not edge.used:
                add_helper(pts, edge, edges)
        ans = [(pts[i], pts[i + 1], pts[i + 2]) for i in range(0, len(pts), 3)]
        return ans
        


def String_Algorithms:
    def __init__(self):
        self.n = 0
        self.text = ''

    def init_data(self, new_text):
        self.n = len(new_text):
        self.text = new_text

    def prepare_text_data(self, new_text):
        self.text = new_text
        self.text_len = len(new_text)

    def prepare_pattern_data(self, new_pattern):
        self.pattern = new_pattern
        self.pattern_len = len(new_pattern)

    def prepare_suffix_array_data(self, new_len):
        self.text_len = new_len
        self.counts_len = max(300, self.text_len)
        self.suffix_array = [i for i in range(self.text_len)]
        self.rank_arrary = [ord(self.text[i]) for i in range(self.text_len)]
        self.powers_of_2 = [2**i for i in range(32) if 2**i < self.text_len]

    def prepare_rolling_hash_data(self):
        self.prime_p = 131
        self.mod_m = 10**9 - 7
        self.powers = [0] * self.text_len
        self.h_vals = [0] * self.text_len
        self.math_algos = Math_Algorithms()

    def kmp_preprocess(self, target):
        self.prepare_pattern_data(target)
        self.back_table = [0] * (self.pattern_len + 1)
        self.back_table[0], j = -1, -1
        for i in range(self.pattern_len):
            while j >= 0 and self.pattern[i] != self.pattern[j]:
                j = self.back_table[j]
            j += 1
            self.back_table[i+1] = j

    def kmp_search_find_indices(self):
        ans = []
        j = 0
        for i in range(self.text_len):
            while j >= 0 and self.text[i] != self.pattern[j]:
                j = self.back_table[j]
            j += 1
            if j == self.pattern_len:
                ans.append(1 + i - j)
                j = self.back_table[j]
        return ans

    def suffix_array_preprocess(self, new_text):
        self.prepare_text_data(''.join([new_text, chr(0)*100010]))
        self.epare_suffix_array_data(len(new_text) + 1)

    def suffix_array_counting_sort(self, k):
        suffix_array_temp = [0] * self.text_len
        counts = [0] * self.counts_len
        ind = 0
        counts[0] = self.text_len - (self.text_len - k)
        for i in range(self.text_len - k):
            counts[self.rank_array[i + k]] += 1
        for i in range(self.counts_len):
            counts_i = counts[i]
            counts[i] = ind
            ind += counts_i
        for i in range(self.text_len):
            pos = 0
            if self.suffix_array[i] + k < self.text_len:
                pos = self.rank_array[self.suffix_array[i] + k]
            suffix_array_temp[counts[pos]] = self.suffix_array[i]
            counts[pos] += 1
        self.suffix_array = [el for el in suffix_array_temp]

    def suffix_array_build_array(self):
        for k in self.powers_of_2:
            self.suffix_array_counting_sort(k)
            self.suffix_array_counting_sort(0)
            rank_array_temp = [0] * self.text_len
            rank = 0
            for i in range(1, self.text_len):
                suffix_1 = self.suffix_array[i]
                suffix_2 = self.suffix_array[i - 1]
                r = r if (self.rank_array[suffix_1] == self.rank_array[suffix_2] and 
                          self.rank_array[suffix_1 + k] == self.rank_array[suffix_2 + k] else r + 1)
                rank_array_temp[suffix_1] = r
            self.rank_array = [el for el in rank_array_temp]

    def suffix_array_check_from_ind(self, ind):
        for i in range(self.pattern_len):
            if self.pattern[i] != self.text[ind + i]:
                return 1 if ord(self.text[ind + i]) > ord(self.pattern[i]) else -1
        return 0

    def suffix_array_binary_search(self, lo, hi, comp_val):
        while lo < hi:
            mid = (lo + hi)//2
            lo, hi = lo, mid if self.suffix_array_check_from_ind(self.suffix_array[mid]) > comp_val \
                            else mid + 1, hi
        return lo, hi

    def suffix_array_string_matching(self, new_pattern):
        self.prepare_text_data(new_pattern)
        lo, _ = self.suffix_array_binary_search(0, self.text_len - 1, -1)
        if self.suffix_array_check_from_ind(self.suffix_array[lo]) != 0:
            return (-1, -1)
        _, hi = self.suffix_array_binary_search(lo, self.text_len - 1, 0)
        if self.suffix_array_check_from_ind(self.suffix_array[hi]) != 0:
            hi -= 1
        return (lo, hi)

    def compute_longest_common_prefix(self):
        permuted_lcp = [0] * self.text_len
        phi = [0] * self.text_len
        l = 0
        phi[0] = -1
        for i in range(1, self.text_len):
            phi[self.suffix_array[i]] = self.suffix_array[i - 1]
        for i in range(self.text_len):
            if phi[i] == -1:
                permuted_lcp[i] = 0
                continue
            while (i + l < self.text_len and 
                   phi[i] + l < self.text_len and 
                   self.text[i + l] == self.text[phi[i] + l]):
                l += 1
            permuted_lcp[i] = l
            l = max(l - 1, 0)
        self.longest_common_prefix = [permuted_lcp[el] for el in self.suffix_array]

    def compute_longest_repeated_substring(self):
        ind, max_lcp = 0, -1
        for i in range(1, self.text_len):
            if self.longest_common_prefix[i] > max_lcp:
                ind, max_lcp = i, self.longest_common_prefix[i]
        return (max_lcp, ind)

    def owner(self, ind):
        return 1 if ind < self.text_len - self.pattern_len - 1 else 2

    def compute_longest_common_substring(self):
        ind, max_lcp = 0, -1
        for i in range(1, self.text_len):
            if (self.owner(self.suffix_array[i]) != self.owner(self.suffix_array[i - 1]) and
                self.longest_common_prefix[i] > max_lcp):
                ind, max_lcp = i, self.longest_common_prefix[i]
        return (max_lcp, ind)
        
    def compute_rolling_hash(self, new_text):
        self.prepare_text_data(new_text)
        self.prepare_rolling_hash_data()
        self.powers[0] = 1
        self.h_vals[0] = 0
        for i in range(1, self.text_len):
            self.powers[i] = (self.powers[i - 1] * self.prime_p) % self.mod_m
        for i in range(self.text_len):
            if i != 0:
                self.h_vals[i] = self.h_vals[i - 1]
            self.h_vals[i] = (self.h_vals[i] + (ord(self.text[i]) * self.powers[i]) % self.mod_m) % self.mod_m

    def hash_fast(self, l, r):
        if l == 0:
            return self.h_vals[r]
        ans = ((self.h_vals[r] - self.h_vals[l - 1]) % self.mod_m + self.mod_m) % self.mod_m
        ans = (ans * self.math_algos.mod_inverse(self.powers[l], self.mod_m)) % self.mod_m
        return ans

class Matrix:
    def __init__(self, n, m):
        self.num_rows = n
        self.num_cols = m
        selt.mat = [[0 for _ in range(m)] for _ in range(n)]

    def __mul__(self, multiplier):
        product = Matrix(self.num_rows, self.num_rows)
        for k in range(self.num_rows):
            for i in range(self.num_rows):
                if self.mat[i][k] != 0:
                    for j in range(self.num_rows):
                        product.mat[i][j] += (self.mat[i][k] * multiplier.mat[k][j])
        return product 

    def set_identity(self):
        for i in range(base.num_rows):
            for j in range(base.num_rows):
                self.mat[i][j] = 1 if i==j else 0

    def fill_matrix(self, new_matrix, a, b):
        for i in range(new_matrix.num_rows):
            for j in range(new_matrix.num_cols):
                self.mat[i + a][j + b] = new_matrix.mat[i][j]

    def get_augmented_matrix(self, matrix_b):
        augmented = Matrix(self.num_rows + matrix_b.num_rows, 
                           self.num_cols + matrix_b.num_cols)
        augmented.fill_matrix(self, 0, 0)
        augmented.fill_matrix(matrix_b, 0, self.num_cols)
        return augmented

    def get_determinant_matrix(self):
        det = Matrix(self.num_rows, self.num_cols)
        r = 1
        det.fill_matrix(self, 0, 0)
        for i in range(self.num_rows):
            for j in range(self.num_rows):
                while determinant.mat[j][i] != 0:
                    ratio = det.mat[i][i] / det.mat[j][i]
                    for k in range(i, self.num_rows):
                        det.mat[i][k] -= (ratio * det.mat[j][k])
                        det.mat[i][k], det.mat[j][k] = det.mat[j][k], det.mat[i][k]
                    r = -r
            r = r * det[i][i]
        return r

class Matrix_Algorithhms:
    def __init__(self):
        self.matrix_A = []
        self.matrix_b = []
        self.matrix_x = []
        self.num_rows = 0
        self.num_cols = 0

    def matrix_exponentiation(self, base, power):
        result = Matrix(base.num_rows, base.num_rows)
        result.set_identity()
        while power:
            if power%2 == 1:
                result = result * base
            base = base * base
            power //= 2
        return result
    
    def init_matrices(self):
        self.matrix_A = [[0 for _ in range(self.num_rows)] for _ in range(self.num_rows)] 
        self.matrix_b = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)] 
    
    def init_constants(self, n, m):
        self.num_rows = n
        self.num_cols = m

    def init_data(self, n, m):
        self.init_constants(n, m)

    def gauss_elimination(self, aug_Ab):
        n = aug_Ab.num_rows-1
        for i in range(n):
            tmp, pos = 0.0, -1
            for j in range(i, n):
                if abs(aug_Ab.mat[j][i]) > tmp):
                    tmp, pos = abs(aug_Ab.mat[j][i]), j
            if pos != -1:
                for k in range(n + 1):
                    aug_Ab.mat[pos][k], aug_Ab.mat[i][k] = aug_Ab.mat[i][k], aug_Ab.mat[pos][k]
                tmp = aug_Ab.mat[i][i]
                for k in range(n + 1):
                    aug_Ab.mat[i][k] /= tmp
                for j in range(i + 1, n):
                    tmp = aug_Ab.mat[j][i]
                    for k in range(n + 1):
                        aug_Ab.mat[j][k] -= (tmp * aug_Ab.mat[i][k])
        for i in range(n, -1, -1):
            for j in range(i):
                aug_Ab.mat[j][n] -= (aug_Ab.mat[i][n] * aug_Ab.mat[j][i])
                aug_Ab.mat[j][i] = 0
                
    

            
    
        
    
    


