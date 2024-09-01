// GLOBALS used throughout the booklet
#include <bits/stdc++.h> // This include is mostly for catch all and speed

using namespace std;	// okay for contests not for real life

// GLOBAL redefines
typedef int									int32;	// careful to ensure its 32bits
typedef unsigned int				uint32;	// careful to ensure its 32bits
typedef long long						int64;
typedef unsigned long long	uint64;
typedef __uint128_t					uint128;
typedef string							dataType; // required to be hashable sometimes for unordered_map

// mostly used in the graph functions
typedef tuple<int32, int32> int32_Pair;
typedef tuple<int32, int32, int32> int32_Triple;
typedef tuple<int32, int32, int32, int32> int32_Quadruple;

typedef vector<int32> int32_Vec;  
typedef vector<int64> int64_Vec;
typedef vector<int32_Pair> int32_Pair_Vec;
typedef vector<int32_Triple> int32_Triple_Vec;
typedef vector<int32_Quadruple> int32_Quadruple_Vec;

typedef unordered_map<dataType, int32>	DAT_type_to; // Direct access table helper.
typedef vector<dataType>								DAT_type_from; // Direct access table helper.




//////////////////////////////////////////////////////////////////////////////////////////////////////////
//#include <vector>		// for vec_int
//#include <algorithm>	// for iota, swap

class UnionFindDisjointSets {
private:
	int32_Vec parent, rank, setSizes;
	int32 numSets;
public:
	UnionFindDisjointSets(int32 n) {
		// Attributes declared here must be passed in or global if not used in classes
		parent.assign(n, 0);
		setSizes.assign(n, 1);							// optional information
		rank.assign(n, 0);									// optional optimization 
		iota(rank.begin(), rank.end(), 0);	// rank = {0, 1, 2...}
		numSets = n;												// optional information
	}

	int findSet(int u) {
		/*Recursively find which set u belongs to. Memoize on the way back up.

		Complexity: Time: O(\alpha(n)) -> O(1), inverse ackerman practically constant
				   Space: Amortized O(1) stack space
		*/
		return (parent[i] == i) ? i : (parent[i] = findSet(parent[i]));
	}

	bool isSameSet(int u, int v) {
		// Checks if u and v in same set. TIME and SPACE Complexity is the same as findSet
		return findSet(u) == findSet(v);
	}

	void unionSet(int u, int v) {
		/* Join the set that contains u with the set that contains v.

		Complexity: Time: O(\alpha(n)) -> O(1), inverse ackerman practically constant
				   Space: Amortized O(1) stack space
		*/
		if (!isSameSet(u, v)) {
			int uParent = findSet(u), vParent = findSet(v);
			if (rank[uParent] > rank[vParent])	// uParent shorter than vParent
				swap(uParent, vParent);
			if (rank[uParent] == rank[vParent]) // optional speedup
				parent[vParent]++;
			parent[uParent] = vParent;				// line that joins u and v
			setsize[vParent] += setSizes[uParent];	// u = v so add join the size
			numSets--;
		}
	}

	int sizeOfSet(int u) {
		// Gives you the size of set u. TIME and SPACE Complexity is the same as find_set
		return setSizes[findSet(u);
	}

	int getNumSets() {
		return numSets;
	}
};

/// <summary>
/// ///////////////////////////////////////////////////////////////////
/// </summary>

typedef pair<int, int> adj_pair;
typedef vector<adj_pair> adj_edge_list_wt;
typedef vector<adj_edge_list_wt> adj_list_wt;
typedef vector<int> adj_edge_list;
typedef vector<adj_edge_list> adj_list;
typedef vector<int> mat_row;
typedef vector<adj_row> adj_mat;
typedef pair<int, int> edge_a_b;
typedef pair<int, edge_a_b> edge_weighted;
typedef vector<edge_weighted> edge_list;
typedef unordered_map<int, int> data_map;	// can change left side to whatever you need
typedef vector<int> index_map;				// type must match leftside type above
typedef vector<int> grid_row;
typedef vector<grid_row> grid_graph;


class Graph {
private:
	int numEdges, numNodes, numRows, numCols;
	adj_list adjList, adjListTrans;
	adj_list_wt adjListWt, adjListTransWt
	adj_mat adjMatrix;
	edge_list = edgeList;	// can also make this into a heap
	data_map dataToIndex;
	index_map IndexToData;
	grid_graph grid;
public:
	int convertDataExample(int data) {
		/*Converts data to the form: int u | 0 <= u < |V|, stores (data, u) pair.*/
		if (!dataToIndex.contains(data) {
			dataToIndex[data] = indexToData.size();
			indexToData.push_back(data);
		}
		return dataToIndex[data];
	}

	void addEdgeUVWExample(int u, int v, int wt, int data) {
		/*A pick and choose function will convert u, v into index form then add it to the
		structure you need.
		*/
		int node_u = convertDataExample(u);
		int node_v = convertDataExample(v);

		adjList[node_u].push_back(node_v);
		//adjListWt[node_u].push_back({ node_v, wt });	// use this for weighted
		adjMatrix[node_u][node_v] = wt;					// Adjacency matrix usage
		edgeList.push_back({ wt, {node_u, node_v} });	// edge list usage
		edgeList.push_back({ not_v, {wt, data} });		// this one is used for flow
		adjList[node_u].push_back(edgeList.size() - 1));
	}

	void addEdgeUndirected(int u, int v, int wt, int data) {
		/*undirected graph version of the previous function. wt can be omitted.*/
		addEdgeUVWExample(u, v, wt, data);
		addEdgeUVWExample(v, u, wt, data);
	}

	void fillGridGraph(grid_graph newGrid) {
		numRows = newGrid.size();
		numCols = newGrid[0].size();
		copy(newGrid.begin(), newGrid.end(), back_inserter(grid));
	}
};

const int INF = 1 >> 31; // is this safe ?

class GraphAlgorithms {
private:
	vector<pair<int, int>> dirRC = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };
	Graph graph;
	edge_list minSpanningTree;
public:
	void floodFillViaDFS(int row, int col, int oldVal, int newVal) {
		/*Computes flood fill graph traversal via recursive depth first search.
		* 
		* Complexity: Time: O(|V| + |E|), Space: O(|V|): for grids |V|=row*col and |E|=4*|V|
		* More uses: Region Colouring, Connectivity, Area/island Size, misc
		* Input:
		*  row, col: integer pair representing current grid position
		*  old_val, new_val: unexplored state, the value of an explored state
		*/
		graph.grid[row][col] = newVal;
		for (const auto& [rowMod, colMod] : dirRC) {
			int newRow = row + rowMod, newCol = col + colMod;
			if (0 <= newRow && newRow < graph.numRows &&
				0 <= newCol && newCol < graph.numCols &&
				graph.grid[newRow][newCol] == oldVal)
				floodFillViaDFS(newRow, newCol, oldVal, newVal);
		}
	}

	void floodFillViaBFS(int startRow, int startCol, int oldVal, int newVal) {
		/* Computes flood fill graph traversal via breadth first search. Use on grid graphs.
		* 
		* Complexity: Time: O(|V| + |E|), Space: O(|V|): for grids |V|=row*col and |E|=4*|V|
		* More uses: previous uses tplus shortest connected pah
		*/
		deque<pair<int, int>> q = { {startRow, startCol} };	// q to avoid name conflicts
		while (!q.empty()) {
			const auto& [newRow, newCol] = q.back(); q.pop_back();
			for (const auto& [rowMod, colMod] : dirRC) {
				int newRow = row + rowMod, newCol = col + colMod;
				if (0 <= newRow && newRow < graph.numRows &&
					0 <= newCol && newCol < graph.numCols &&
					graph.grid[newRow][newCol] == oldVal) {
					graph.grid[newRow][newCol] = newVal;
					q.push_front({ newRow, newCol });
		} } } // 3 brackets closes up to the loop
	}

	void minSpanningTreeViaKruskalsWithHeaps() {
		/*Computes mst of graph G stored in edge_list, space optimized via heap.
		* 
		* Complexity per call: Time: O(|E| log |V|), Space: O(|E| log |E|) + Union_Find
		* More uses: finding min spanning tree
		* Variants: min spanning subgraph and forrest, max spanning tree, 2nd min best spanning
		* Optimization: We use a heap to make space comp. O(|E|). instead of O(|E|log |E|)
		* when using sort, however edge_list is CONSUMED. Also uses space optimization
		*/
		int vertices = graph.numNodes;
		sort(graph.edgeList.begin(), graph.edgeList.end());  // optimization line here
		auto UFDS = UnionFindDisjointSets(vertices);
		for (const auto& [wt, [u, v]] : adj_edge_list) { // need to pop heap in loop
			if (UFDS.getNumSets() == 1) break;
			if (!UFDS.isSameSet(u, v)) {
				minSpanningTree.push_back({ wt, {u, v} });
				UFDS.unionSet(u, v);
		} } //  2 brack closed on loop
	}

	void primsViaAdjMatrix(int source) {
		/*Find min weight edge in adjacency matrix implementation of prims.
		* 
		* Complexity per call: Time: O(|V|^2), T(|V| * 4|V|), Space: O(|V|), S(~5|V|)
		*/
		int vertices = graph.numNodes;

	}
};


/// <summary>
/// ///////////////////////////////////////////////////////////////////
/// </summary>

typedef vector<bool> vec_bool;
typedef vector<int> vec_int32;
typedef vector<long long> vec_int64;

class MathAlgorithms {
private:
	vec_int32 primesList, minPrimes;
	vec_int32 numDiv, numPF, numDiffPF;
	vec_int32 numFactorialPF;
	vec_int64 sumDiv, sumPF, sumDiffPF;

public:
	void sieveOfEratosthenes(int nInclusive) {
		/*Generates list of primes up to n via eratosthenes method.

		Complexity: Time: O(n lnln(n)), Space: post call O(n/ln(n)), mid-call O(n)
		Variants: number and sum of prime factors, of diff prime factors, of divisors, and phi
		*/
		int limit = (int)sqrt(nInclusive);
		vec_bool primeSieve = vec_bool(++nInclusive, True);
		for (int prime = 2; prime < limit; ++prime)
			if (primeSieve[prime])
				for (int multiple = prime * prime; multiple < nInclusive; multiple += prime)
					primeSieve[multiple] = false;
		for (int prime = 2; prime < nInclusive; ++prime)
			if (primeSieve[prime])
				primesList.push_back(prime);
	}

	void primeSeiveFaster(int limit) {
		/*Block sieve that builds up block by block to the correct amount needed.

		Complexity: Time: O(max(n lnln(sqrt(n)), n)),
				   Space: post call O(n / ln(n)), mid - call O(sqrt(n))
		*/
		const int sqrtBlock = round(sqrt(limit)); // block size in this version we use sqrt(n)
		const int high = (limit - 1) / 2;	
		vector<char> blockSieve(sqrtBlock + 1, true); // apparently char was faster than bool?
		vector<array<int, 2>> prime_and_blockStart;	  // holds prime, block start: pair
		for (int i = 3; i < sqrtBlock; i += 2) { // fast pre-computation up to sqrt(n)
			if (blockSieve[i]) {
				prime_and_blockStart.push_back({i, (i*i-1) / 2});
				for (int j = i*i; j <= sqrtBlock; j += 2*i)
					blockSieve[j] = false;
			}
		}
		blockSieve.pop_back();	// blockSieve needs to be sqrt(n) for the next section
		for (int low = 0; low <= high; low += sqrtBlock) {		// here we fill the primes
			fill(blockSieve.begin(), blockSieve.end(), true);	// list in blocks of size
			for (auto& prime_and_beg : prime_and_blockStart) {	// sqrt(n)
				int prime = prime_and_beg[0], idx = prime_and_beg[1];
				for (; idx < sqrtBlock; idx += prime)
					blockSieve[idx] = false;
				prime_and_beg[1] = idx - sqrtBlock;	//  this line resets the block to the maintain module
			}
			if (low == 0)	// small corner case we need to handle
				blockSieve[0] = false; 
			for (int i = 0; i < sqrtBlock && low + i <= high; i++) // fill primesList up 
				if (blockSieve[i])
					primesList.push_back((low + i) * 2 + 1);
		}
	}


	void sieveOfMinPrimes(int limit) {  
		/*Stores the min or max prime divisor for each number up to n.

			Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space : post call O(n)
		*/
		minPrimes.assign(limit + 1, 0);
		iota(minPrimes.begin(), minPrimes.end(), 0);
		for (int two = 2; two <= limit; two += 2)
			minPrimes[two] = 2;
		for (int prime = 3; prime * prime <= limit; prime += 2)
			if (minPrimes[prime] == prime)  // we found a prime
				for (int multiple = prime * prime; multiple <= limit; multiple += (2 * prime))
					if (minPrimes[multiple] == multiple) // min not set yet
						minPrimes[multiple] = prime;
	}
	void sieveOfEratosthenesVariants(int n_inclusive) {
		/*Seven variants of prime sieve listed above.

			Complexity :
				function 1 : Time : O(n log(n)), Space : O(n)
				function 2 : Time : O(n lnln(n)), Space : O(n)
				function 3 : Time : O(n lnln(n) log(n)), Space : O(n)
				function 4 : Time : O(n lnln(n) log(n)), Space : O(n)
			*/
		euler_phi_plus_sum_and_number_of_diff_prime_factors(n_inclusive);
		num_and_sum_of_divisors(n_inclusive);
		num_and_sum_of_prime_factors(n_inclusive);
	}

	void numAndSumOfDivisors(int limit) {
		//  Does a basic sieve. Complexity function 1.
		numDiv.assign(limit + 1, 1);
		sumDiv.assign(limit + 1, 1ll);  // likely needs to be long long
		for (int divisor = 2; divisor <= limit; divisor++)
			for (int multiple = divisor; multiple <= limit; multiple += divisor) {
				numDiv[multiple]++;
				sumDiv[multiple] += divisor;
			}
	}

	void eulerPhiPlusSumAndNumOfDiffPrimeFactors(int limit) {
		// This is basically same as sieve just using different ops. Complexity function 2.
		numDiffPF.assign(limit + 1, 0);
		sumDiffPF.assign(limit + 1, 0ll);	// likely needs to be long long
		phi.assign(limit + 1, 0ll);			// likely needs to be long long
		iota(phi.begin(), phi.end(), 0);
		for (int prime = 2; prime <= limit; prime++)
			if (numDiffPF[prime] == 0)
				for (int multiple = prime; multiple <= limit; multiple += prime) {
					numDiffPF[multiple]++;
					sumDiffPF[multiple] += prime;
					phi[multiple] = (phi[multiple] / prime) * (prime - 1);
				}
	}

	void numAndSumOfPrimeFactors(int limit) {
		// This uses similar idea to sieve but avoids divisions. Complexity function 3.
		numPF.assign(limit + 1, 0);
		sumPF.assign(limit + 1, 0ll);	// likely needs to be long long
		for (int prime = 2; prime <= limit; prime++)
			if (numDiffPF[prime] == 0) {
				int exponentLimit = 0;	// p^n | p^n <= limit < p^(n+1)
				long long primePows[32];	// 32 limits us to 2^31 (2^0 == 1)
				for (long long primeToN = 1; primeToN <= limit; primeToN *= prime)
					primePows[exponentLimit++] = primeToN;
				for (int exponent = 1; exponent < exponentLimit; exponent++) {
					int primeToN = primePows[exponent];
					for (int multiple = primeToN; multiple <= limit; multiple += primeToN) {
						numPF[multiple]++;
						sumPF[multiple] += prime;
			} } } // 3 closing brackets
	}


	void numAndSumOfDivisorsFaster(int limit) {
		// This uses similar idea to sieve but avoids divisions. Complexity function 4
		// sumDiv.assign(limit + 1, 1ll);  // likely needs to be long long
		numDiv.assign(limit + 1, 1);
		curPow.assign(limit + 1, 1); // here a
		for (int prime = 2; prime <= limit; prime++)
			if (numDiv[prime] == 1) {
				int exponentLimit = 0; long long primePows[32];
				for (long long primeToExp = 1; primeToExp <= limit; primeToExp *= prime)
					primePows[exponentLimit++] = primeToExp;
				// primePows[exponentLimit] = primePows[exponentLimit-1] * prime; / use if calculating sumDiv
				for (int exponent = 1; exponent < exponentLimit; exponent++)
					for (int mul = primePows[exponent], primeToN = primePows[exponent]; mul <= limit; mul += primeToN)
						curPow[mul]++;
				for (int multiple = prime; multiple <= limit; multiple += prime) {
					// sumDiv[multiple] *= ((primePows[curPow[multiple]] - 1) / (prime - 1));
					numDiv[multiple] *= curPow[multiple];
					curPow[multiple] = 1; // needs to happen regardless of version you are using
				}
			}
	}

	bool isPrimeTrivial(int n) {
		/*Tests if n is prime via divisors up to sqrt(n).

		Complexity per call: Time: O(sqrt(n)), T(sqrt(n)/3), Space: O(1)
		Optimizations: 6k+i method, since we checked 2 and 3 only need test form 6k+1 and 6k+5
		*/
		if (n < 4) return n > 1;					// base case of n in 0, 1, 2, 3
		if (n % 2 == 0 || n % 3 == 0) return false; // this is what allows us to use 6k + i
		int limit = lrint(sqrt(n)) + 1;
		for (int prime = 5; prime < limit; prime += 6)
			if (n % prime == 0 || n % (prime + 2) == 0)
				return false;
		return true;
	}

	uint64 mrptPowMod(uint64 base, uint64 exp, uint64 mod) {
		/* binary exponentiation 
		* taken from https://cp-algorithms.com
		*/
		uint64 result = 1;
		base %= mod;
		while (exp) {
			if (exp & 1)
				result = (uint128)result * base % mod;
			base = (uint128)base * base % mod;
			exp >>= 1;
		}
		return result;
	}

	bool mrptCompositeCheck(uint64 n, uint64 a uint64 d, int s) {
		/*The witness test of miller rabin.
		taken from https://cp-algorithms.com
		Complexity per call: Time O(log^3(n)), Space: O(1)
		*/
		uint64 x = mrptPowMod(a, d, n);
		if (x == 1 || x == n - 1) return false;
		for (int r = 1; r < s; ++r) {
			x = (uint128)x * x % n;
			if (x == n - 1)
				return false;
		}
		return true;
	}

	bool isPrimeMRPT(uint64 n) {
		/*Handles all numbers up to 2^64-1 (maybe need to test it to gain confidence).
		taken from https://cp-algorithms.com
		Complexity per call: Time O(12 log^3(n)), Space: O(log2(n)) bits
		Optimizations: test first 12 primes before running the algorithm or generate sieve.
		*/
		if (n < 2) return false;
		int r = 0;
		uint64 d = n - 1;
		for (; (d & 1) == 0; d >>= 1, r++) {}	// turned whileloop into forloop

		for (int a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
			if (n == a || mrptCompositeCheck(n, a, d, r))
				return n == a;
		return true;
	}

	vec_int32 primeFactorizeN(int n) {
		/*A basic prime factorization of n function. without primes its just O(sqrt(n))
		* 
		* Complexity: Time: O(sqrt(n)/ln(sqrt(n))), Space: O(log n)
		* Variants: number and sum of prime factors, of diff prime factors, of divisors, and phi
		*/
		vec_int32 factors;
		for (auto& prime : primesList) {
			if (prime * prime > n) break;
			for (; n % prime == 0; n /= prime)
				factors.push_back(prime);
		}
		if (n > 1)
			factors.push_back(n);
		return factors;
	}

	vec_int32 primeFactorizeNLogN(int n) {
		/*An optimized prime factorization of n function based on min primes already sieved.
		*
		* Complexity: Time: O(log n), Space: O(log n)
		* Optimization: assign append to function and assign minPrimes[n] to a value in the loop
		*/
		vec_int32 factors;
		for (; n > 1; n /= minPrimes[n])
			factors.push_back(minPrimes[n]);
		return factors;
	}

	int primeFactorizeNVariants(int n) {
		/*Covers all the variants listed above, holds the same time and space complexity*/
		uint64 sumDiffPFV = 0, sumPFV = 0, sumDivV = 1, eulerPhi = n;
		int numDiffPFV = 0, numPFV = 0, numDivV = 1;
		for (auto& prime : primesList) {  // for(int prime = 2; prime*prime <= n; prime++)
			if (prime * prime > n) break;
			if (n % prime == 0) {
				uint64 total = 1;	// for sum of divisors
				int power = 0;		// for num of divisors
				for (uint64 mul = prime; n % prime == 0; n /= prime) {
					power++;	// for num prime factors, num divisors, and sum prime factors
					total += mul;	// for sum divisors
					mul *= prime;	// for sum divisors
				}
				sumDiffPFV += prime;
				numDiffPFV++;
				sumPFV += (prime * power);
				numPFV += power;
				numDivV *= (power + 1);
				sumDivV *= total;
				eulerPhi -= (eulerPhi / prime);
			}
		}
		if (n > 1) {
			sumDiffPFV += n;
			numDiffPFV++;
			sumPFV += n;
			numPFV++;
			numDivV *= 2;
			sumDivV *= (n + 1);
			eulerPhi -= (eulerPhi / n);
		}
		return numDiffPFV;
	}

	void factorialPrimeFactors(int limit) {
		/* NHI I lost this or were I got the idea but it works for factorizing a factorial
		* 
		* Complexity: Time: O(n log n)), Space: O(n)
		*/
		int endPoint = upper_bound(primesList.begin(), primesList.end(), limit);
		endPoint = endPoint - primesList.begin();
		numFactorialPF.assign(endPoint, 0);
		for (int idx = 0; idx < endPoint; ++idx) {
			int prime = primesList[idx];
			uint64 primeAmt = 0;
			for (int x = limit; x; primeAmt += x)
				x /= prime;
			numFactorialPF[idx] = primeAmt;
		}
	}

	uint64 rhoMulMod(uint64 a, uint64 b, uint64 mod) {
		return (uint128)a * b % mod;
	}

	uint64 rhoF(uint64 x, uint64 c, uint64 mod) {
		return (rhoMulMod(x, x, mod) + c) % mod;
	}

	uint64 rhoPollard(uint64 n, uint64 x0 = 2, uint64 c = 1) {
		uint64x x = x0, y = x0, g = 1;
		while (g == 1) {
			x = rhoF(x, c, n);
			y = rhoF(y, c, n);
			y = rhoF(y, c, n);
			g = gcd(abs(x - y), n);
		}
		return g;
	}

};