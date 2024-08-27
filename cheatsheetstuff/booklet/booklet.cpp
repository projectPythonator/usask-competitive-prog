#include <bits/stdic++.h> // This include is mostly for catch all and speed

typedef vector<int> vec_int;  // TBD if we go with typedefs

//#include <vector>		// for vec_int
//#include <algorithm>	// for iota, swap

class UnionFindDisjointSets {
private:
	vec_int parent, rank, setSizes;
	int numSets;
public:
	UnionFindDisjointSets(int n) {
		// Attributes declared here must be passed in or global if not used in classes
		parent.assign(n, 0);
		numSets = n;						// optional information
		setSizes.assign(n, 1);				// optional information
		rank.assign(n, 0);					// optional optimization 
		iota(rank.begin(), rank.end(), 0);	// rank = {0, 1, 2...}
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
};


/// <summary>
/// ///////////////////////////////////////////////////////////////////
/// </summary>

typedef vector<bool> vec_bool;
typedef vector<int> vec_int;
class MathAlgorithms {
private:
	xy;
	abs;
	vec_int primesList;
public:
	void sieveOfEratosthenes(int nInclusive) {
		/*Generates list of primes up to n via eratosthenes method.

		Complexity: Time: O(n lnln(n)), Space: post call O(n/ln(n)), mid-call O(n)
		Variants: number and sum of prime factors, of diff prime factors, of divisors, and phi
		*/
		int limit = (int)sqrt(nInclusive);
		vec_int primeSieve = vec_bool(++nInclusive, True);
		for (int prime = 2; prime < limit; ++prime)
			if (primeSieve[prime])
				for (int composite = prime * prime; composite < nInclusive; composite += prime)
					primeSieve[composite] = false;
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
		vector<char> blockSieve(sqrtBlock + 1, true); // apparently char was faster than bool
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
				i[1] = idx - sqrtBlock;	//  this line resets the block to the maintain module
			}
			if (low == 0)	// small corner case we need to handle
				blockSieve[0] = false; 
			for (int i = 0; i < sqrtBlock && low+i <= high; i++) // fill primesList up 
				if (blockSieve[i])
					primesList.push_back((low + i) * 2 + 1)
		}
	}
};