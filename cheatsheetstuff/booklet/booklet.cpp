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
	vec_int primesList, minPrimes;
	vec_int numDiv, sumDiv;
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
				i[1] = idx - sqrtBlock;	//  this line resets the block to the maintain module
			}
			if (low == 0)	// small corner case we need to handle
				blockSieve[0] = false; 
			for (int i = 0; i < sqrtBlock && low+i <= high; i++) // fill primesList up 
				if (blockSieve[i])
					primesList.push_back((low + i) * 2 + 1)
		}
	}


	void sieveOfMinPrimes(int nInclusive) {  
		/*Stores the min or max prime divisor for each number up to n.

			Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space : post call O(n)
		*/
		minPrimes.assign(nInclusive + 1, 0);
		iota(minPrimes.begin(), minPrimes.end(), 0);
		for (int two = 2; two <= nInclusive; two += 2)
			minPrimes[two] = 2;
		for (int prime = 3; prime * prime <= nInclusive; prime += 2)
			if (minPrimes[prime] == prime)  // we found a prime
				for (int composite = prime * prime; composite <= nInclusive; composite += (2 * prime))
					if (minPrimes[composite] == composite) // min not set yet
						minPrimes[composite] = prime;
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
		limit++;
		numDiv.assign(limit, 1);
		sumDiv.assign(limit, 1);
		for (int divisor = 2; divisor < limit; divisor++)
			for (int multiple = divisor; multiple < limit; multiple += divisor) {
				numDiv[multiple]++;
				sumDiv[multiple] += divisor;
			}
	}

	void eulerPhiPlusSumAndNumOfDiffPrimeFactors(int limit) {
		// This is basically same as sieve just using different ops. Complexity function 2.
		limit++;
		numDiffPF.assign(limit, 0);
		sumDiffPF.assign(limit, 0);
		phi.assign(limit, 0);
		iota(phi.begin(), phi.end(), 0);
		for (int prime = 2; prime < limit; prime++)
			if (numDiffPF[prime] == 0)
				for (int multiple = prime; multiple < limit; multiple += prime) {
					numDiffPF[multiple]++;
					sumDiffPF[multiple] += prime;
					phi[multiple] = (phi[multiple] / prime) * (prime - 1);
				}
	}

	void numAndSumOfPrimeFactors(int limit) {
		// This uses similar idea to sieve but avoids divisions. Complexity function 3.
		inclusiveLimit++;
		numPF.assign(inclusiveLimit, 0);
		sumPF.assign(inclusiveLimit, 0);
		for (int prime = 2; prime < inclusiveLimit; prime++)
			if (numDiffPF[prime] == 0) {
				int exponentLimit = lrint(log(limit) / log(prime)) + 2;
				for (int exponent = primeToPowerN = 1; exponent < exponentLimit; exponent++) {
					primeToPowerN *= prime;
					for (int multiple = primeToPowerN; multiple < inclusiveLimit; multiple += primeToPowerN) {
						numPF[multiple]++;
						sumPF[multiple] += prime;
					}
				}
			}
	}


	void numAndSumOfDivisorsFaster(int limit) {
		// This uses similar idea to sieve but avoids divisions. Complexity function 3.
		inclusiveLimit++;
		numDiv.assign(inclusiveLimit, 1);
		sumDiv.assign(inclusiveLimit, 1);
		curPow.assign(inclusiveLimit, 1);
		primePowers.assign(32, 0); // use this 
		for (int prime = 2; prime < inclusiveLimit; prime++)
			if (numDiv[prime] == 1) {
				int exponentLimit = lrint(log(limit) / log(prime)) + 2;
				for (int exponent = primeToPowerN = 1; exponent < exponentLimit; exponent++) {
					primeToPowerN *= prime;
					primePowers[exponent] = primeToPowerN;
					for (int multiple = primeToPowerN; multiple < inclusiveLimit; multiple += primeToPowerN)
						curPow[multiple]++;
				}
				int tmp = prime - 1;
				for (int multiple = prime; multiple < inclusiveLimit; multiple += prime) {
					numDiv[multiple] *= curPow[multple];
					sumDiv[multiple] *= ((primePowers[curPow[multiple]] - 1) / tmp);
					curPow[multiple] = 1;
				}
			}
	}
};