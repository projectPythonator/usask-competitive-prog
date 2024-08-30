#include <bits/stdc++.h> // This include is mostly for catch all and speed

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
typedef vector<int> vec_int32;
typedef vector<long long> vec_int64;
typedef uint64_t uint64;
typedef __uint128_t uint128;

class MathAlgorithms {
private:
	vec_int32 primesList, minPrimes;
	vec_int32 numDiv, numPF, numDiffPF;
	vec_int64 sumDiv, sumPF, sumDiffPF

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
};