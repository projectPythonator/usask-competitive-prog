/// <summary>
/// ///////////////////////////////////////////////////////////////////
/// </summary>
/// 

class MathAlgorithms {
private:
	int32_Vec primesList, minPrimes;
	int32_Vec numDiv, numPF, numDiffPF;
	int32_Vec numFactorialPF;
	int64_Vec sumDiv, sumPF, sumDiffPF;

public:
	//Generates list of primes up to n via eratosthenes method.
	//
	//	Complexity: Time: O(n lnln(n)), Space: post call O(n/ln(n)), mid-call O(n)
	//	Variants: number and sum of prime factors, of diff prime factors, of divisors, and phi
	void sieveOfEratosthenes(int32 limit) {
		int32 sqrtLimit = lrint(sqrt(limit)) + 1;
		bool_vec primeSieve = bool_vec(limit + 1, True);
		for (int prime = 2; prime < sqrtLimit; ++prime)
			if (primeSieve[prime])
				for (int multiple = prime * prime; multiple <= limit; multiple += prime)
					primeSieve[multiple] = false;
		for (int prime = 2; prime <= limit; ++prime)
			if (primeSieve[prime])
				primesList.push_back(prime);
	}

	// Block sieve that builds up block by block to the correct amount needed.
	//
	//	Complexity: Time: O(max(n lnln(sqrt(n)), n)),
	//		    Space: post call O(n / ln(n)), mid - call O(sqrt(n))
	void primeSieveFaster(int32 limit) {
		const int sqrtBlock = round(sqrt(limit)) + 1; // block size + 1 for safety :)
		const int high = (limit - 1) / 2;	
		char_vec blockSieve(sqrtBlock + 1, true);		// apparently char was faster than bool?
		int32_Pair_Vec prime_and_blockStart;				// holds prime, block start: pair
		for (int32 i = 3; i < sqrtBlock; i += 2) { // fast pre-computation up to sqrt(n)
			if (blockSieve[i]) {
				prime_and_blockStart.push_back({i, (i*i-1) / 2});
				for (int32 j = i*i; j <= sqrtBlock; j += 2*i)
					blockSieve[j] = false;
		} } // 2 brackets closes the forloop
		blockSieve.pop_back();	// blockSieve needs to be sqrt(n) for the next section
		for (int32 low = 0; low <= high; low += sqrtBlock) {		// here we fill the primes
			fill(blockSieve.begin(), blockSieve.end(), true);	// list in blocks of size
			for (auto& prime_idx : prime_and_blockStart) {	// prime_idx is a tuple pair
				auto [prime, idx] = prime_idx;					// need to define the pair here since it
				for (; idx < sqrtBlock; idx += prime)		// is a lot slower to modify the tuple
					blockSieve[idx] = false;
				get<1>(prime_idx) = idx - sqrtBlock;
			}
			if (0 == low)	// small corner case we need to handle
				blockSieve[0] = false; 
			for (int32 i = 0; i < sqrtBlock && low + i <= high; i++) // fill primesList up 
				if (blockSieve[i])
					primesList.push_back((low + i) * 2 + 1);
		}
	}

	// Stores the min or max prime divisor for each number up to n.
	//
	//	Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space : post call O(n)
	void sieveOfMinPrimes(int32 limit) {
		minPrimes.assign(limit + 1, 0);
		iota(minPrimes.begin(), minPrimes.end(), 0);
		for (int32 two = 2; two <= limit; two += 2)
			minPrimes[two] = 2;
		for (int32 prime = 3; prime * prime <= limit; prime += 2)
			if (minPrimes[prime] == prime)  // improvised prime bool check
				for (int32 multiple = prime * prime; multiple <= limit; multiple += (2 * prime))
					if (minPrimes[multiple] == multiple) // min not set this was than not checking
						minPrimes[multiple] = prime;
	}

	// Seven variants of prime sieve listed above.
	//
	//	Complexity :
	//	function 1 : Time : O(n log(n)), Space : O(n)
	//	function 2 : Time : O(n lnln(n)), Space : O(n)
	//	function 3 : Time : O(n lnln(n) log(n)), Space : O(n)
	//	function 4 : Time : O(n lnln(n) log(n)), Space : O(n)
	void sieveOfEratosthenesVariants(int32 limit) {
		euler_phi_plus_sum_and_number_of_diff_prime_factors(limit);
		num_and_sum_of_divisors(limit);
		num_and_sum_of_prime_factors(limit);
	}

	// Does a basic sieve. Complexity function 1.
	void numAndSumOfDivisors(int32 limit) {
		numDiv.assign(limit + 1, 1);
		sumDiv.assign(limit + 1, 1ll);  // likely needs to be long long
		for (int32 divisor = 2; divisor <= limit; divisor++)
			for (int32 multiple = divisor; multiple <= limit; multiple += divisor) {
				numDiv[multiple]++;
				sumDiv[multiple] += divisor;
			}
	}

	// This is basically same as sieve just using different ops. Complexity function 2.
	void eulerPhiPlusSumAndNumOfDiffPrimeFactors(int32 limit) {
		numDiffPF.assign(limit + 1, 0);
		sumDiffPF.assign(limit + 1, 0ll);	// likely needs to be long long
		phi.assign(limit + 1, 0ll);				// likely needs to be long long
		iota(phi.begin(), phi.end(), 0);
		for (int32 prime = 2; prime <= limit; prime++)
			if (0 == numDiffPF[prime])
				for (int32 multiple = prime; multiple <= limit; multiple += prime) {
					numDiffPF[multiple]++;
					sumDiffPF[multiple] += prime;
					phi[multiple] = (phi[multiple] / prime) * (prime - 1);
				}
	}

	// This uses similar idea to sieve but avoids divisions. Complexity function 3.
	void numAndSumOfPrimeFactors(int32 limit) {
		numPF.assign(limit + 1, 0);
		sumPF.assign(limit + 1, 0ll);	// likely needs to be long long
		for (int32 prime = 2; prime <= limit; prime++)
			if (0 == numDiffPF[prime]) {
				int32 exponentLimit = 0;	// p^n | p^n <= limit < p^(n+1)
				int64 primePows[32];			// 32 limits us to 2^31 (2^0 == 1), increase if needed
				for (int64 primeToN = 1; primeToN <= limit; primeToN *= prime)
					primePows[exponentLimit++] = primeToN;
				for (int32 exponent = 1; exponent < exponentLimit; exponent++) {
					int32 primeToN = primePows[exponent];	// WARNING assumption that limit = int32
					for (int32 multiple = primeToN; multiple <= limit; multiple += primeToN) {
						numPF[multiple]++;
						sumPF[multiple] += prime;
			} } } // 3 closing brackets
	}

	// This uses similar idea to sieve but avoids divisions. Complexity function 4
	void numAndSumOfDivisorsFaster(int32 limit) {
		// sumDiv.assign(limit + 1, 1ll);  // likely needs to be long long
		numDiv.assign(limit + 1, 1);
		curPow.assign(limit + 1, 1); // here a
		for (int32 prime = 2; prime <= limit; prime++)
			if (1 == numDiv[prime]) {
				int32 exponentLimit = 0;	// p^n | p^n <= limit < p^(n+1)
				int64 primePows[32];			// 32 limits us to 2^31 (2^0 == 1), increase if needed
				for (int64 primeToExp = 1ll; primeToExp <= limit; primeToExp *= prime)
					primePows[exponentLimit++] = primeToExp;
				// primePows[exponentLimit] = primePows[exponentLimit-1]*prime; // used in sumDiv
				for (int32 exponent = 1; exponent < exponentLimit; exponent++) {
					int32 primeToN = primePows[exponent];	// WARNING assumption that limit = int32
					for (int32 mul = primeToN; mul <= limit; mul += primeToN)
						curPow[mul]++;
				}
				for (int32 multiple = prime; multiple <= limit; multiple += prime) {
					// sumDiv[multiple] *= ((primePows[curPow[multiple]] - 1) / (prime - 1));
					numDiv[multiple] *= curPow[multiple];
					curPow[multiple] = 1; // needs to happen regardless of version you are using
				}
			}
	}


	//Tests if n is prime via divisors up to sqrt(n).
	//
	//	Complexity per call : Time: O(sqrt(n)), T(sqrt(n) / 3), Space : O(1)
	//	Optimizations: 6k+i method, since we checked 2 and 3 only need test form 6k+1 and 6k+5
	bool isPrimeTrivial(int32 n) {
		if (n < 4) return n > 1;										// base case of n in 0, 1, 2, 3
		if (n % 2 == 0 || n % 3 == 0) return false; // this is what allows us to use 6k+i
		int32 limit = lrint(sqrt(n)) + 1;
		for (int32 prime = 5; prime < limit; prime += 6)
			if (n % prime == 0 || n % (prime + 2) == 0)
				return false;
		return true;
	}

	// binary exponentiation
	//	taken from https https://cp-algorithms.com
	//	Complexity Time: log(base), Space O(1) 
	uint64 powMod(uint64 base, uint64 exp, uint64 mod) {
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

	// The witness test of miller rabin.
	//	taken from https https://cp-algorithms.com
	//	Complexity per call: Time O(log^3(n)), Space: O(1)
	bool mrptCompositeCheck(uint64 n, uint64 a, uint64 d, int32 s) {
		uint64 x = powMod(a, d, n);
		if (x == 1 || x == n - 1) return false;
		for (int32 r = 1; r < s; ++r) {
			x = (uint128)x * x % n;
			if (x == n - 1)
				return false;
		}
		return true;
	}

	// Handles all numbers up to 2 ^ 64 - 1 (maybe need to test it to gain confidence).
	//	taken from https https://cp-algorithms.com
	//	Complexity per call : Time O(12 log^3(n)), Space : O(log2(n)) bits
	//	Optimizations : test first 12 primes before running the algorithm or generate sieve.
	bool isPrimeMRPT(uint64 n) {
		if (n < 2) return false;
		int32 r = 0;
		uint64 d = n - 1;
		for (; (d & 1) == 0; d >>= 1, r++) {}	// turned whileloop into forloop

		for (int32 a: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
			if (0 == (n % a) || mrptCompositeCheck(n, a, d, r)) // 0==(n%a) replaced n==a
				return n == a;
		return true;
	}

	// A basic prime factorization of n function.without primes its just O(sqrt(n))
	//
	//	Complexity: Time: O(sqrt(n) / ln(sqrt(n))), Space : O(log n)
	//	Variants: number and sum of prime factors, of diff prime factors, of divisors, and phi
	int32_Vec primeFactorizeN(int32 n) {
		int32_Vec factors;
		for (const auto& prime : primesList) {	// assumes you have a prime list upto sqrt(n)
			if (prime * prime > n) break;					// this check here since we use loop iterator
			for (; n % prime == 0; n /= prime)
				factors.push_back(prime);
		}
		if (n > 1)	// only happens when n had a prime factor above the sqrt
			factors.push_back(n);
		return factors;
	}

	// An optimized prime factorization of n function based on min primes already sieved.
	//
	//	Complexity: Time: O(log n), Space : O(log n)
	//	Optimization: assign append to function and assign minPrimes[n] to a value in the loop
	int32_Vec primeFactorizeNLogN(int32 n) {
		int32_vec factors;
		for (; 1 < n; n /= minPrimes[n])
			factors.push_back(minPrimes[n]);
		return factors;
	}

	// Covers all the variants listed above, holds the same time and space complexity
	int32 primeFactorizeNVariants(int32 n) {
		int64 sumDiffPFV = 0, sumPFV = 0, sumDivV = 1, eulerPhi = n;
		int32 numDiffPFV = 0, numPFV = 0, numDivV = 1;
		for (const auto& prime : primesList) {  // for(int prime=2; prime*prime<=n; prime++)
			if (prime * prime > n) break;
			if (n % prime == 0) {
				int64 total = 1;	// for sum of divisors
				int32 power = 0;	// for num of divisors
				for (int64 mul = prime; n % prime == 0; n /= prime) {
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

	// Multiplicity of P and Legendre's formula allow us to calculate factors of factorial
	//	taken from https://cp-algorithms.com
	//	Complexity: Time: O(n log n)), Space: O(n / log n) [number of primes to limit]
	void factorialPrimeFactors(int32 limit) {
		int32 endPoint = upper_bound(primesList.begin(), primesList.end(), limit);
		endPoint = endPoint - primesList.begin();	// this fixes the offset
		numFactorialPF.assign(endPoint, 0);
		for (int32 primeIdx = 0; primeIdx < endPoint; ++primeIdx) {
			int32 prime = primesList[primeIdx];
			int64 primeAmt = 0;
			for (int32 x = limit; x; primeAmt += x)
				x /= prime;
			numFactorialPF[primeIdx] = primeAmt;
		}
	}

	uint64 rhoMulMod(uint64 a, uint64 b, uint64 mod) {
		return (uint128)a * b % mod;
	}

	uint64 rhoF(uint64 x, uint64 c, uint64 mod) {
		return (rhoMulMod(x, x, mod) + c) % mod;
	}

	uint64 rhoPollard(uint64 n, uint64 x0 = 2, uint64 c = 1) {
		uint64 x = x0, y = x0, g = 1;
		while (g == 1) {
			x = rhoF(x, c, n);
			y = rhoF(y, c, n);
			y = rhoF(y, c, n);
			g = gcd(abs(x - y), n);
		}
		return g;
	}

};
