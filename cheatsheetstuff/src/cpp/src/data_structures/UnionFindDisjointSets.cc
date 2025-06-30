#include <vector>		// for vec_int
#include <algorithm>	// for iota, swap

class UnionFindDisjointSets {
private:
	int32_Vec parent, rank, setSizes;
	int32 numSets;
public:
	// Attributes declared here must be passed in or global if not class based implementation
	UnionFindDisjointSets(int32 n) {
		parent.assign		(n, 0);
		setSizes.assign	(n, 1);							// optional information
		rank.assign			(n, 0);							// optional optimization 
		iota(rank.begin(), rank.end(), 0);	// rank = {0, 1, 2...}
		numSets = n;												// optional information
	}

	// Recursively find which set u belongs to. Memoize on the way back up.
	//
	// Complexity: Time: O(\alpha(n)) -> O(1), inverse ackerman practically constant
	//						Space: Amortized O(1) stack space
	int32 findSet(int32 u) { return (parent[u] == u) ? u : (parent[u] = findSet(parent[u])); }
	
	// Checks if u and v in same set. TIME and SPACE Complexity is the same as findSet
	bool isSameSet(int32 u, int32 v) { return findSet(u) == findSet(v); }

	// Gives you the size of set u. TIME and SPACE Complexity is the same as find_set
	int32 sizeOfSet(int32 u) { return setSizes[findSet(u); }

	// just returns private value numSets
	int32 getNumSets() const { return numSets; }
	
	// Join the set that contains u with the set that contains v.
	//
	//	Complexity: Time: O(\alpha(n))->O(1), inverse ackerman practically constant
	//						 Space: Amortized O(1) stack space
	void unionSet(int32 u, int32 v) {
		if (!isSameSet(u, v)) {
			int32 uParent = findSet(u), vParent = findSet(v);
			if (rank[uParent] > rank[vParent])	// uParent shorter than vParent
				swap(uParent, vParent);
			if (rank[uParent] == rank[vParent]) // optional speedup
				parent[vParent]++;
			parent[uParent] = vParent;							// line that joins u and v
			setsize[vParent] += setSizes[uParent];	// u = v so add join the size
			numSets--;	// if you need numSets keep this line
		}
	}
};

