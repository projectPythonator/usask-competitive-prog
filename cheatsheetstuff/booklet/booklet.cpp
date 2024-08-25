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


