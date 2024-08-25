#include <bits/stdic++.h> // This include is mostly for catch all and speed

typedef vector<int> vec_int;

#include <vector>		// for vec_int
#include <algorithm>	// for iota

class UnionFindDisjointSets {
private:
	vec_int parent, rank, set_sizes;
	int num_sets;
public:
	UnionFindDisjointSets(int n) {
		// Attributes declared here must be passed in or global if not used in classes
		parent.assign(n, 0);
		num_sets = n;						// optional information
		set_sizes.assign(n, 1);				// optional information
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
		// Checks if u and v in same set. TIME and SPACE Complexity is the same as find_set
		return findSet(u) == findSet(v);
	}


};