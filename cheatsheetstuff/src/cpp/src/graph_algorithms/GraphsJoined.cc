/// <summary>
/// ///////////////////////////////////////////////////////////////////
/// </summary>

typedef int32_pair row_col;			
typedef int32_pair edge_U2V;			// used for unweighted edges 
typedef int32_pair edge_VWt;			// this might change if wt is not an int32
typedef int32_triple edge_WTU2V;	// this might change if wt is not an int32
typedef int32_triple edge_VWtDat;	// this might change if wt or data is not an int32

typedef vector<edge_WTU2V> edge_list;
typedef vector<edge_VWtDat> edge_list_flow;	// used for flow graphs
typedef vector<int32_Pair_Vec> adj_list_wt;	// change to match edge_WtV if needed
typedef vector<int32_Vec> adj_list;	
typedef vector<int32_Vec> adj_mat;					// change inner type if matrix doesn't use int
typedef vector<int32_Vec> grid_graph;				// change inner type if grid doesn't use int

typedef DAT_type_to data_map;			// can change data type as needed. default at top of book
typedef DAT_type_from index_map;	// type must match leftside type above

class Graph {
private:
	// only take what you actually need for optimial performance
	int32 numEdges, numNodes, numRows, numCols;
	adj_list adjList, adjListTrans;
	adj_list_wt adjListWt, adjListTransWt;
	adj_mat adjMatrix;
	edge_list edgeList;	// can also make this into a heap
	edge_list_flow edgeListFlow;
	data_map dataToIndex;
	index_map indexToData;
	grid_graph grid;
public:
	// Converts data to the form: int u | 0 <= u < |V|, stores (data, u) pair.
	int32 convertDataExample(dataType data) {
		if (!dataToIndex.contains(data) {
			dataToIndex[data] = indexToData.size();
			indexToData.push_back(data);
		}
		return dataToIndex[data];
	}

	// A pick and choose function will convert u, v into index form then add it to the
	// structure you need.
	void addEdgeUVWtExample(dataType u, dataType v, int32 wt, int32 data) {
		int32 node_u = convertDataExample(u);
		int32 node_v = convertDataExample(v);

		adjList[node_u].push_back(node_v);
		// adjListWt[node_u].push_back({ node_v, wt });	// use this for weighted
		adjMatrix[node_u][node_v] = wt;									// Adjacency matrix usage
		edgeList.push_back({ wt, node_u, node_v });			// edge list usage
		edgeList.push_back({ node_v, wt, data });				// this one is used for flow
		adjList[node_u].push_back(edgeList.size() - 1));// also for flow ?
	}

	// undirected graph version of the previous function. wt can be omitted.
	void addEdgeUndirected(dataType u, dataType v, int32 wt, int32 data) {
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
	int32_Pair_Vec dirRC = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };
	Graph graph;
	edge_list minSpanningTree;
public:

	// Computes flood fill graph traversal via recursive depth first search.
	//
	//	Complexity : Time: O(|V|+|E|), Space: O(|V|): for grids |V| = row*col and |E| = 4*|V|
	//	More uses : Region Colouring, Connectivity, Area / island Size, misc
	//	Input:
	//		row, col : integer pair representing current grid position
	//		old_val, new_val : unexplored state, the value of an explored state
	void floodFillViaDFS(int32 row, int32 col, int32 oldVal, int32 newVal) {
		graph.grid[row][col] = newVal;
		for (const auto& [rowMod, colMod] : dirRC) {
			int32 newRow = row + rowMod, newCol = col + colMod;
			if (0 <= newRow && newRow < graph.numRows &&
					0 <= newCol && newCol < graph.numCols &&
					graph.grid[newRow][newCol] == oldVal)
				floodFillViaDFS(newRow, newCol, oldVal, newVal);
		}
	}

	// Computes flood fill graph traversal via breadth first search.Use on grid graphs.
	//
	//	Complexity: Time: O(|V|+|E|), Space: O(|V|): for grids |V| = row*col and |E| = 4*|V|
	//	More uses : previous uses tplus shortest connected pah
	void floodFillViaBFS(int32 startRow, int32 startCol, int32 oldVal, int32 newVal) {
		deque<row_col> q = { {startRow, startCol} };	// q to avoid name conflicts with queue
		while (!q.empty()) {
			const auto& [newRow, newCol] = q.back(); q.pop_back();	// THIS IS TWO LINES IN ONE!
			for (const auto& [rowMod, colMod]: dirRC) {
				int32 newRow = row + rowMod, newCol = col + colMod;
				if (0 <= newRow && newRow < graph.numRows &&
						0 <= newCol && newCol < graph.numCols &&
						graph.grid[newRow][newCol] == oldVal) {
					graph.grid[newRow][newCol] = newVal;
					q.push_front({ newRow, newCol });
		} } } // 3 brackets closes up to the while loop
	}

	//Computes mst of graph G stored in edge_list, space optimized via heap.
	//
	//	Complexity per call: Time: O(|E| log|V|), Space: O(|E| log|E|) + Union_Find
	//	More uses : finding min spanning tree
	//	Variants : min spanning subgraph and forrest, max spanning tree, 2nd min best spanning
	//	Optimization : We use a heap to make space comp.O(| E | ).instead of O(| E | log | E | )
	//	when using sort, however edge_list is CONSUMED.Also uses space optimization
	void minSpanningTreeViaKruskalsWithHeaps() {
		int32 vertices = graph.numNodes;
		sort(graph.edgeList.begin(), graph.edgeList.end());  // optimization line here
		auto UFDS = UnionFindDisjointSets(vertices);
		for (const auto& [wt, u, v]: adj_edge_list) { // for heaps: need to pop heap in loop 
			if (UFDS.getNumSets() == 1) break;
			if (!UFDS.isSameSet(u, v)) {
				minSpanningTree.push_back({ wt, u, v });
				UFDS.unionSet(u, v);
		} } //  2 brack closed on the for loop
	}

	void primsViaAdjMatrix(int source) {
		/*Find min weight edge in adjacency matrix implementation of prims.
		* 
		* Complexity per call: Time: O(|V|^2), T(|V| * 4|V|), Space: O(|V|), S(~5|V|)
		*/
		int vertices = graph.numNodes;

	}
};
