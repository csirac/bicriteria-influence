#include <igraph.h>
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <queue>

using namespace std;
typedef unsigned long myint;
typedef unordered_set< myint >  uset;

struct mypair {
  myint first;
  myint second;
};

class influence_oracles {
public:
  myint ell;
  myint k;
  myint n;

  vector< igraph_t* > v_instances;

  vector< vector< myint > > global_sketches;

  void compute_oracles();
  
  void estimate_influence() {}

  void merge_sketches(vector< myint >& sketch_1, vector< myint >& sketch_2, 
		      vector< myint >& result ) {}

  influence_oracles( vector< igraph_t* > in_instances,
                     myint in_ell, 
                     myint in_k,
                     myint in_n) :
    v_instances( in_instances ), 
    ell( in_ell ), 
    k( in_k ),
    n( in_n )
  { }

private:

  void update_local_sketches( igraph_t* G_i, 
			      mypair root, 
			      vector< vector< myint > >& local_sketches ) {
    //root is a pair, first element is rank,
    //second is vertex

    //want to insert the rank of root to sketches of backwards reachable nodes
    //BFS backwards
    queue <myint> Q;
    vector < int > dist;
    dist.reserve( n );
    for (myint i = 0; i < n; ++i) {
      dist[i] = -1; //infinity

    }

    dist[ root.second ] = 0;
    Q.push( root.second );
    while (!(Q.empty())) {

      myint current = Q.front();
      Q.pop();
      //get backwards neighbors of current
      igraph_vector_t neis;
      igraph_vector_init( &neis, 0 );
      igraph_neighbors( G_i, &neis, current, IGRAPH_IN );
      for (myint i = 0; i < igraph_vector_size( &neis ); ++i) {
	myint aneigh = VECTOR( neis )[i];
	if (dist[ aneigh ] == -1 ) {
	  dist[ aneigh ] = dist[ current ] + 1;
	  Q.push( aneigh );
	  //update the local sketch of 'aneigh'
	  //with the rank of the root
	  //since we look at ranks in increasing order,
	  //pushing onto the end will maintain the
	  //property that the sketches are sorted
	  if ( local_sketches[ aneigh ].size() < k ) {
	    local_sketches[ aneigh ].push_back( root.first );
	  }
	}
      }
      igraph_vector_destroy( &neis );
    } //BFS finished

  }

};

void influence_oracles::compute_oracles() {
  // create the permutation of size n*ell
  myint K = n * ell;
  vector< mypair > perm;
  perm.reserve( K );
  for (myint u = 0; u < n; ++u) {
    for (myint i = 0; i < ell; ++i) {

      mypair tmp;
      tmp.first = u;
      tmp.second = i;
      perm.push_back( tmp );
    }
  }

  random_shuffle( perm.begin(), perm.end() );

  // group ranks by instance
  // the pairs are in order of rank
  vector< vector < mypair > > instanceRanks;
  for (myint r = 1; r <= K; ++r ) {
    myint i = perm[ r - 1 ].second;
    mypair tmp;
    tmp.first = r;
    tmp.second = perm[ r - 1 ].first;
    instanceRanks[i].push_back( tmp );
  }

  // compute combined bottom-k rank sketches
  
  // for each instance i, compute local bottom-k sketches
  vector < vector< myint > > local_sketches;
  vector < myint > empty_sketch;
  global_sketches.assign( n, empty_sketch );

  for (myint i = 0; i < ell; ++i) {
    local_sketches.clear();
    local_sketches.assign( n, empty_sketch );
      
    // for each vertex in instance i by increasing rank
    for (myint j = 0; j < n; ++j) {
      myint vertex = instanceRanks[i][j].second;
      // Run reverse BFS in instance i from 'vertex', 
      // updating sketches of discovered vertices
      update_local_sketches( v_instances[ i ],
			     instanceRanks[i][j],
			     local_sketches );
    }

    // merge local_sketches in instance i
    // into the global sketch for each node
    vector< myint > new_sketch;
    for (myint u = 0; u < n; ++u) {
      new_sketch.assign( local_sketches[u].size()
                         + global_sketches[u].size(),
                         0
                        );

      merge( local_sketches[u].begin(),
             local_sketches[u].end(),
             global_sketches[u].begin(),
             global_sketches[u].end(),
             new_sketch.begin() );

      if (new_sketch.size() > k)
        new_sketch.resize( k );

      global_sketches[u].swap( new_sketch );
      
    }


  }
}





