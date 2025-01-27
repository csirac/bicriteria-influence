#include <igraph.h>
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <queue>
#include <iostream>
#include <sstream>
#include <pthread.h>
#include <bitset>
#include <list>

#define MAX_N 10000000//for bitset

using namespace std;
typedef igraph_integer_t myint;
typedef double myreal;
typedef unordered_set< myint >  uset;

std::random_device rd_or;
std::mt19937 gen_or(rd_or());

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

struct mypair {
  myint first;
  myint second;
};

struct npair {
  myint node;
  myreal nrank;
};

struct sk_type {
  myint node;
  myreal rank;
  
};

struct est_node {
  myint node;
  myreal sk_rank;
  //vector< npair > sketch; 
  //the rank values assigned to
  //this node
  list < myreal > which_assigned; 
};

class C2_est {
public:
  vector< vector< myreal > >& ugs; //the uniform sketches

  //maintains sorted order by rank value
  //largest to smallest
  list < est_node > mem; 
  myreal estimate;
  myint& k;

  void add_node( myint id ) {
    //get sketch of this node, and its rank
    vector< myreal >& id_sketch = ugs[ id ];
    myreal id_rank;
    if (id_sketch.size() == k) {
      id_rank = id_sketch.back();
      id_sketch.pop_back(); //don't want to consider the sketch rank
      //need to add this back at the end
    } else {
      id_rank = 1.0;
    }
    
    est_node en_id;
    en_id.node = id;
    en_id.sk_rank = id_rank;

    list< myreal >& id_which_assigned = en_id.which_assigned;

    //find position of id in mem
    list< est_node >::iterator it_mem = mem.begin();
    vector< bool > b_assign( id_sketch.size(), false );
    while ( it_mem -> sk_rank > id_rank ) {
      //need to check the sketches of nodes of larger rank
      //to determine if they contain elements in id's sketch
      bool bfound = false;
      list < myreal >& wa_ref = it_mem -> which_assigned ; 
      list < myreal >::iterator it_wa; 
      for ( unsigned i = 0; i < id_sketch.size(); ++i ) {
	it_wa = wa_ref.begin();
	while (it_wa != wa_ref.end() ) {
	  if ( id_sketch[i] == (*it_wa) ) {
	    b_assign[i] = true;
	    break;
	  }
	  ++it_wa;
	}
      }

      ++it_mem;
    }
    //    cout << "here1" << endl;
    //id_rank >= it_mem -> sk_rank
    //remember the position to insert
    list< est_node >::iterator it_pos = it_mem;

    //next, we need to see if lower rank assignments
    //should be re-assigned to id
    //if so, estimate will change
    vector< myreal >::iterator it_sk;
    vector< bool >::iterator it_bsk = b_assign.begin();
    bool bfnd;
    list< myreal >::iterator it_wa;
    while( it_mem != mem.end() ) {
      it_wa = (it_mem -> which_assigned).begin();
      while (it_wa != (it_mem -> which_assigned ).end()) {
	bfnd = false;
	it_sk = id_sketch.begin();
	it_bsk = b_assign.begin();
	while (it_sk != id_sketch.end()) {
	  if ( (*it_sk)  == ( *it_wa ) ) {
	    bfnd = true; //it_wa was found
	    (*it_bsk) = true; //taken care of this member of sketch
	    //add to id -> which_assigned
	    id_which_assigned.push_back( *it_wa );
	    //delete from it_mem -> which_assigned,
	    it_wa = (it_mem -> which_assigned).erase( it_wa );
	    //update estimate accordingly
	    //need to subtract off 1 / it_mem -> sk_rank
	    estimate -= 1.0 / it_mem -> sk_rank;
	    estimate += 1.0 / id_rank;
	  }
	  ++it_sk;
	  ++it_bsk;
	}
	if (!bfnd)
	  ++it_wa;
      }

      ++it_mem;
    }

    //at this point, any unassigned members of id's
    //sketch should be assigned to id itself
    //this will change the estimate
    it_sk = id_sketch.begin();
    it_bsk = b_assign.begin();

    while ( it_sk != id_sketch.end() ) {
      if ( !(*it_bsk) ) {
	id_which_assigned.push_back( *it_sk );
	estimate += 1.0 / id_rank;
      }

      ++it_sk;
      ++it_bsk;
    }

    id_sketch.push_back( id_rank );
    

    en_id.node = id;
    //insert en_id to mem
    mem.insert( it_pos, en_id );
  }

  //dry does not actually add
  //the node to the set
  //just simulates it
  myreal add_node_dry( myint id ) {
    myreal dry_estimate = estimate;

    //get sketch of this node, and its rank
    vector< myreal >& id_sketch = ugs[ id ];
    myreal id_rank;
    if (id_sketch.size() == k) {
      id_rank = id_sketch.back();
      id_sketch.pop_back(); //don't want to consider the sketch rank
      //need to add this back at the end
    } else {
      id_rank = 1.0;
    }
    
    est_node en_id;
    en_id.node = id;
    en_id.sk_rank = id_rank;

    list< myreal >& id_which_assigned = en_id.which_assigned;

    //find position of id in mem
    list< est_node >::iterator it_mem = mem.begin();
    vector< bool > b_assign( id_sketch.size(), false );
    while ( it_mem -> sk_rank > id_rank ) {
      //need to check the sketches of nodes of larger rank
      //to determine if they contain elements in id's sketch
      bool bfound = false;
      list < myreal >& wa_ref = it_mem -> which_assigned ; 
      list < myreal >::iterator it_wa; 
      for ( unsigned i = 0; i < id_sketch.size(); ++i ) {
	it_wa = wa_ref.begin();
	while (it_wa != wa_ref.end() ) {
	  if ( id_sketch[i] == (*it_wa) ) {
	    b_assign[i] = true;
	    break;
	  }
	  ++it_wa;
	}
      }

      ++it_mem;
    }
    //    cout << "here1" << endl;
    //id_rank >= it_mem -> sk_rank
    //remember the position to insert
    list< est_node >::iterator it_pos = it_mem;

    //next, we need to see if lower rank assignments
    //should be re-assigned to id
    //if so, estimate will change
    vector< myreal >::iterator it_sk;
    vector< bool >::iterator it_bsk = b_assign.begin();
    bool bfnd;
    list< myreal >::iterator it_wa;
    while( it_mem != mem.end() ) {
      it_wa = (it_mem -> which_assigned).begin();
      while (it_wa != (it_mem -> which_assigned ).end()) {
	bfnd = false;
	it_sk = id_sketch.begin();
	it_bsk = b_assign.begin();
	while (it_sk != id_sketch.end()) {
	  if ( (*it_sk)  == ( *it_wa ) ) {
	    bfnd = true; //it_wa was found
	    (*it_bsk) = true; //taken care of this member of sketch
	    //add to id -> which_assigned
	    id_which_assigned.push_back( *it_wa );
	    //don't delete from it_mem -> which_assigned, this
	    //is dry run
	    //it_wa = (it_mem -> which_assigned).erase( it_wa );
	    ++it_wa;
	    //update estimate accordingly
	    //need to subtract off 1 / it_mem -> sk_rank
	    dry_estimate -= 1.0 / it_mem -> sk_rank;
	    dry_estimate += 1.0 / id_rank;
	  }
	  ++it_sk;
	  ++it_bsk;
	}
	if (!bfnd)
	  ++it_wa;
      }

      ++it_mem;
    }

    //at this point, any unassigned members of id's
    //sketch should be assigned to id itself
    //this will change the estimate
    it_sk = id_sketch.begin();
    it_bsk = b_assign.begin();

    while ( it_sk != id_sketch.end() ) {
      if ( !(*it_bsk) ) {
	id_which_assigned.push_back( *it_sk );
	dry_estimate += 1.0 / id_rank;
      }

      ++it_sk;
      ++it_bsk;
    }

    id_sketch.push_back( id_rank );
    

    en_id.node = id;
    //do not insert en_id to mem, this is dry run
    //mem.insert( it_pos, en_id );
    return dry_estimate;
  }

  C2_est( vector< vector < double > >& in_ugs,
	  myint& in_k ) : ugs( in_ugs ), k( in_k ) {
    estimate = 0.0;
  }
};

bool mycomp (const npair& n1, const npair& n2 ) {
  return n1.nrank < n2.nrank;
}

bool mycomp2 (const npair& n1, const npair& n2 ) {
  return n1.nrank > n2.nrank;
}

class influence_oracles {
public:
  myint ell;
  myint k;
  myint n;
  myint or_max_dist;

  myreal t_oracles;
  myreal t_oracles_wall;

  C2_est imp_est;

  vector< igraph_t* > v_instances;
  vector< vector< myint > > global_sketches;
  vector< vector< mypair > > instanceRanks;

  myreal offset;
  vector< vector< myreal > > uniform_global_sketches;

  void compute_oracles();
  void alt_compute_oracles();

  void compute_oracles_online_init();
  void compute_oracles_online_step( igraph_t* H_i,
                                    myint i );

  void compute_uniform_oracles_online_init();
  void compute_uniform_oracles_online_step( igraph_t* H_i,
					    myint i, myreal offset,
					    bitset< MAX_N >& bs_ext);

  void write_oracles( ostream& os ) {
    os << k <<  ' ' << ell << ' ' << n << ' ' << t_oracles << ' ' << t_oracles_wall << ' ' << or_max_dist << ' ' << offset << endl;
    for (myint i = 0; i < n; ++i) {
      for (unsigned j = 0; j < uniform_global_sketches[ i ].size(); ++j ) {
	os << uniform_global_sketches[ i ][ j ] << ' ';

      }
      os << endl;
    }

  }

  void read_oracles( istream& is ) {
    string sline;
    getline( is, sline );
    istringstream iss;
    iss.clear();
    iss.str( sline );
    iss >> k >> ell >> n >> t_oracles >> t_oracles_wall >> or_max_dist >> offset;

    myint i = 0;
    vector< myreal > empty_sketch;
    uniform_global_sketches.assign( n, empty_sketch );
    while( getline( is, sline ) ) {
      iss.clear();
      iss.str( sline );
      myreal tmp;
      while (iss >> tmp) {
	uniform_global_sketches[ i ].push_back( tmp );
      }
      ++i;
    }

    imp_est.mem.clear();
  }

  myreal estimate_reachability_sketch( vector< myint >& asketch ) {
    myreal uniform_rank;
    myreal estimate;
    if (asketch.size() == k) {
      myint T = asketch.back();
      uniform_rank = ((myreal) T - 1.0)/(ell * n - 1.0 );
      estimate = ((myreal) k - 1)/ uniform_rank;
    }
    else {
      estimate = ((myreal) asketch.size());
    }
      
    
    //myreal estimate = 1.0 + 
    //      ((myreal)(k - 1)*(n*ell - 1)) / (T - 1);

    return estimate / ell;

  }

  myreal estimate_reachability_uniform_sketch( vector< myreal >& asketch ) {
    myreal uniform_rank;
    myreal estimate;
    if (asketch.size() == k) {
      uniform_rank = asketch.back();
      estimate = ((myreal) k - 1)/ uniform_rank;
    }
    else {
      estimate = asketch.size();
    }
      
    
    //myreal estimate = 1.0 + 
    //      ((myreal)(k - 1)*(n*ell - 1)) / (T - 1);

    return estimate / ell;

  }

  

  myreal better_estimate_cohen( vector< myint >& s, double& t1, double& t2 ) {
    //assumes reachability sets are disjoint
    double estimate = 0.0;
    npair tmp;
    vector< npair > rank_values;
    vector< npair > sketches;

    clock_t start, finish;
    start = clock();
    for (unsigned i = 0; i < s.size(); ++i) {
      tmp.node = s[i];
      if (uniform_global_sketches[ s[i] ].size() == k)
	tmp.nrank = uniform_global_sketches[ s[i] ].back();
      else
	tmp.nrank = 1.0;

      sketches.push_back( tmp );

      //merge the sketches together,
      //keeping track of which node
      //each rank came from
      vector< npair > track_sketch;
      track_sketch.reserve( k );
      for (unsigned j = 0; j < uniform_global_sketches[ s[i] ].size(); ++j ) {
	npair tmp;
	tmp.node = s[i];
	tmp.nrank = uniform_global_sketches[ s[i] ][ j ];
	track_sketch.push_back(tmp );
      }
      if (track_sketch.size() == k)
	track_sketch.pop_back(); //discard the rank of the sketch

      vector< npair > result;
      npair tmp2;
      result.assign( rank_values.size() + track_sketch.size(), tmp2 );
      merge( rank_values.begin(), rank_values.end(),
	     track_sketch.begin(), track_sketch.end(),
	     result.begin(), mycomp );

      rank_values.swap( result );
		       
    }

    finish = clock();
    t1 = ( (double )finish - start) / CLOCKS_PER_SEC;

    start = clock();
    std::sort( sketches.begin(), sketches.end(), mycomp2 );

    //for each distinct rank_value, find 
    //the largest sketch rank to which it belongs
    //to that sketch
    vector< npair >::iterator itrv = rank_values.begin();
    vector< npair >::iterator itrvend = rank_values.end();
    vector< myint > v_nodes;
    while (itrv != itrvend) {
      v_nodes.clear();
      //      cout << i << ' '  << rank_values[i] << endl;
      //if a rank value is not unique, it is equal
      //to the one before it, the rank_values are sorted
      v_nodes.push_back( itrv -> node );
      while ( ( (itrv + 1) != itrvend) && ((itrv + 1)->nrank == itrv->nrank ) ) {
	v_nodes.push_back( (itrv + 1) -> node );
	++itrv;
      }
      
      //find which node corresponds to the max. rank
      bool bfound = false;

      for (unsigned j = 0; j < sketches.size(); ++j) {
	myint snode = sketches[j].node;
	for (unsigned k = 0; k < v_nodes.size(); ++k) {
	  if (v_nodes[k] == snode) {
	    //add to estimate
	    estimate += 1.0 / sketches[j].nrank;
	    bfound = true;
	    break;
	  }
	}
      
	if (bfound) {
	  break;
	}
      }
      ++itrv;
    }

    finish = clock();
    t2 = ( (double) finish - start) / CLOCKS_PER_SEC;
      return estimate / ell;
  }

  myreal estimate_reachability( myint vertex ) {
    myreal uniform_rank;
    myreal estimate;
    if (global_sketches[ vertex ].size() == k) {
      myint T = global_sketches[ vertex ].back();
      uniform_rank = ((myreal) T - 1.0)/(ell * n - 1.0 );
      estimate = ((myreal) k - 1)/ uniform_rank;
    }
    else {
      estimate = ((myreal) global_sketches[vertex].size());
    }
      




    
    //myreal estimate = 1.0 + 
    //      ((myreal)(k - 1)*(n*ell - 1)) / (T - 1);

    return estimate / ell;
  }

  myreal average_reachability( myint vertex ) {
    myint tot_reach = 0;
    for (myint i = 0; i < ell; ++i) {
      tot_reach += forwardBFS( v_instances[i], vertex );

    }

    return ((myreal) tot_reach) / ell;
  }


  //This version of the constructor does not
  //store any of the instances.
  //To compute the oracles, the online
  //functions must be called
  influence_oracles( myint in_ell,
                     myint in_k,
                     myint in_n ) :
    ell( in_ell ), 
    k( in_k ),
    n( in_n ), 
    imp_est( uniform_global_sketches, k )
  { }

  influence_oracles( vector< igraph_t* >& in_instances,
                     myint in_ell, 
                     myint in_k,
                     myint in_n) :
    v_instances( in_instances ), 
    ell( in_ell ), 
    k( in_k ),
    n( in_n ), 
    imp_est( uniform_global_sketches, k )
  { }

  influence_oracles( vector< igraph_t >& in_instances,
                     myint in_ell, 
                     myint in_k,
                     myint in_n) :
    ell( in_ell ), 
    k( in_k ),
    n( in_n ), 
    imp_est( uniform_global_sketches, k )
  { 
    igraph_t* nullpoint;
    v_instances.assign( ell, nullpoint);

    for (myint i = 0; i < ell; ++i) {
      v_instances[i] = &(in_instances[i]);

    }
  }

  myint forwardBFS( igraph_t* G_i, myint vertex ) {

    queue <myint> Q;
    vector < int > dist;
    dist.reserve( n );
    for (myint i = 0; i < n; ++i) {
      dist.push_back( -1 ); //infinity
    }

    dist[ vertex ] = 0;
    Q.push( vertex );
    while (!(Q.empty())) {

      myint current = Q.front();
      Q.pop();
      //get forwards neighbors of current
      igraph_vector_t neis;
      igraph_vector_init( &neis, 0 );
      igraph_neighbors( G_i, &neis, current, IGRAPH_OUT );
      for (myint i = 0; i < igraph_vector_size( &neis ); ++i) {
	myint aneigh = VECTOR( neis )[i];
	if (dist[ aneigh ] == -1 ) {
	  dist[ aneigh ] = dist[ current ] + 1;
	  Q.push( aneigh );

	}
      }

      igraph_vector_destroy( &neis );
    } //BFS finished

    myint count = 0;
    for (myint i = 0; i < n; ++i) {
      if (dist[i] != -1) {
	//i is reachable from vertex in G_i
	++count;
      }
    }

    return count;

  }

  void rankorder_BFS( igraph_t* G_i, 
                      igraph_vector_t& vRO,
                      igraph_vector_ptr_t& res) {

    //get neighborhoods for all nodes
    igraph_vs_t vs;
    igraph_vs_vector( &vs, &vRO );

    igraph_neighborhood(
			G_i,
			&res,
			vs,
			or_max_dist, //4, //hop=4
			IGRAPH_IN );

    
    igraph_vs_destroy( &vs );
  }

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

  cerr << "Permutation shuffled" << endl;

  // group ranks by instance
  // the pairs are in order of rank
  vector< mypair > emptyInstance;
  vector< vector < mypair > > instanceRanks( ell, emptyInstance );

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
    cerr << i << endl;
    local_sketches.clear();
    local_sketches.assign( n, empty_sketch );
      
    // for each vertex in instance i by increasing rank
    cerr << "Updating local ranks..." << endl;
    for (myint j = 0; j < n; ++j) {
      myint vertex = instanceRanks[i][j].second;

      // Run reverse BFS in instance i from 'vertex', 
      // updating sketches of discovered vertices
      // (and not including vertex itself)
      update_local_sketches( v_instances[ i ],
			     instanceRanks[i][j],
			     local_sketches );
    }
    
    cerr << "Performing merges..." << endl;
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

void influence_oracles::alt_compute_oracles() {
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

  cerr << "Permutation shuffled" << endl;

  // group ranks by instance
  // the pairs are in order of rank
  vector< mypair > emptyInstance;
  vector< vector < mypair > > instanceRanks( ell, emptyInstance );

  for (myint r = 1; r <= K; ++r ) {
    myint i = perm[ r - 1 ].second;
    mypair tmp;
    tmp.first = r;
    tmp.second = perm[ r - 1 ].first;
    instanceRanks[i].push_back( tmp );
  }

  // compute combined bottom-k rank sketches
  // for each instance i, compute local 
  // bottom-k sketches
  vector < vector< myint > > local_sketches;
  vector < myint > empty_sketch;
  global_sketches.assign( n, empty_sketch );
  
  for (myint i = 0; i < ell; ++i) {
    cerr << i << endl;
    local_sketches.clear();
    local_sketches.assign( n, empty_sketch );
      
    // for each vertex in instance i by increasing rank
    igraph_vector_t vRankOrder;
    igraph_vector_init( &vRankOrder, 0 );

    for (myint j = 0; j < n; ++j) {
      myint vertex = instanceRanks[i][j].second;
      igraph_vector_push_back( &vRankOrder, vertex );
    }

    // Run reverse BFS in instance i from 'vertex', 
    // updating sketches of discovered vertices
    // (and not including vertex itself)
    igraph_vector_ptr_t res;
    igraph_vector_ptr_init( &res, 0 );
    rankorder_BFS( v_instances[i],
                   vRankOrder, res );
    //res now contains the neighborhood
    //lists, in the correct order
    
    //insert ranks of roots to reverse
    //reachable nodes
    for (myint ii = 0; 
         ii < igraph_vector_ptr_size( &res ); 
         ++ii ) {
      igraph_vector_t* v;
      //get the nbhd list for the node at this rank
      v = (igraph_vector_t* ) igraph_vector_ptr_e( &res, i );
      for (myint j = 0; 
           j < igraph_vector_size( v ); 
           ++j) {
        //aa is a reverse reachable node from rank ii
        myint aa = igraph_vector_e( v, j );
        //push the rank of the ii'th node
        //onto the sketch of aa
        if (local_sketches[aa].size() < k)
          local_sketches[aa].push_back( instanceRanks[i][ii].first );
      }
    }
    //can deallocate res now
    //    void (*mypointer)( igraph_vector_t* ) = &igraph_vector_destroy;
    //    igraph_vector_ptr_set_item_destructor(&res,
    //					  mypointer );
    igraph_vector_ptr_destroy_all( &res );
    igraph_vector_destroy( &vRankOrder );

    cerr << "Performing merges..." << endl;
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

void influence_oracles::compute_oracles_online_init() {
  //create the permutation and ranks
  //we already know ell, even though we don't have
  //the graphs
  
  // create the permutation of size n*ell
  vector< mypair >::size_type K = ((vector< mypair>::size_type) n) * ((vector< mypair >::size_type) ell );
  vector< mypair > perm;

  cout << "K: " << K << endl;

  //  perm.reserve( K );

  for (myint u = 0; u < n; ++u) {
    for (myint i = 0; i < ell; ++i) {
      mypair tmp;
      tmp.first  = u;
      tmp.second = i;
      perm.push_back( tmp );
    }
  }

  //shuffle the permutation
  random_shuffle( perm.begin(), perm.end() );

  //group ranks by instance
  //the pairs are in order of rank
  vector< mypair > emptyInstance;
  instanceRanks.assign( ell, emptyInstance );

  

  for (myint r = 1; r <= K; ++r ) {
    myint i = perm[ r - 1 ].second;
    mypair tmp;
    tmp.first = r;
    tmp.second = perm[ r - 1 ].first;
    instanceRanks[i].push_back( tmp );
  }

  // for (myint r = 1; r <= K; ++r ) {
  //   myint i = perm.back().second;
  //   mypair tmp;
  //   tmp.first = r;
  //   tmp.second = perm.back().first;
  //   instanceRanks[i].push_back( tmp );

  //   perm.pop_back();
  // }

  vector < myint > empty_sketch;
  global_sketches.assign( n, empty_sketch );

}

void influence_oracles::compute_uniform_oracles_online_init() {
  //uniform version is a lot simpler
  offset = 0.0;
  vector < myreal > empty_sketch;
  uniform_global_sketches.assign( n, empty_sketch );

}

void influence_oracles::
compute_uniform_oracles_online_step(
				    igraph_t* H_i,
				    myint i,
				    myreal offset_in,
				    bitset< MAX_N >& bs_ext
				    ) {
  vector < vector< myreal > > local_sketches;
  vector < myreal > empty_sketch;
  local_sketches.assign( n, empty_sketch );

  vector < npair > ranks_i;
  ranks_i.reserve( n );

  // give each (v, i) a uniform rank
  std::uniform_real_distribution<> dis(0, 1);
  for (myint i = 0; i < n; ++i) {
    npair tmp;
    tmp.node = i;
    tmp.nrank = dis( gen_or );

    ranks_i.push_back( tmp );
  }

  std::sort( ranks_i.begin(), ranks_i.end(), mycomp );

  igraph_vector_t vRankOrder;
  igraph_vector_init( &vRankOrder, 0 );
  
  for (myint j = 0; j < n; ++j) {
    myint vertex = ranks_i[j].node;
    if (!(bs_ext.test( vertex )))
      igraph_vector_push_back( &vRankOrder, vertex );
  }

  // Run reverse BFS in instance i from 'vertex', 
  // updating sketches of discovered vertices
  // (and not including vertex itself)
  // (or including it)

  igraph_vector_ptr_t res;
  igraph_vector_ptr_init( &res, 0 );
  rankorder_BFS( H_i,
                 vRankOrder, res );
  //res now contains the neighborhood
  //lists, in the correct order
  //insert ranks of roots to reverse
  //reachable nodes
  for (myint ii = 0; 
       ii < igraph_vector_ptr_size( &res ); 
       ++ii ) {
    igraph_vector_t* v;
    //get the nbhd list for the node at this rank
    v = (igraph_vector_t* ) igraph_vector_ptr_e( &res, ii );
    for (myint j = 0; 
         j < igraph_vector_size( v ); 
         ++j) {
      //aa is a reverse reachable node from rank ii
      myint aa = igraph_vector_e( v, j );
      //push the rank of the ii'th node
      //onto the sketch of aa
      if (local_sketches[aa].size() < k)
        local_sketches[aa].push_back( ranks_i[ ii ].nrank );
    }
  }
  //can deallocate res now
  IGRAPH_VECTOR_PTR_SET_ITEM_DESTRUCTOR( &res, &igraph_vector_destroy );
  igraph_vector_ptr_destroy_all( &res );
  igraph_vector_destroy( &vRankOrder );

  //get the lock first
  pthread_mutex_lock( &mutex1 );
  offset += offset_in;
  // merge local_sketches in instance i
  // into the global sketch for each node
  vector< myreal > new_sketch;
  for (myint u = 0; u < n; ++u) {
    new_sketch.assign( local_sketches[u].size()
                       + uniform_global_sketches[u].size(),
                       0
                       );

    merge( local_sketches[u].begin(),
           local_sketches[u].end(),
           uniform_global_sketches[u].begin(),
           uniform_global_sketches[u].end(),
           new_sketch.begin() );

    if (new_sketch.size() > k)
      new_sketch.resize( k );

    uniform_global_sketches[u].swap( new_sketch );
  }

  //and unlock 
  pthread_mutex_unlock( &mutex1 );

}

void influence_oracles::
compute_oracles_online_step(
                            igraph_t* H_i,
                            myint i
                            ) {
  vector < vector< myint > > local_sketches;
  vector < myint > empty_sketch;
  local_sketches.assign( n, empty_sketch );
  // for each vertex in instance i by increasing rank
  igraph_vector_t vRankOrder;
  igraph_vector_init( &vRankOrder, 0 );
  
  for (myint j = 0; j < n; ++j) {
    myint vertex = instanceRanks[i][j].second;
    igraph_vector_push_back( &vRankOrder, vertex );
  }

  // Run reverse BFS in instance i from 'vertex', 
  // updating sketches of discovered vertices
  // (and not including vertex itself)
  // (or including it)

  igraph_vector_ptr_t res;
  igraph_vector_ptr_init( &res, 0 );
  rankorder_BFS( H_i,
                 vRankOrder, res );
  //res now contains the neighborhood
  //lists, in the correct order
  //insert ranks of roots to reverse
  //reachable nodes
  for (myint ii = 0; 
       ii < igraph_vector_ptr_size( &res ); 
       ++ii ) {
    igraph_vector_t* v;
    //get the nbhd list for the node at this rank
    v = (igraph_vector_t* ) igraph_vector_ptr_e( &res, ii );
    for (myint j = 0; 
         j < igraph_vector_size( v ); 
         ++j) {
      //aa is a reverse reachable node from rank ii
      myint aa = igraph_vector_e( v, j );
      //push the rank of the ii'th node
      //onto the sketch of aa
      if (local_sketches[aa].size() < k)
        local_sketches[aa].push_back( instanceRanks[i][ii].first );
    }
  }
  //can deallocate res now
  IGRAPH_VECTOR_PTR_SET_ITEM_DESTRUCTOR( &res, &igraph_vector_destroy );
  igraph_vector_ptr_destroy_all( &res );
  igraph_vector_destroy( &vRankOrder );
   
  // merge local_sketches in instance i
  // into the global sketch for each node
  //get the lock first
  pthread_mutex_lock( &mutex1 );
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

  pthread_mutex_unlock( &mutex1 );
}





