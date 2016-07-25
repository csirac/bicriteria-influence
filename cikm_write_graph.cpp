#include <igraph.h>
#include "influence_oracles.cpp"
#include <vector>
#include <queue>
#include <iostream>
#include <random>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <ctime>
#include <pthread.h>
#include <bitset>
#define HEAD_INFO
#include "sfmt/SFMT.h"
#include "head.h"
class Argument{
public:
    int k;
    string dataset;
    double epsilon;
    string model;
    double T;
};
#include "graph.h"
#include "infgraph.h"
#include "imm.h"

using namespace std;

//global random number generator
std::random_device rd;
std::mt19937 gen(rd());

//global thread variables

unsigned nthreads;
igraph_t base_graph;
string graph_filename;
myint n;
myreal beta;
myreal alpha;
myreal int_maxprob;
myreal ext_maxprob;
string output_filename;
vector< myreal > IC_weights;
vector< myreal> node_probs;
myint ell;
myint max_dist;
influence_oracles my_oracles( 0, 0, 0 );

//function prototypes
void bicriteria_be2( influence_oracles& oracles, 
		     myint n, 
		     myint T,
		     myreal offset,
		     //out parameters
		     vector< myint >& seeds,
		     double alpha,
		     igraph_t& base_graph,
		     vector< myreal >& IC,
		     vector< myreal >& NP,
		     bool stop_criterion );

void bicriteria_better_est( influence_oracles& oracles, 
                 myint n, 
                 myint T,
                 myreal offset,
		 vector< myint >& seeds,
		 double alpha,
			    bool stop_criterion);
double monte_carlo( vector< myint >& seed_set, igraph_t& G, vector< myreal >& IC, vector< myreal >& NP, unsigned L);

void construct_independent_cascade( igraph_t& G, vector< myreal >& edge_weights,
				    myreal int_maxprob);
void construct_external_influence( igraph_t& G, vector< myreal >& node_probs, myreal max_prob );
myint kempe_greedy_TA( igraph_t& G, 
			 vector< myreal >& IC,
			vector< myreal >& NP,
			 vector< myint >& seed_set,
			 unsigned L, //number of samples
			double T // threshold
		       );

myreal construct_reachability_instance( igraph_t& G, 
				      vector< igraph_t* >& vgraphs,
				      vector< myreal >& IC, //IC weights
				      vector< myreal >& NP, //Node probabilities
				      myint ell );
void sample_independent_cascade( igraph_t& G, 
				 vector< myreal >& edgne_weights, 
				 igraph_t& sample_graph );

void sample_external_influence( igraph_t& G, 
				vector< myreal >& node_probs, 
				vector< myint >& nodes_sampled );

myint forwardBFS( igraph_t* G_i, vector< myint >& vseeds, vector< myint >& v_neighborhood );
myint forwardBFS( igraph_t* G_i, 
		  vector< myint >& vseeds, 
		  vector< myint >& vseeds2,
		  vector< myint >& v_neighborhood,
		  int max_distance );
void my_merge( vector< myint >& sk1, vector< myint >& sk2,
	       vector< myint >& res_sk, 
               myint k );
void my_merge( vector< myreal >& sk1, vector< myreal >& sk2,
	       vector< myreal >& res_sk, 
               myint k );

void* compute_oracles_online( void * );

void print_sketch( vector< myreal >& sk1 );

void bicriteria( influence_oracles& oracles, 
                 myint n, 
                 myint T,
                 myreal offset,
		 //out parameters
		 vector< myint >& seeds,
		 double alpha,
		 igraph_t& base_graph,
		 vector< myreal >& IC,
		 vector< myreal >& NP,
		 bool stop_criterion);

void my_merge2( vector< myint >& sk1, vector< myint >& sk2,
	       vector< myint >& res_sk, 
		myint k );

struct cmg_args {
  myint spos;
  myint fpos; //finish position + 1
  vector <myint> seeds;
  double mmarge;
  myint nnode;

};

void weight_vector_from_matrix(
			       igraph_t& G,
			       vector< double >& v_ew,
			       vector< vector < double > >& wA ) {
  myint n = igraph_vcount( &G );
  myint m = igraph_ecount( &G );
  myint eid;
  v_ew.assign( m, 1.0 );
  for (myint i = 0; i < n; ++i ) {
    for (myint j = 0; j < n; ++j ) {
      igraph_get_eid( &G, &eid, i, j, true, false );
      if (eid != -1) {
	//this edge exists, so let's assign the weight
	v_ew[ eid ] = wA[i][j];
      }
    }
  }
  
}

void read_weighted_graph( igraph_t& G, istream& ifile,
			  vector< vector< double > >& wA,
			  vector< double >& v_ew,
			  myint& n ) {
  cout << "Reading input file..." << endl;
  //vertex ids are from 0 to (n-1)
  bool is_directed; //, is_weighted;

  istringstream iss;
  string sline;

  myint nparent, nchild;
  double weight;
  igraph_vector_t edges_to_add;
  igraph_vector_init( &edges_to_add, 0 );

  vector< double > zero_weight;

  unsigned n_lines = 0;

  //create graph
  igraph_empty( &G, n, false );  //for now, undirected

  while( getline( ifile, sline ) ) {
    iss.clear();
    iss.str( sline );

    //skip comments
    if (sline[0] != '#') {
      if (n_lines == 0) {
	iss >> n;

	iss >> is_directed;

	zero_weight.assign( n, 0.0 );
	wA.assign( n, zero_weight );

      } else {
	iss >> nparent;
	iss >> nchild; 
	iss >> weight;
	wA[ nparent ][ nchild ] = weight;
	wA[ nchild ][ nparent ] = weight;

	igraph_vector_push_back( &edges_to_add, nparent );
	igraph_vector_push_back( &edges_to_add, nchild );
      }

      ++n_lines;

      if (n_lines > 15000) {
	n_lines = 1;
	igraph_add_edges( &G, &edges_to_add, 0 );
	igraph_vector_destroy( &edges_to_add );
	igraph_vector_init( &edges_to_add, 0 );
      }
    }

  }

  cout << "Finished reading file..." << endl;
  
  igraph_add_edges( &G, &edges_to_add, 0 );
  igraph_vector_destroy( &edges_to_add );

  igraph_simplify( &G, true, true, 0 ); //make sure we don't have a multigraph

  weight_vector_from_matrix( G, v_ew, wA );

  cout << "Graph constructed." << endl;
  
}

void *compute_marginal_gain( void* ptr ) {
  cmg_args* my_args = (cmg_args*) ptr;
  myint start = my_args -> spos;
  myint finish = my_args -> fpos;
  double t1, t2, t3, t4;
  vector <myint>& cmg_seed = my_args -> seeds;
  double curr_tau = my_oracles.better_estimate_cohen( cmg_seed, t1, t2 );
  //double curr_tau = my_oracles._cohen( cmg_seed, t1, t2 );

  cmg_seed.push_back( 0 );
  unsigned ss = cmg_seed.size();
  myint next_node = 0;
  double max_marg = 0.0;
  for (myint u = start; u < finish; ++u) {
    cmg_seed[ ss - 1 ] = u;
    double tmp_marg = my_oracles.better_estimate_cohen( cmg_seed, t3, t4 ) - curr_tau;

    if (tmp_marg > max_marg) {
      max_marg = tmp_marg;
      next_node = u;
    }
  }

  my_args -> nnode = next_node;
  my_args -> mmarge = max_marg;

}

// void remove_isolated_vertices( igraph_t* Graph ) {
//   igraph_vector_t v_deg0;
  
//   myint n_g = igraph_vcount( Graph );
//   igraph_vector_t res;
//   igraph_vector_init( &res, 0 );
//   for (myint i = 0; i < n_g; ++i) {
//     igraph_degree( Graph, &res, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS );
//     if (igraph_vector_e( &res, 0 ) == 0)
//       igraph_delete_vertices( Graph, igraph_vss_1( i ) );
//   }
//   cout << "here" << endl;

//   // igraph_vs_t vs;
//   // igraph_vs_vector( &vs, &v_deg0 );
//   // igraph_delete_vertices( Graph, vs );

//   // igraph_vs_destroy( &vs );
//   // igraph_vector_destroy( &res );
//   // igraph_vector_destroy( &v_deg0 );
// }

void write_imm_input( string imm_dir, 
		      igraph_t& G,
		      vector< myreal >& edge_weights ) {
  string s_attr = imm_dir + "attribute.txt";
  myint n = igraph_vcount( &G );
  myint m = igraph_ecount( &G );
  ofstream of_atr( s_attr.c_str() );
  of_atr << "n=" << n << endl;
  of_atr << "m=" << m << endl;
  of_atr.close();

  string s_graph = imm_dir + "graph_ic.inf";
  ofstream of_gr( s_graph.c_str() );
  myint from, to;
  for (myint i = 0; i < m; ++i ) {
    igraph_edge( &G, i, &from, &to );
    of_gr << from << ' ' << to << ' ' << edge_weights[ i ] << endl;
    of_gr << to << ' ' << from << ' ' << edge_weights[ i ] << endl;
  }
}

void write_weighted_graph(
			  string outfname, 
		      igraph_t& G,
		      vector< myreal >& edge_weights ) {
  igraph_simplify( &G, true, true, 0 ); //make sure we don't have a multigraph

  ofstream ofile( outfname.c_str() );
  myint n = igraph_vcount( &G );
  myint m = igraph_ecount( &G );

  myint from, to;
  for (myint i = 0; i < m; ++i ) {
    igraph_edge( &G, i, &from, &to );
    ofile << from << ' ' << to << endl;
  }

  ofile.close();
  outfname += "_w";
  ofstream ofile2( outfname.c_str() );
  for (myint i = 0; i < m; ++i) {
    ofile2 << edge_weights[i] << ' ';
  }
  ofile2.close();
}

bool bin_search( InfGraph& g, Argument& arg, int first, int last, int& res, igraph_t& base_graph, vector< myreal >& IC, vector< myreal >& NP ) {
  std::cout << first << ' ' << last << endl;
  if (last < first)
    return false;

  vector< myint > ss;

  g.init_hyper_graph(); 
  if (first == last) {
    arg.k = first;
    Imm::InfluenceMaximize(g, arg);
    //    ss.assign( g.seedSet.begin(), g.seedSet.end() );
    //    double infl = monte_carlo( ss, base_graph, IC, NP, 100 );
    double infl = g.InfluenceHyperGraph();
    if (infl >= arg.T) {
      res = first;
      return true;
    }
    else {
      return false;
    }
  }

  int mid = (first + last) / 2;
  arg.k = mid;
  Imm::InfluenceMaximize(g, arg);
  //  ss.assign( g.seedSet.begin(), g.seedSet.end() );
  //  double infl = monte_carlo( ss, base_graph, IC, NP, 100 );
  double infl = g.InfluenceHyperGraph();
  if ( infl >= arg.T ) {
    if ( bin_search( g, arg, first, mid - 1, res, base_graph, IC, NP ) ) {
      return true;
    } else {
      //the answer must be mid, couldn't find a k value excluding it
      res = mid;
      return true;
    }
  } else {
    return bin_search( g, arg, mid + 1, last, res, base_graph, IC, NP );
    
  }
}


void run_imm( string in_dir,
	      vector< myint >& ss,
	      double T,
	      igraph_t& G,
	      vector< myreal >& IC,
	      vector< myreal >& NP) {
  Argument arg;
  arg.dataset = in_dir;
  arg.epsilon = 0.1;
  arg.T = T;
  arg.k = 1;
  arg.model = "IC";

  string graph_file = arg.dataset + "graph_ic.inf";
  InfGraph g( arg.dataset, graph_file );
  g.setInfuModel( InfGraph::IC );

  int res;
  bin_search( g, arg, 1, T + 1, res, G, IC, NP );
  
  cout << "res = " << res << endl;
  arg.k = res;
  Imm::InfluenceMaximize( g, arg );

  cout << "IMM: " <<  g.InfluenceHyperGraph() << endl;
  ss.assign( g.seedSet.begin(), g.seedSet.end() );

}

void read_params(
		 string& which_estimator,
		 double& delta,
		 myint& N,
		 myint& max_dist,
		 bool& bdir,
		 myint& n,
		 unsigned& rp_nthreads,
		 string& graph_filename,
		 myreal& beta,
		 myreal& alpha,
		 myreal& int_maxprob,
		 myreal& ext_maxprob,
		 string& output_filename,
		 istream& is
		  ) {
  is >> graph_filename;
  if (graph_filename == "ER") {
    is >> n;
  }

  if (graph_filename == "BA") {
    is >> n;
  }

  is >> beta;
  is >> alpha;
  is >> int_maxprob;
  is >> ext_maxprob;
  is >> output_filename;
  is >> rp_nthreads;
  string sdir;
  is >> sdir;
  if (sdir == "true") {
    bdir = true;
  } else {
    bdir = false;
  }
  is >> max_dist;
  is >> N;
  is >> delta;
  is >> which_estimator;
}

myreal actual_influence(
			vector< myint >& seed_set,
			igraph_t& base_graph,
			vector< myreal >& IC,
			vector< myreal >& NP,
			unsigned L ) {
  myreal activated = 0.0;

  for (unsigned i = 0; i < L; ++i) {
    if (i % 1 == 0) {
      cout << "\r                                     \r"
    	   << ((myreal) i )/ L * 100 
    	//	   << i
    	   << "\% done";
      cout.flush();
    }

    igraph_t G_i;
    sample_independent_cascade( base_graph, IC, G_i );
    vector< myint > ext_act;
    sample_external_influence( base_graph, NP, ext_act );
    
    vector< myint > v_reach;
    activated += forwardBFS( &G_i,
		ext_act,
		seed_set,
		v_reach,
			     igraph_vcount( &G_i ) );
		//max_dist );
		
    //    activated += v_reach.size();
    //    cout << ext_act.size() + seed_set.size() << ' ' << v_reach.size() << endl;

    igraph_destroy( &G_i );
  }

  cout << "\r                              \r100% done" << endl;

  return activated / L;
}



int main(int argc, char** argv) {
  if (argc < 2) {
    //cerr << "Usage: " << argv[0] << " <input filename>\n";
    cerr << "Usage: "
	 << argv[0]
	 << " <graph filename>"
	 << " <beta>"
	 << " <alpha>"
	 << " <int_maxprob>"
	 << " <ext_maxprob>"
	 << " <output filename>"
      	 << " <nthreads>"
	 << " <is_directed>"
	 << " <max_dist>"
	 << " <N_trials>"
	 << " <delta> * 100"
	 << " <which_estimator>\n";
    return 1;
  }

  ifstream ifile;
  istringstream iss;
  myint N;
  bool bdir;
  double delta;
  string which_estimator;

  if (argc > 2) {
    //read parameters from command line
    string str_params;
    for (unsigned i = 1; i < argc; ++i) {
      str_params = str_params + " " + argv[i];
    }
    iss.str( str_params );

    read_params( which_estimator,
		delta, 
		N, //number of trials 
		max_dist,
		 bdir, 
		 n, nthreads, graph_filename,
		 beta,
		 alpha,
		 int_maxprob,
		 ext_maxprob,
		 output_filename,
		 iss );
  } else {
    
    if (argc == 2) {
      //read parameters from input file
      string str_ifile( argv[1] );
      ifile.open( str_ifile.c_str() );

      read_params( which_estimator,
		   delta,
		   N,
		   max_dist,
		   bdir, 
		   n, nthreads, graph_filename,
		   beta,
		   alpha,
		   int_maxprob,
		   ext_maxprob,
		   output_filename,
		   ifile );

    }

  }
  
  if (graph_filename == "ER") {
    igraph_erdos_renyi_game(&base_graph, 
			    IGRAPH_ERDOS_RENYI_GNP, 
			    n, 2.0 / n,
			    IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
  } else {
    if (graph_filename == "BA") {
      cout << "Constructing BA graph..." << endl;
      igraph_barabasi_game( &base_graph,
			    n,
			    1,
			    2.2,
			    NULL,
			    false,
			    1,
			    bdir,
			    IGRAPH_BARABASI_PSUMTREE,
			    NULL );

    } else {

      //ifstream ifile( graph_filename.c_str());
      FILE* fp;
      fp = fopen( graph_filename.c_str(), "r" );
      //if the graph is undirected
     
      igraph_read_graph_edgelist( &base_graph, fp, 0, bdir ); 
      fclose( fp );

      //
      //      remove_isolated_vertices( &base_graph );
      //      vector< vector < double > > wA; //unused
      //      read_weighted_graph( base_graph, ifile, wA,
      //			   IC_weights, n );
			   
    }
  }

  n = igraph_vcount( &base_graph );

  cout << "Base graph read from " << graph_filename
       << "...";

  if (bdir)
    cout << "is directed.";
  else
    cout << "is undirected.";
      
  cout << endl;

  
  cout << "n = " << n << endl;
  cout << "m = " << igraph_ecount( &base_graph ) << endl;
  myint T;
  if (beta > 1.0) {
    T = beta;
  } else {
    T = beta * n;
  }
  if (max_dist == 0) 
    max_dist = igraph_vcount( &base_graph );

  myint C = 2;
  myint K = 1.0 / (C * alpha);
  delta /= 100;
  ell = log( 2 / delta ) / (alpha * alpha) / 2;
  //  ell = 0.7 * n;
  myint k_cohen = ell;// (myint)( log ( ((double) n) ) );//ell / 2.0;//;//ell / 2.0;//25 * ;//ell;//((double) ell);//25 * 
  //ell = ell / 3.0;

  myreal epsilon = alpha * n;
  cout << "epsilon = " << epsilon << endl;
  cout << "alpha * T = " << alpha * T << endl;
  cout << "k_cohen = " << k_cohen << endl;
  cout << "T = " << T << endl;
  cout << "alpha = " << alpha << endl;
  cout << "beta = " << beta << endl;
  cout << "ell = " << ell << endl;
  cout << "K = " << K << endl;
  cout << "int_maxprob = " << int_maxprob << endl;
  cout << "ext_maxprob = " << ext_maxprob << endl;
  cout << "N = " << N << endl;

  cout << "Minimum memory required: " 
       << k_cohen << 'x'
       << sizeof( myreal ) << 'x'
       << n << " = " << k_cohen * sizeof( myreal ) * n / 1000000.0 
       << " MB" << endl;
    

  system("sleep 2");

  //run the simulation N times
  vector< myint > seed_set;
  double avg_seed_set_size = 0.0;
  double avg_t_bicriteria = 0.0;
  double avg_t_oracle = 0.0;
  double avg_t_total = 0.0;
  double avg_act = 0.0;
  double avg_offset = 0.0;

  for (unsigned nsim = 0; nsim < N; ++nsim) {

    cout << "Constructing the IC model..." << endl;

    construct_independent_cascade( base_graph, IC_weights, int_maxprob );

    write_weighted_graph( output_filename, base_graph, IC_weights );

    exit( 0 );

    //this is a simple model of external influence
    cout << "Constructing external influence..." << endl;
    construct_external_influence( base_graph, node_probs, ext_maxprob );

    cerr << "Computing oracles online...\n";

    //  vector< igraph_t* > v_graphs; // the ell graphs
    
    // double offset = construct_reachability_instance( 
    //                                  base_graph, 
    //       			   v_graphs, 
    //       			   IC_weights,
    //       			   node_probs,
    //       			   ell );
    
    clock_t t_start = clock();
    //create the oracles

    my_oracles.n = n;
    my_oracles.ell = ell;
    my_oracles.k = k_cohen;
    my_oracles.or_max_dist = max_dist;
    my_oracles.compute_uniform_oracles_online_init();
      
    myint ell_tmp = ((double) ell) / nthreads;
    if (which_estimator != "naive" &&
	which_estimator != "imm" ) {      
      
      pthread_t* mythreads = new pthread_t[ nthreads ];
      for (unsigned i = 0; i < nthreads; ++i) {
	pthread_create( &( mythreads[i] ), NULL, compute_oracles_online, &ell_tmp );
	
      }

      for (unsigned i = 0; i < nthreads; ++i) {
	pthread_join( mythreads[i] , NULL );
	
      }
      
      delete [] mythreads;
    }
    clock_t t_finish = clock();
    myreal t_oracle = myreal ( t_finish - t_start ) / CLOCKS_PER_SEC;
    avg_t_oracle += t_oracle;

    //run the bicriteria alg.
    t_start = clock();
    seed_set.clear();
    string imm_dir = "./tmp_imm/";
    if (which_estimator == "imm") {
      write_imm_input( imm_dir, base_graph, IC_weights );
    }

    myreal offset = my_oracles.offset / my_oracles.ell;
    avg_offset += offset;
     
    if (which_estimator == "simple") {
      bicriteria( my_oracles, n, T, offset,
		  seed_set, alpha, 
		  base_graph, 
		  IC_weights,  node_probs,
		  true );
    } else {
      if (which_estimator == "complex") {
	bicriteria_better_est( my_oracles, n, T, offset,
				       seed_set, alpha, false );

      } else {

	if (which_estimator == "naive") {
	  kempe_greedy_TA( base_graph, IC_weights, node_probs, 
			   seed_set, 1000, T );
	} else {
	  if (which_estimator == "imm") {
	    run_imm( imm_dir, seed_set, T, base_graph,
		     IC_weights, node_probs );
	  } else {
	    if (which_estimator == "complex2") {
	      	bicriteria_be2( my_oracles, n, T, offset,
		    seed_set, alpha, 
		    base_graph, 
		    IC_weights,  node_probs,
		    false );
	    } else {
	      if (which_estimator == "all") {
		

	      }
	    }
	  }
	}
      }
    }

    t_finish = clock();
    myreal t_bicriteria = myreal (t_finish - t_start) / CLOCKS_PER_SEC;
    avg_t_bicriteria += t_bicriteria;
    myreal t_total = t_oracle + t_bicriteria;
    avg_t_total += t_total;

    
    cout << "Size of seed set: " << seed_set.size() << endl;
    avg_seed_set_size += seed_set.size();

    cout << "Finished in: " << t_total << " seconds" << endl;

    //compute "actual" influence of seed set
    cout << "Estimating influence of seed set by Monte Carlo..." << endl;
    myreal act_infl = actual_influence( seed_set, base_graph, IC_weights, node_probs, 1000U );
    cout << act_infl << endl;
    avg_act += act_infl;
  }

  avg_seed_set_size /= N;
  avg_t_bicriteria /= N;
  avg_t_oracle /= N;
  avg_t_total /= N;
  avg_act /= N;
  avg_offset /= N;

  igraph_destroy( &base_graph);

  ofstream ofile( output_filename.c_str(), 
		     ofstream::app );
  ofile << n << ' '
	<< beta << ' '
	<< T << ' '
	<< alpha << ' '
	<< epsilon << ' '
	<< ell << ' '
	<< k_cohen << ' '
	<< int_maxprob << ' '
	<< ext_maxprob << ' '
	<< avg_offset << ' '
	<< avg_seed_set_size << ' '
	<< avg_act << ' ' 
	<< avg_t_bicriteria << ' '
	<< avg_t_oracle << ' '
	<< avg_t_total << ' '
	<< N << ' '
	<< (T - avg_act) / (alpha * T) << ' '
	<< (T - avg_act) / epsilon << ' '
	<< max_dist << ' '
	<< delta << ' '
	<< which_estimator << endl;

  ofile.close();
  return 0;
}

void print_sketch( vector< myint >& sk1 ) {
  for (myint i = 0; i < sk1.size(); ++i) {
    cout << sk1[i] << ' ';
  }

  cout << endl;
}

void print_sketch( vector< myreal >& sk1 ) {
  for (myint i = 0; i < sk1.size(); ++i) {
    cout << sk1[i] << ' ';
  }

  cout << endl;
}


class compare_pair
{
public:
  bool operator()( npair a, npair b )
  {
    return (a.nrank < b.nrank);
  }
};

void bicriteria_be2
( influence_oracles& oracles, 
                 myint n, 
                 myint T,
                 myreal offset,
		 //out parameters
		 vector< myint >& seeds,
		 double alpha,
		 igraph_t& base_graph,
		 vector< myreal >& IC,
		 vector< myreal >& NP,
		 bool stop_criterion = false ) {
  cerr << "offset = " << offset << endl;
  seeds.clear();

  myreal est_infl = offset;
  myreal max_marg = 0.0;
  myint next_node = 0;
  myreal curr_tau = 0.0;

  bool bcont = true;
  unsigned n_iter = 0;
  priority_queue< npair, vector< npair >, compare_pair > Q;
  
  for (myint i = 0; i < n; ++i) {
    npair tmp;
    tmp.node = i;
    tmp.nrank = n + 1;

    Q.push( tmp );
  }

  vector< bool > b_valid;

  while (bcont ) {
    ++n_iter;
    //select the node with the max. marginal gain
    //for each u
    b_valid.assign( n, false );
    curr_tau = oracles.imp_est.estimate / oracles.ell ; 

    while( true ) {
      npair pmax = Q.top();
      Q.pop();
      if ( b_valid[ pmax.node ] ) {
	oracles.imp_est.add_node( pmax.node );
	seeds.push_back( pmax.node );
	break;
      } else {
	myreal marg = oracles.imp_est.add_node_dry( pmax.node ) / oracles.ell - curr_tau;
	pmax.nrank = marg;
	Q.push( pmax );
	b_valid[ pmax.node ] = true;
      }
    }
    
    est_infl = offset + oracles.imp_est.estimate / oracles.ell;

    cerr << est_infl << ' ' << curr_tau << endl;
    bcont = (est_infl < T);

  }
}


void bicriteria( influence_oracles& oracles, 
                 myint n, 
                 myint T,
                 myreal offset,
		 //out parameters
		 vector< myint >& seeds,
		 double alpha,
		 igraph_t& base_graph,
		 vector< myreal >& IC,
		 vector< myreal >& NP,
		 bool stop_criterion = false ) {

  cerr << "offset = " << offset << endl;

  //  vector< myint > sketch;
  vector< myreal > sketch;

  myreal est_infl = offset;
  myreal max_marg = 0.0;
  myint next_node = 0;
  myreal curr_tau = 0.0;

  //  vector< myint > tmp_sketch;
  vector< myreal > tmp_sketch;
  bool bcont = true;
  unsigned n_iter = 0;
  while (bcont ) {
    ++n_iter;
    //select the node with the max. marginal gain
    //for each u

    max_marg = 0.0;
    //    curr_tau = oracles.estimate_reachability_sketch( sketch );
        curr_tau = oracles.estimate_reachability_uniform_sketch( sketch );
    for (myint u = 0; u < n; ++u) {
      //tmp_sketch = merge( sketch, sketch_u )
      my_merge( sketch, oracles.uniform_global_sketches[ u ], tmp_sketch, oracles.k );
      
      myreal tmp_marg = oracles.estimate_reachability_uniform_sketch( tmp_sketch ) - curr_tau;
      if (tmp_marg > max_marg) {
	max_marg = tmp_marg;
	next_node = u;
      }
    }

    my_merge( sketch, oracles.uniform_global_sketches[ next_node ], tmp_sketch, oracles.k );
    sketch.swap( tmp_sketch );

    seeds.push_back( next_node );

    est_infl = offset + curr_tau + max_marg;// + seeds.size();

    cerr << est_infl << ' ' << curr_tau << ' ' <<  max_marg << ' ' << next_node << endl;
    cerr << "perm rank " << sketch.back() << endl;

    if (stop_criterion) {
      //      if (max_marg < 2.0) { //alpha * T) {
	//we're getting a small gain, so stop
      //	cout << "Getting worse ratio" << endl;
      //	break;
      //    }
      if ( est_infl > T ) {
	if ( n_iter % 10 == 0  ) {
	  myreal act_infl = actual_influence( seeds, base_graph, IC, NP, 10 );
	  cout << act_infl << endl;
	  if ( act_infl > T ) {
	    bcont = false;
	  }
	}
      }
    } else {
	bcont = (est_infl < T);
    }
  }
}


void bicriteria_better_est( influence_oracles& oracles, 
                 myint n, 
                 myint T,
                 myreal offset,
		 vector< myint >& seeds,
		 double alpha,
		 bool stop_criterion = false) {
  seeds.clear();
  myreal est_infl = offset;
  myreal max_marg = 0.0;
  myint next_node = 0;
  myreal curr_tau = 0.0;
  myint size_of_seeds = 0;
  double t1, t2;
  while (est_infl < T ) {

    //select the node with the max. marginal gain
    //for each u
    curr_tau = oracles.better_estimate_cohen( seeds, t1, t2 );
    //    curr_tau = oracles.imp_est.estimate;

    pthread_t* mythreads = new pthread_t [ nthreads ];
    vector< cmg_args* > v_thread_results;
    cmg_args* thread_args;
    for (unsigned v = 0; v < nthreads; ++v) {
      thread_args = new cmg_args;
      double dfactor = ( (double) n ) / nthreads;
      myint spos = dfactor * v;
      myint fpos = dfactor * v + dfactor;

      if (v == nthreads - 1) {
	fpos = n;
      }

      thread_args -> spos = spos;
      thread_args -> fpos = fpos;
      (thread_args -> seeds).assign( seeds.begin(), seeds.end() );

      void * tptr = (void *) thread_args;
      pthread_create( &( mythreads[ v ] ), NULL, compute_marginal_gain, tptr );
      
      v_thread_results.push_back( thread_args );
    }

    //wait for the threads to finish
    for (unsigned i = 0; i < nthreads; ++i) {
      pthread_join( mythreads[i] , NULL );
    
    }

    delete [] mythreads;
    
    //find the max. marginal gain returned by the threads
    max_marg = 0.0;
    next_node = 0;
    for (unsigned i = 0; i < nthreads; ++i) {
      thread_args = v_thread_results[i];
      double tmp_marg = thread_args -> mmarge;
      if (tmp_marg > max_marg) {
	max_marg = tmp_marg;
	next_node = thread_args -> nnode;
      }

      delete (v_thread_results[i]);
    }
    ++size_of_seeds;
    seeds.push_back( 0 );
    seeds[ size_of_seeds - 1 ] = next_node;

    est_infl = offset + curr_tau + max_marg;// + seeds.size();

    cerr << est_infl << ' ' << curr_tau << ' ' <<  max_marg << ' ' << next_node << endl;

    if (stop_criterion) {
      if (max_marg < 2.0) { //alpha * T) {
	//we're getting a small gain, so stop
	cout << "Getting worse ratio" << endl;
	break;
      }
    }
  }


}

void my_merge2( vector< myint >& sk1, vector< myint >& sk2,
	       vector< myint >& res_sk, 
               myint k ) {
  //this is not really a merge, seems to be what cohen is saying
  //in the 1997 paper. Rather is coordinate-wise min.
  
  vector< myint >::iterator it1 = sk1.begin();
  vector< myint >::iterator it2 = sk2.begin();

  myint l = sk1.size();
  if (sk2.size() > l)
    l = sk2.size();

  res_sk.assign( l, 0 );

  vector< myint >::iterator res_it = res_sk.begin();

  myint s = 0; //size

  while (s < l) {
    //add the smallest of coordinates
    if (it1 != sk1.end()) {
      if (it2 != sk2.end()) {
	if ( (*it1) < (*it2) ) {
	  (*res_it) = (*it1);
	  ++it1;
	  ++it2;
	} else {
	  if ( (*it1) > (*it2) ) {
	    (*res_it) = (*it2);
	    ++it2;
	    ++it1;
	  } else {
	    //the values are equal
	    //but we can only have
	    //an element appear once
	    //so insert it, and increment both
	    (*res_it) = (*it1);
	    ++it1;
	    ++it2;
	  }
	}
      } else { //we're done with sketch2.
	//just add from sk1
	(*res_it) = (*it1);
	++it1;
      }
    } else {
      //done with sk1,
      //add from sk2
      (*res_it) = (*it2);
      ++it2;
    }

    ++res_it;
    ++s;
  }

}

void my_merge( vector< myint >& sk1, vector< myint >& sk2,
	       vector< myint >& res_sk, 
               myint k ) {
  //	       , vector< myint >& seeds ) {

  vector< myint >::iterator it1 = sk1.begin();
  vector< myint >::iterator it2 = sk2.begin();

  myint l = k;
  if (l > (sk1.size() + sk2.size())) {
    l = sk1.size() + sk2.size();
  }

  res_sk.assign( l, 0 );
  vector< myint >::iterator res_it = res_sk.begin();

  myint s = 0; //size

  while (s < l) {
    //add the smallest next one
    if (it1 != sk1.end()) {
      if (it2 != sk2.end()) {
	if ( (*it1) < (*it2) ) {
	  (*res_it) = (*it1);
	  ++it1;
	} else {
	  if ( (*it1) > (*it2) ) {
	    (*res_it) = (*it2);
	    ++it2;
	  } else {
	    //the values are equal
	    //but we can only have
	    //an element appear once
	    //so insert it, and increment both
	    (*res_it) = (*it1);
	    ++it1;
	    ++it2;
	  }
	}
      } else { //we're done with sketch2.
	//just add from sk1
	(*res_it) = (*it1);
	++it1;
      }
    } else {
      //done with sk1,
      //add from sk2
      (*res_it) = (*it2);
      ++it2;
    }

    ++res_it;
    ++s;
  }


}
void my_merge( vector< myreal >& sk1, vector< myreal >& sk2,
	       vector< myreal >& res_sk, 
               myint k ) {
  //	       , vector< myint >& seeds ) {

  vector< myreal >::iterator it1 = sk1.begin();
  vector< myreal >::iterator it2 = sk2.begin();

  myint l = k;
  if (l > (sk1.size() + sk2.size())) {
    l = sk1.size() + sk2.size();
  }

  res_sk.assign( l, 0 );
  vector< myreal >::iterator res_it = res_sk.begin();

  myint s = 0; //size

  while (s < l) {
    //add the smallest next one
    if (it1 != sk1.end()) {
      if (it2 != sk2.end()) {
	if ( (*it1) < (*it2) ) {
	  (*res_it) = (*it1);
	  ++it1;
	} else {
	  if ( (*it1) > (*it2) ) {
	    (*res_it) = (*it2);
	    ++it2;
	  } else {
	    //the values are equal
	    //but we can only have
	    //an element appear once
	    //so insert it, and increment both
	    (*res_it) = (*it1);
	    ++it1;
	    ++it2;
	  }
	}
      } else { //we're done with sketch2.
	//just add from sk1
	(*res_it) = (*it1);
	++it1;
      }
    } else {
      //done with sk1,
      //add from sk2
      (*res_it) = (*it2);
      ++it2;
    }

    ++res_it;
    ++s;
  }


}



//takes place of 'construct_reachability_instance'
//does not store any of the reachability graphs
//works with the oracle online functions
void *compute_oracles_online( void* ptr ) {
  igraph_t& G = base_graph;
  vector< myreal >& IC = IC_weights;
  vector< myreal >& NP = node_probs;
  influence_oracles& O = my_oracles;

  myint N = *( (myint*) ptr );

  myint n = igraph_vcount( &G );
  myreal offset = 0.0;
  for (myint i = 0; i < N; ++i) {
    if (i % 1 == 0) {
      cout << "\r                                     \r"
	   << ((myreal) i )/ N * 100 
	//	   << i
	   << "\% done";
      cout.flush();
    }

    igraph_t G_i; 

    sample_independent_cascade( G, IC, G_i );
    //G_i now has an IC instance. Let's select the
    //externally activated nodes
    vector< myint > ext_act;
    //need to initialize ext_act
    sample_external_influence( G, NP, ext_act );
    //remove the reachable set in G_i, of the externally activated set.
    //actually, what we want is to remove all 
    //edges incident to reachable
    //set.
    vector< myint > v_reach;

    igraph_vector_t v_edges_to_remove;
    igraph_vector_init( &v_edges_to_remove, 0 );
    igraph_vector_t v_tmp;
    igraph_vector_init( &v_tmp, 0 );

    forwardBFS( &G_i, ext_act, ext_act, v_reach, max_dist );

    offset = v_reach.size();
    bitset< MAX_N > bs_ext;
  
    for (myint iii = 0; iii < v_reach.size(); ++iii) {
      igraph_incident( &G_i, &v_tmp, v_reach[iii], IGRAPH_ALL );

      igraph_vector_append( &v_edges_to_remove, &v_tmp );
      bs_ext.set( v_reach[ iii ] );
    }
    igraph_vector_destroy(&v_tmp);

    igraph_es_t es; //vertex selector for the reachable set to remove
    igraph_es_vector( &es, &v_edges_to_remove );
    igraph_delete_edges( &G_i, es );
    igraph_es_destroy( &es );
    igraph_vector_destroy( &v_edges_to_remove );

    //All edges incident to the reachable 
    //set have been removed
    //That is, H_i has been created
    
    //use it for the oracle computation,
    //but do then discard it

    O.compute_uniform_oracles_online_step( &G_i, i, offset,
					   bs_ext );
    igraph_destroy( &G_i );
  }

  cout << "\r                               \r100% done" << endl;



}


myreal construct_reachability_instance( igraph_t& G, 
				      vector< igraph_t* >& vgraphs,
				      vector< myreal >& IC, //IC weights
				      vector< myreal >& NP, //Node probabilities
				      myint ell ) {
  //create a gen. reachability instance
  vgraphs.clear();

  myint n = igraph_vcount( &G );
  myreal offset = 0.0;
  for (myint i = 0; i < ell; ++i) {
    if (i % 1 == 0) {
      cerr << "\r                                     \r"
	   << ((myreal) i )/ ell * 100 
	//	   << i
	   << "\% done";
    }

    igraph_t* G_i = new igraph_t; //want a new address in memory each iteration    

    sample_independent_cascade( G, IC, *G_i );
    //G_i now has an IC instance. Let's select the
    //externally activated nodes
    vector< myint > ext_act;
    //need to initialize ext_act
    sample_external_influence( G, NP, ext_act );
    //remove the reachable set in G_i, of the externally activated set.
    //actually, what we want is to remove all edges incident to reachable
    //set.
    vector< myint > v_reach;

    igraph_vector_t v_edges_to_remove;
    igraph_vector_init( &v_edges_to_remove, 0 );
    igraph_vector_t v_tmp;
    igraph_vector_init( &v_tmp, 0 );

    //    cerr << "\n" << ' ';
    forwardBFS( G_i, ext_act, v_reach );

    offset += v_reach.size();

    for (myint i = 0; i < v_reach.size(); ++i) {

      igraph_incident( G_i, &v_tmp, v_reach[i], IGRAPH_ALL );

      igraph_vector_append( &v_edges_to_remove, &v_tmp );

    }

    //cerr << igraph_vector_size( &v_edges_to_remove ) << "\n";

    igraph_vector_destroy(&v_tmp);

    igraph_es_t es; //vertex selector for the reachable set to remove
    igraph_es_vector( &es, &v_edges_to_remove );
    igraph_delete_edges( G_i, es );
    igraph_es_destroy( &es );
    igraph_vector_destroy( &v_edges_to_remove );

    //    cerr << "BFS done\n";
    //All edges incident to the reachable set 
    //have been removed
    //That is, H_i has been created
    vgraphs.push_back( G_i );
  }

  cout << "\r                               \r100% done" << endl;

  return offset / ell;
}





int test_estimation()
{
  igraph_integer_t diameter;
  igraph_t graph;
  igraph_rng_seed(igraph_rng_default(), 42);
  myint ell = 10;
  myint n = 100000;
  vector < igraph_t > vgraphs( ell, graph );

  for (myint i = 0; i < ell; ++i) {
    igraph_erdos_renyi_game(&(vgraphs[i]), IGRAPH_ERDOS_RENYI_GNP, n, 1.0/ n,
			  IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
  }

  influence_oracles my_oracles( vgraphs, ell, (myint) 10, n );

  cerr << "computing oracles...\n";

  my_oracles.compute_oracles();

  cerr << "done" << endl;

  cout << "avg reachability and estimates: ";
  for (myint i = 0; i < n; ++i) {
    cout << i << ' ' << my_oracles.average_reachability( i ) << ' ' << my_oracles.estimate_reachability( i ) << endl;
  }

  return 0;
}



void construct_independent_cascade( igraph_t& G, vector< myreal >& edge_weights, myreal int_maxprob) {
  edge_weights.clear();

  std::uniform_real_distribution<> dis(0, int_maxprob);

  myint m = igraph_ecount( &G );

  for (myint i = 0; i < m; ++i) {
    edge_weights.push_back( dis(gen) );
  }

}

void construct_external_influence( igraph_t& G, vector< myreal >& node_probs, myreal max_prob ) {
  node_probs.clear();
  myint n = igraph_vcount( &G );
  std::uniform_real_distribution<> dis(0, max_prob );

  for (myint i = 0; i < n; ++i) {
    node_probs.push_back( dis(gen) );
  }
}

void sample_independent_cascade( igraph_t& G, vector< myreal >& edge_weights, igraph_t& sample_graph ) {
  
  igraph_copy( &sample_graph, &G );

  std::uniform_real_distribution<> dis(0, 1);

  myint m = igraph_ecount( &G );

  myreal cvalue;
  igraph_vector_t edges_to_delete;
  igraph_vector_init( &edges_to_delete, 0L );
  igraph_vector_reserve( &edges_to_delete, m );

  for (myint i = 0; i < m; ++i ) {
    //toss a coin
    cvalue = dis(gen);
    if (cvalue < edge_weights[i]) {
      //keep edge i
    } else {
      //delete edge i
      igraph_vector_push_back( &edges_to_delete, i );
    }
  }

  igraph_es_t es; //edge selector of edges to destroy
  igraph_es_vector( &es, &edges_to_delete );

  //actually delete the edges from the sample graph
  igraph_delete_edges( &sample_graph, es );

  igraph_es_destroy( &es );
  igraph_vector_destroy( &edges_to_delete );
  
}

void sample_external_influence( igraph_t& G, 
				vector< myreal >& node_probs, 
				vector< myint >& nodes_sampled ) {
  

  std::uniform_real_distribution<> dis(0, 1);

  myint n = igraph_vcount( &G );

  myreal cvalue;

  nodes_sampled.clear();

  for (myint i = 0; i < n; ++i ) {
    //toss a coin
    cvalue = dis(gen);
    if (cvalue < node_probs[i]) {
      //sample this node i
      //igraph_vector_push_back( &nodes_sampled, i );
      nodes_sampled.push_back( i );
    } 
  }

}

myint forwardBFS( igraph_t* G_i, 
		  vector< myint >& vseeds, 
		  vector< myint >& v_neighborhood ) {
  queue <myint> Q;
  vector < int > dist;
  myint n = igraph_vcount( G_i );
  dist.reserve( n );
  for (myint i = 0; i < n; ++i) {
    dist.push_back( -1 ); //infinity
  }

  //  igraph_vector_t v_edges_to_remove;
  //  igraph_vector_init( &v_edges_to_remove, 0);

  for (myint i = 0; i < vseeds.size(); ++i) {
    dist[ vseeds[i] ] = 0;
    Q.push( vseeds[i] );
    //    igraph_vector_push_back( &v_neighborhood, vseeds[i] );
    v_neighborhood.push_back( vseeds[i] );
  }

  while (!(Q.empty())) {

    myint current = Q.front();
    Q.pop();
    //get forwards neighbors of current
    igraph_vector_t neis;
    igraph_vector_init( &neis, 0 );
    igraph_neighbors( G_i, &neis, current, IGRAPH_OUT );
    for (myint i = 0; i < igraph_vector_size( &neis ); ++i) {
      myint aneigh = VECTOR( neis )[i];
      if (dist[ aneigh ] == -1 ) { //if aneigh hasn't been discovered yet
	dist[ aneigh ] = dist[ current ] + 1;
	Q.push( aneigh );
	//	igraph_vector_push_back( &v_neighborhood, aneigh );
	v_neighborhood.push_back( aneigh );

	//flag this edge for removal from the graph
	//	myint eid;
	//	igraph_get_eid( G_i, &eid, current, 
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

myint forwardBFS( igraph_t* G_i, 
		  vector< myint >& vseeds, 
		  vector< myint >& vseeds2,
		  vector< myint >& v_neighborhood,
		  int max_distance ) {
  queue <myint> Q;
  vector < int > dist;
  myint n = igraph_vcount( G_i );
  dist.reserve( n );
  for (myint i = 0; i < n; ++i) {
    dist.push_back( -1 ); //infinity
  }

  //  igraph_vector_t v_edges_to_remove;
  //  igraph_vector_init( &v_edges_to_remove, 0);

  for (myint i = 0; i < vseeds.size(); ++i) {
    if (dist[ vseeds[i] ] != 0) {
      dist[ vseeds[i] ] = 0;
      Q.push( vseeds[i] );
      //    igraph_vector_push_back( &v_neighborhood, vseeds[i] );
      v_neighborhood.push_back( vseeds[i] );
    }
  }

  for (myint i = 0; i < vseeds2.size(); ++i) {
    if (dist[ vseeds2[i] ] != 0) {
      dist[ vseeds2[i] ] = 0;
      Q.push( vseeds2[i] );
      v_neighborhood.push_back( vseeds2[i] );
    }
  }

  while (!(Q.empty())) {

    myint current = Q.front();
    Q.pop();
    //get forwards neighbors of current
    igraph_vector_t neis;
    igraph_vector_init( &neis, 0 );
    igraph_neighbors( G_i, &neis, current, IGRAPH_OUT );
    for (myint i = 0; i < igraph_vector_size( &neis ); ++i) {
      myint aneigh = VECTOR( neis )[i];

      if ( (dist[ aneigh ] == -1) || (dist[ aneigh ] > dist[ current ] + 1) ) { //if aneigh hasn't been discovered yet

	if ((dist[ aneigh ] == -1) && ((dist[ current ] + 1) < max_distance))
	  Q.push( aneigh );

	if ((dist[ current ] + 1) <= max_distance ) {
	  if ((dist[ aneigh ] > max_distance) || (dist[ aneigh ] == -1))
	    v_neighborhood.push_back( aneigh );

	}
	
	dist[ aneigh ] = dist[ current ] + 1;
      } 
    }

    igraph_vector_destroy( &neis );
  } //BFS finished

  myint count = v_neighborhood.size();

  return count;

}
myint kempe_greedy_TA( igraph_t& G, 
			 vector< myreal >& IC,
			vector< myreal >& NP,
			 vector< myint >& seed_set,
			 unsigned L, //number of samples
			double T // threshold
		     ) {
  cout << "starting CELF..." << endl;
  seed_set.clear();
  myint n = igraph_vcount( &G );
  
  double curr_act = 0.0;
  double tmp_marge = 0.0;
  double max_marge = 0.0;
  priority_queue< npair, vector< npair >, compare_pair > Q;
  
  for (myint i = 0; i < n; ++i) {
    npair tmp;
    tmp.node = i;
    tmp.nrank = n + 1;

    Q.push( tmp );
  }

  myint nn = 0;
  vector< bool > b_valid;
  bool bcont = true;
  curr_act = monte_carlo( seed_set, G, IC, NP, L );
  while ( bcont ) {
    b_valid.assign( n, false );
    while( true ) {
      npair pmax = Q.top();
      Q.pop();
      if ( b_valid[ pmax.node ] ) {
	seed_set.push_back( pmax.node );
	break;
      } else {
	seed_set.push_back( pmax.node );
	myreal marg = monte_carlo( seed_set, G, IC, NP, L )  - curr_act;
	pmax.nrank = marg;
	Q.push( pmax );
	b_valid[ pmax.node ] = true;
	seed_set.pop_back();
      }

    }
    curr_act = monte_carlo( seed_set, G, IC, NP, L );
    cout << curr_act << endl;

    bcont = ( curr_act < T );
  }

}

double monte_carlo( vector< myint >& seed_set, igraph_t& G, vector< myreal >& IC, vector< myreal >& NP, unsigned L) {
  vector <myint> vreach;
  vector <myint> ext;
  double tmp_eact = 0.0;
  for (unsigned i = 0; i < L; ++i) {
    vreach.clear();
    ext.clear();
    igraph_t* G_i = new igraph_t;
    sample_independent_cascade(G, IC, *G_i );
    sample_external_influence( G, NP, ext );
    forwardBFS( G_i, seed_set, ext, vreach, igraph_vcount( &G) );
    tmp_eact += vreach.size();
    igraph_destroy( G_i );
  }

  return tmp_eact / L;
}


double kempe_greedy_max( igraph_t& G, 
			 vector< myreal >& IC,
			 vector< myint >& seed_set,
			 unsigned L, //number of samples
			 unsigned kk //number of seed nodes
		     ) {
  double eact = 0;
  seed_set.clear();
  seed_set.reserve( kk );
  double max_marge;
  vector< myint > v_reach;

  igraph_t* G_i;

  for (unsigned iter = 0; iter < kk; ++iter) {

    seed_set.push_back( 0 ); // we're going to be adding a new
    //seed node, identity to be determined
    myint next_node = 0;
    max_marge = 0.0;
    //try all nodes
    for (unsigned i = 0; i < n; ++i) {

      double tmp_eact = 0.0;
      for (unsigned j = 0; j < L; ++j) {
	v_reach.clear();
	G_i = new igraph_t;
	sample_independent_cascade(G, IC, *G_i );
	seed_set[ iter ] = i; // test the ith seed node
	forwardBFS( G_i, seed_set, v_reach );
	tmp_eact += v_reach.size();
	igraph_destroy( G_i );
      }

      tmp_eact /= L;
      double marge_i = tmp_eact - eact;
      if (marge_i > max_marge) {
	max_marge = marge_i;
	next_node = i;
      }
    }

    //we've found node with max marginal gain
    seed_set[iter] = next_node;
    eact += max_marge;
  }

  return eact;
}
