#include "arc_genetic.h"


extern "C" {


    ArcGenetic* ArcGenetic_(double* upper_bounds_, double* lower_bounds_, double* adj_, int adj_size_, int* tolls_idxs_, 
                            int* n_usr, int* origins_, int* destinations_, short n_commodities_, short n_tolls_, 
                            short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
                            short num_threads_, bool verbose_, short seed) {
        return new ArcGenetic(upper_bounds_, lower_bounds_, adj_, adj_size_, tolls_idxs_, n_usr, origins_, destinations_, n_commodities_, n_tolls_, 
                                pop_size_, off_size_, mutation_rate_, recombination_size_, 
                                num_threads_, verbose_, seed);
    }

    void run_arc_genetic_(ArcGenetic* g, double* population, int iterations) {g -> run(population,iterations);}
    double get_gen_best_val_(ArcGenetic* g) {return g -> get_best_val();}
    double* get_population_ (ArcGenetic* g) {return g -> get_population();}
    double* get_vals_ (ArcGenetic* g) {return g-> get_vals();}
    void destroy(ArcGenetic* g) {delete g;}

}