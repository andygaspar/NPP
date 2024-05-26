#include "genetic_pso.h"


extern "C" {




    GeneticPso* GeneticPso_(double* upper_bounds_, double* lower_bounds_, double* comm_tax_free, int* n_usr, double* trans_costs, short n_commodities_, short n_paths_, 
                        short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
                        short pso_size_, short pso_selection_, short pso_every_, short pso_iterations_, short pso_final_iterations_, short pso_no_update_lim_,
                        short num_threads_, bool verbose_, short seed) {
        return new GeneticPso(upper_bounds_, lower_bounds_, comm_tax_free, n_usr, trans_costs, n_commodities_, n_paths_, 
                        pop_size_, off_size_, mutation_rate_, recombination_size_, 
                        pso_size_, pso_selection_, pso_every_, pso_iterations_, pso_final_iterations_, pso_no_update_lim_,
                        num_threads_, verbose_, seed);
    }

    void run_genetic_(GeneticPso* g, double* population, int iterations) {g -> run(population,iterations);}
    double get_gen_best_val_(GeneticPso* g) {return g -> get_best_val();}
    double* get_population_ (GeneticPso* g) {return g -> get_population();}
    double* get_vals_ (GeneticPso* g) {return g-> get_vals();}
    void destroy(GeneticPso* g) {delete g;}

}