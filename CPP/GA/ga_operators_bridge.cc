#include "genetic_operators.h"



extern "C" {

    GeneticOperators* GeneticOperators_(double* upper_bounds_, double* comm_tax_free, short* n_usr, double* trans_costs, short n_commodities_, short pop_size_, 
                        short off_size_, short n_paths_, double mutation_rate_, short recombination_size_, short num_threads)
    {return new GeneticOperators(upper_bounds_, comm_tax_free, n_usr, trans_costs, n_commodities_,pop_size_, off_size_, n_paths_, mutation_rate_, recombination_size_, num_threads);}



    // void generate_(Genetic* genetic, double* a_parent, double* b_parent, double* child, double* upper_bounds) { genetic -> generate(a_parent, b_parent, child, upper_bounds);}
    double* generation_(GeneticOperators* genetic, double* population) { return genetic -> generation(population);}
    double* eval_parallel_(GeneticOperators* genetic, double* new_population) {return genetic -> eval_parallel(new_population);}

}