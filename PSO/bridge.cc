#include "genetic.h"


extern "C" {

    Swarm* Swarm_(double* const comm_tax_free, int* const n_usr, double* transf_costs, double* u_bounds,double* l_bounds,
                    short n_comm, short n_tolls_, short n_parts, int n_iter, int no_update_lim_, short num_th)
    {return new Swarm(comm_tax_free, n_usr, transf_costs, u_bounds, l_bounds, n_comm, n_tolls_, n_parts, n_iter, no_update_lim_, num_th);}

    void run_(Swarm* swarm, double* p_init, double* v_init, bool stats, bool verbose, short seed)
    { swarm -> run(p_init, v_init, stats, verbose, seed); }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}

    double* get_best_ (Swarm* swarm) {return swarm -> get_best();}

    double* get_particles_(Swarm* swarm) {return swarm -> get_particle_position();}
    double* get_values_(Swarm* swarm) {return swarm -> get_particle_values();}

    int get_actual_iteration_(Swarm* swarm){return swarm-> get_actual_iteration();}
    int get_stats_len_(Swarm* swarm){return swarm-> get_stats_len();}
    double* get_p_means_(Swarm* swarm){return swarm -> get_p_means();}
    double* get_v_means_(Swarm* swarm){return swarm -> get_v_means();}
    double* get_p_stds_(Swarm* swarm){return swarm -> get_p_stds();}
    double* get_v_stds_(Swarm* swarm){return swarm -> get_v_stds();}

    


    GeneticOperators* Genetic_(double* upper_bounds_, double* comm_tax_free, short* n_usr, double* trans_costs, short n_commodities_, short pop_size_, 
                        short off_size_, short n_paths_, double mutation_rate_, short recombination_size_, short num_threads)
    {return new GeneticOperators(upper_bounds_, comm_tax_free, n_usr, trans_costs, n_commodities_,pop_size_, off_size_, n_paths_, mutation_rate_, recombination_size_, num_threads);}


    Genetic* Genetic2_(double* upper_bounds_, double* comm_tax_free, int* n_usr, double* trans_costs, short n_commodities_, short n_paths_, 
                        short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
                        short pso_size_, short pso_selection_, short pso_every_, short pso_iterations_, short pso_final_iterations_, short pso_no_update_lim_,
                        short num_threads_, bool verbose_, short seed) {
        return new Genetic(upper_bounds_, comm_tax_free, n_usr, trans_costs, n_commodities_, n_paths_, 
                        pop_size_, off_size_, mutation_rate_, recombination_size_, 
                        pso_size_, pso_selection_, pso_every_, pso_iterations_, pso_final_iterations_, pso_no_update_lim_,
                        num_threads_, verbose_, seed);
    }

    void run_genetic_(Genetic* g, double* population, int iterations) {g -> run(population,iterations);}

    // void generate_(Genetic* genetic, double* a_parent, double* b_parent, double* child, double* upper_bounds) { genetic -> generate(a_parent, b_parent, child, upper_bounds);}
    double* generation_(GeneticOperators* genetic, double* population) { return genetic -> generation(population);}
    double* eval_parallel_(GeneticOperators* genetic, double* new_population) {return genetic -> eval_parallel(new_population);}

}