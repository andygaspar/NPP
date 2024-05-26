#include "swarm.h"


extern "C" {

    Swarm* Swarm_(double* const comm_tax_free, int* const n_usr, double* const transf_costs, double* u_bounds,double* l_bounds,
                    short n_comm, short n_tolls_, short n_parts, int n_iter, int no_update_lim_, short num_th, short seed)
    {return new Swarm(comm_tax_free, n_usr, transf_costs, u_bounds, l_bounds, n_comm, n_tolls_, n_parts, n_iter, no_update_lim_, num_th, seed);}

    void run_(Swarm* swarm, double* p_init, double* v_init, short n_particles, bool stats, bool verbose)
    { 
        std::vector<std::vector<double>> p_init_ = std::vector<std::vector<double>> (n_particles, std::vector<double> (swarm->n_tolls, 0));
        for(int i=0; i< n_particles; i++){
             for(int j=0; j< swarm->n_tolls; j++){
                p_init_[i][j] = p_init[swarm->n_tolls * i + j];
             }
        }

        std::vector<std::vector<double>> v_init_ = std::vector<std::vector<double>> (n_particles, std::vector<double> (swarm->n_tolls, 0));
        for(int i=0; i< n_particles; i++){
             for(int j=0; j< swarm->n_tolls; j++){
                v_init_[i][j] = v_init[swarm->n_tolls * i + j];
             }
        }

        swarm -> run(p_init_, v_init_, stats, verbose); }

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

    void destroy(Swarm* swarm) {delete swarm;}

    

}