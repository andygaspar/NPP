#include "swarm.h"


extern "C" {

    Swarm* Swarm_(double* const comm_tax_free, int* const n_usr, double* transf_costs, double* u_bounds,double* l_bounds,
                    short n_comm, short n_tolls_, short n_parts, int n_iter, int no_update_lim_, short num_th)
    {return new Swarm(comm_tax_free, n_usr, transf_costs, u_bounds, l_bounds, n_comm, n_tolls_, n_parts, n_iter, no_update_lim_, num_th);}

    void run_(Swarm* swarm, double* p_init, double* v_init, bool stats, bool verbose, short seed)
    { swarm -> run(p_init, v_init, stats, verbose, seed); }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}

    double* get_best_ (Swarm* swarm) {return swarm -> get_best();}

    int get_actual_iteration_(Swarm* swarm){return swarm-> get_actual_iteration();}
    int get_stats_len_(Swarm* swarm){return swarm-> get_stats_len();}
    double* get_p_means_(Swarm* swarm){return swarm -> get_p_means();}
    double* get_v_means_(Swarm* swarm){return swarm -> get_v_means();}
    double* get_p_stds_(Swarm* swarm){return swarm -> get_p_stds();}
    double* get_v_stds_(Swarm* swarm){return swarm -> get_v_stds();}


}