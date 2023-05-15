#include "swarm.h"


extern "C" {

    Swarm* Swarm_(double* const comm_tax_free, int* const n_usr, double* transf_costs, double* const obj_coef,
                    short n_comm, short n_tolls_, short n_parts, int n_iter, int no_update_lim_, short num_th)
    {return new Swarm(comm_tax_free, n_usr, transf_costs, obj_coef, n_comm, n_tolls_, n_parts, n_iter, no_update_lim_, num_th);}

    void run_(Swarm* swarm, double* p_init, double* v_init,  double* u_bounds,double* l_bounds, bool stats = false, bool verbose = false)
    { swarm -> run(p_init, v_init, u_bounds, l_bounds, stats, verbose); }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}

    double* get_best_ (Swarm* swarm) {return swarm -> get_best();}


}