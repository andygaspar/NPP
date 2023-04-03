#include "swarm_.cpp"


extern "C" {

    Swarm* Swarm_(double* comm_tax_free, int* n_usr, double* transf_costs, double* u_bounds, double* l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th, short N_PARTS, short n_cut, short N_div, int n_u_l) // float (*f)(Vector<float>))
    {return new Swarm(comm_tax_free, n_usr, transf_costs, u_bounds,l_bounds, n_comm, n_to, n_parts, n_iter, num_th, N_PARTS, n_cut,  N_div, n_u_l);}

    void run_(Swarm* swarm, short stop){ swarm -> run_and_lower(stop); }

    void set_init_sols_(Swarm* swarm, double* solutions, int n_solutions) { swarm -> set_init_sols(solutions, n_solutions); }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}

    double* get_best_ (Swarm* swarm) {return swarm -> get_best();}


}