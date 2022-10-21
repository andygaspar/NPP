#include "swarm_.cpp"


extern "C" {

    Swarm* Swarm_(double* comm_tax_free, int* n_usr, double* transf_costs, double* u_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th) // float (*f)(Vector<float>))
    {return new Swarm(comm_tax_free, n_usr, transf_costs, u_bounds, n_comm, n_to, n_parts, n_iter, num_th);}

    void run_(Swarm* swarm){ swarm -> run(); }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}

    double* get_best_ (Swarm* swarm) {return swarm -> get_best();}


}