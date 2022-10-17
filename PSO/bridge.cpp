#include "swarm.cpp"


extern "C" {

    Swarm* Swarm_(double* cost_array, double* actual_costs, double* scale_factor_array, int n, int n_, int n_iter) // float (*f)(Vector<float>))
    {return new Swarm(cost_array,actual_costs, scale_factor_array, n, n_, n_iter, &obj);}

//    void update_(Swarm* swarm, int n){ swarm -> update(n); }
    double* update_swarm_(Swarm* swarm, int iteration, double* run_results){
    return swarm -> update_swarm(iteration, run_results); }

    double* test_io(double* input, int n) {

        double* output = new double[n];
        for (int i=0; i<n; i++) output[i] = i/2.0;
        return output;
        }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}
    void update_best_(Swarm* swarm, int best_particle, double new_best_val)
    {swarm -> update_best(best_particle, new_best_val);}


    void print_s(Swarm* swarm) {std::cout<<*swarm<<std::endl<<std::endl;}


}