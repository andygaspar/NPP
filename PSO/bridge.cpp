#include "swarm.cpp"


extern "C" {

    Swarm* Swarm_(double* cost_array, int n, int n_) // float (*f)(Vector<float>))
    {return new Swarm(cost_array, n, n_, &obj);}

//    void update_(Swarm* swarm, int n){ swarm -> update(n); }
    void update_swarm_(Swarm* swarm, int iteration, double* run_results){ swarm -> update_swarm(iteration, run_results); }

    double* test_io(double* input, int n) {
        for(int i=0; i< n; i++) std::cout<<input[i]<<" ";
        std::cout<<std::endl;
        double* output = new double[n];
        for (int i=0; i<n; i++) output[i] = i/2.0;
        return output;
        }

    double get_best_val_ (Swarm* swarm) {return swarm -> get_best_val();}
    void update_best_(Swarm* swarm, int best_particle, double new_best_val)
    {swarm -> update_best(best_particle, new_best_val);}


    void print_s(Swarm* swarm) {std::cout<<*swarm<<std::endl<<std::endl;}


}