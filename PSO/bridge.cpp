#include "swarm.cpp"


extern "C" {

    Swarm* Swarm_(double* cost_array, int n, int n_) // float (*f)(Vector<float>))
    {return new Swarm(cost_array, n, n_, &obj);}

    void update_(Swarm* swarm, int n){ swarm -> update(n); }

    double* test_io(double* input, int n) {
        for(int i=0; i< n; i++) std::cout<<input[i]<<" ";
        std::cout<<std::endl;
        double* output = new double[n];
        for (int i=0; i<n; i++) output[i] = i/2.0;
        return output;
        }


    void print_s(Swarm* swarm) {std::cout<<*swarm<<std::endl<<std::endl;}


}