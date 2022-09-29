#include "swarm.cpp"


extern "C" {

    Swarm* Swarm_(int n, int n_) // float (*f)(Vector<float>))
    {return new Swarm(n, n_, &obj);}

    void update_(Swarm* swarm, int n){ swarm -> update(n); }
    void print_s(Swarm* swarm) {std::cout<<*swarm<<std::endl<<std::endl;}


}