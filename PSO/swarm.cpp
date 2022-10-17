#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "particle.cpp"
#include <omp.h>

/*
Swarm class is basically an ensemble of Particle objects, in particular characterized by:
- particles: Vector object with elements of type Particle
- n_particles: number of particles in the ensemble
- n_dim: dimensionality of the space
- lim_l, lim_h: low and high limit of the space
- w, c_soc, c_cog: PSO parameters
- p_best: coordinates associated with the best position found so far
          in the ensemble's history according to a given objective function
- fp: function pointer to the objective function considered

A Swarm object can be initialized randomly placing the particles in the space
and the methods implemented allow to:
- linearly decrease the w PSO parameter
- update each particle in the swarm a certain number of times iteratively
- output stream operator
*/
class Swarm {
    private:
    Vector<Particle> particles;
    int n_particles;
    int n_dim;
    int n_iterations;
    double lim_l=0.0;
    double lim_h=1.0;
    double w=0.9;
    double c_soc=1.49445;
    double c_cog=1.49445;
    double best_val;
    Vector<double> p_best;
    double* actual_costs;
    double (*fp)(Vector<double>);

    int num_threads;

    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    public:
    Swarm(double* cost_array, double* ac, double* sfa, int n, int n_, int n_iter, double (*f)(Vector<double>));
    Swarm() {n_particles=1; n_dim=2; particles=Vector<Particle>{n_particles}; p_best=particles[0].pos(); num_threads=2;}

    double get_best_val() {return best_val;}
    void update_best(int best_particle, double new_best_val)
        {p_best=particles[best_particle].pos(); best_val = new_best_val;}
    void set_objective_function(double (*f)(Vector<double>)) {fp = f;}
    //void test(Particle p);

    void update_w() {w = w - 0.5/(n_iterations);}
    double* update_swarm(int iteration, double* run_results);
    int size() {return n_particles;}
    Vector<double> position(int i) {return particles[i].pos();}
    void print();
};


Swarm::Swarm(double* cost_array, double* ac, double* sfa, int n, int n_, int n_iter, double (*f)(Vector<double>)) {
    n_iterations = n_iter;
    n_particles=n; 
    n_dim=n_;
    best_val = 0;
    actual_costs = ac;
    particles=Vector<Particle>{n_particles};

    for(int i=0;i<n_particles;++i){
        particles[i] = Particle{&cost_array[n_dim*i],&actual_costs[n_dim*i], sfa, n_dim, i};
    p_best=particles[0].pos();
    fp = f;
    }
    w = 0.9;
    num_threads=1;
}

double* Swarm::update_swarm(int iteration, double* run_results) {
//    Vector<double> ress = particles[2].pos();
//    std::cout<<"iteration"<<iteration<<" "<<ress<<std::endl;
    #pragma omp parallel for num_threads(this->num_threads)
    for(int i=0;i<n_particles;++i) {
        if(run_results[i] > particles[i].get_personal_best_val()) {
            particles[i].update_best(run_results[i]);
        }
        particles[i].update_pos();
        particles[i].update_vel(w,c_soc,c_cog,p_best);
    }
//    std::cout<<" actual ";
//    for(int i=0; i<n_particles*n_dim; i++) std::cout<<actual_costs[i]<<" ";
//    std::cout<<std::endl;
    this->update_w();

    return actual_costs;
}



std::ostream& operator<<( std::ostream &os, Swarm& s ) {
    std::cout<<"best pos -> "<<s.p_best<< " best obj -> "<<s.fp(s.p_best)<<std::endl;
    for(int i=0; i<s.n_particles; ++i) {
        std::cout<<s.particles[i]<<"  obj -> "<<s.fp(s.particles[i].pos())<< " best obj -> "<<s.fp(s.particles[i].p_b());
        if (i<(s.n_particles-1))
            std::cout<<std::endl;
    }
    return os;
}

void Swarm::print() {
    std::cout<<*this<<std::endl;
}
