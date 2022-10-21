#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "particle_.cpp"
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
    std::vector<Particle> particles;
    short n_particles;
    int n_iterations;
    short n_commodities;
    short n_tolls;
    double lim_l=0.0;
    double lim_h=1.0;
    double w=0.9;
    double c_soc=1.49445;
    double c_cog=1.49445;
    double best_val;
    short best_particle_idx;
    double* p_best;
    short num_threads;

    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    public:
    Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* u_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th);
    //Swarm() {n_particles=1; n_dim=2; particles=Vector<Particle>{n_particles}; p_best=particles[0].pos(); num_threads=2;}
    void set_init_sols(double* solutions, int n_solutions);
    double get_best_val() {return best_val;}
    double * get_best() {return p_best;}
    void run();

    void update_w() {w = w - 0.5/(n_iterations);}
    int size() {return n_particles;}
    void print();
    void print_particles();
};


Swarm::Swarm(double* comm_tax_free, int* n_usr,double* transf_costs, double* u_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th) {
    n_iterations = n_iter;
    n_particles=n_parts; 
    n_tolls=n_to;
    n_commodities=n_comm;
    w = 0.9;
    num_threads=num_th;
    best_val = 0;

    p_best = new double[n_tolls];
    particles=std::vector<Particle>(n_particles);

    for(int i=0;i<n_particles;++i){
        particles[i] = Particle{comm_tax_free, n_usr ,transf_costs, u_bounds, n_commodities, n_tolls, i};
    
    for(int i=0; i< n_tolls; i++) p_best[i]=particles[0].p[i];
    //fp = f;
    }

}



void Swarm::run(){
    std::vector<double> run_results(n_particles);
    int i;
    bool new_best = false;

    for(int iter=0; iter< n_iterations; iter++) {


        #pragma omp parallel for num_threads(this->num_threads) shared(run_results, particles) //reduction(max : run_result)//implicit(none) private(i) shared(run_results, n_particles, particles)
        for(i=0;i<n_particles;++i) {
            run_results[i] = particles[i].compute_obj_and_update_best();
        }

        for(i=0;i<n_particles;++i){
            if(run_results[i] > best_val) {
                new_best = true;
                best_val = run_results[i];
                best_particle_idx = i;
            }
        }
        if(new_best) {
            for(int i=0; i< n_tolls; i++) p_best[i]=particles[best_particle_idx].p[i];
            new_best = false;
        }
        #pragma omp parallel for num_threads(this->num_threads) shared(w, c_soc, c_cog, p_best)
        for(i=0;i<n_particles;++i) {
            particles[i].update_pos();
            particles[i].update_vel(w,c_soc,c_cog, p_best);
        }
        this->update_w();

        if(iter%1000==0) std::cout<<best_val<<"  iter"<< iter<< std::endl;
    }
}


void Swarm::set_init_sols(double* solutions, int n_solutions) {
    std::cout<<"   init solutions "<< n_solutions<<"  tolls"<<n_tolls<< std::endl;
    int n_sols = n_solutions;
    if(n_sols > n_particles) n_sols = n_particles;

    for(int i=0; i< n_sols; i ++) {
        for(int j=0; j< n_tolls; j++) particles[i].p[j] = solutions[i*n_tolls + j];
    }

    for(int j=0; j< n_tolls; j++) std::cout<<particles[0].p[j]<< " ";
    std::cout<< std::endl; 
    std::cout<<particles[0].compute_obj_and_update_best()<<std::endl;
    std::cout<< "  ++++++  " <<particles[0].personal_best_val<< std::endl;
    for(int i=0; i< n_tolls; i++) p_best[i]=solutions[i];
}





std::ostream& operator<<( std::ostream &os, Swarm& s ) {
    std::cout<<"best pos -> "<<s.p_best<< " best obj -> "<<std::endl;
    for(int i=0; i<s.n_particles; ++i) {
        std::cout<<s.particles[i]<<"  obj -> "<< " best obj -> ";
        if (i<(s.n_particles-1))
            std::cout<<std::endl;
    }
    return os;
}

void Swarm::print() {
    std::cout<<*this<<std::endl;
}

void Swarm::print_particles(){
    for(int i=0; i< n_particles; i++ ) particles[i].print();
}
