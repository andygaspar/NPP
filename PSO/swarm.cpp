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
    double lim_l=0.0;
    double lim_h=1.0;
    double w=0.9;
    double c_soc=1.49445;
    double c_cog=1.49445;
    double best_val;
    Vector<double> p_best;
    double (*fp)(Vector<double>);

    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    public:
    Swarm(int n, int n_, double (*f)(Vector<double>), double ll, double lh);
    Swarm(int n, int n_, double (*f)(Vector<double>));
    Swarm(double* cost_array, int n, int n_, double (*f)(Vector<double>));
    Swarm() {n_particles=1; n_dim=2; particles=Vector<Particle>{n_particles}; p_best=particles[0].pos();}

    double get_best_val() {return best_val;}
    void update_best(int best_particle, double new_best_val)
        {p_best=particles[best_particle].pos(); best_val = new_best_val;}
    void set_ndim(int n) {n_dim=n; *this=Swarm{n_particles,n,fp};}
    void set_nparticles(int n) {n_particles=n; *this=Swarm{n,n_dim,fp};}
    void set_lim(double ll, double lh) {lim_l=ll; lim_h=lh; *this=Swarm{n_particles,n_dim,fp,ll,lh};}
    void set_w(double val) {w=val;}
    void set_objective_function(double (*f)(Vector<double>)) {fp = f;}
    //void test(Particle p);

    void update_w(int n_iter) {w = w - 0.5/n_iter;}
    void update_swarm(int iteration, double* run_results);
    void update(int num);
    int size() {return n_particles;}
    Vector<double> position(int i) {return particles[i].pos();}
    void print();
};


Swarm::Swarm(int n, int n_, double (*f)(Vector<double>)){}

Swarm::Swarm(double* cost_array, int n, int n_, double (*f)(Vector<double>)) {
    n_particles=n; 
    n_dim=n_;
    best_val = 0;
    particles=Vector<Particle>{n_particles};
    for(int i=0;i<n_particles;++i){
        particles[i] = Particle{&cost_array[n_dim*i], n_dim};
    p_best=particles[0].pos();
    fp = f;
    }
}

void Swarm::update_swarm(int iteration, double* run_results) {
    this->update_w(iteration);
    #pragma omp parallel for
    for(int i=0;i<n_particles;++i) {
        if(run_results[i] > particles[i].get_personal_best_val()) {
            particles[i].update_best(run_results[i]);
        }
        particles[i].update_pos();
        particles[i].update_vel(w,c_soc,c_cog,p_best);
//        if (fp(particles[i].pos()) > fp(particles[i].p_b()))
//            particles[i].update_best();
//        if (fp(particles[i].pos()) > fp(p_best))
//            this->update_best(particles[i].pos());
    }
}

//void Swarm::update(int num) {
//    for(int i=0;i<num;++i) {
//        //std::cout<<omp_get_thread_num()<<"/"<<omp_get_num_threads()<<std::endl;
//        this->update_swarm();
//        this->update_w(num);
//        //std::cout<<*this<<std::endl<<std::endl;
//    }
//}

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
