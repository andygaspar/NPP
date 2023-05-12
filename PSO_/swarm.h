#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include "particle.h"
//#include "p_optimum.cpp"

//std::vector<double> p_optimum{};

/*
Swarm class is basically an ensemble of Particle objects, in particular characterized by:
- particles: Vector object with elements of type Particle
- n_particles: number of particles in the ensemble
- n_dim: dimensionality of the space
- search_lb, search_ub: low and high limit of the space
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
    public:
    // problem related features
    short n_commodities;
    short n_tolls;
    double* obj_coefficients;
    double* search_lb;
    double* search_ub;

    //PSO parameters
    int n_iterations;
    int no_update_lim;
    int stop_param=0;
    std::vector<Particle> particles;
    short n_particles;
    
    // computation parameters
    short num_threads;
    bool verbose=false;

    double best_val;
    short best_particle_idx;

    


    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    Swarm( double* const comm_tax_free, int* const n_usr, double* transf_costs, double* const obj_coef, 
                     short n_comm, short n_tolls_, short n_parts, int n_iter, int no_update_lim_, short num_th);
    Swarm() {}

    double* get_best() {
        double* solution = new double[n_tolls]; 
        for(int i=0; i<n_tolls; i++) 
        {solution[i] = particles[best_particle_idx].p[i];}
        return solution;
        }

    double get_best_val() {return best_val;}

    void run(double* p_init, double* v_init, double* const u_bounds,double* const l_bounds);
    
    void print();
    void print_particles();
    void save_particles();
    /*
    bool check(int id) {
        for (int i=0;i<n_tolls;i++) {
            if ((particles[id].p[i]>search_ub[i]) || (particles[id].p[i]<search_lb[i])) {
                std::cout<<"("<<i<<")"<<search_lb[i]<<"|"<<particles[id].p[i]<<"|"<<search_ub[i]<<std::endl;
                return false;
            } 
        }
        return true;
    }*/

};

/*-----------------------------------------------------------------------------------------*/
/* Initialize the swarm object and its particles with random velocity and given positions. */                                                                        
/*-----------------------------------------------------------------------------------------*/
Swarm::Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, 
                     short n_comm, short n_tolls_, 
                    short n_parts, int n_iter, int no_update_lim_, short num_th) {
    n_iterations = n_iter;
    n_particles=n_parts; 
    n_tolls=n_tolls_;
    n_commodities=n_comm;
    obj_coefficients=obj_coef;
    no_update_lim = no_update_lim_,
    num_threads=num_th;
    best_particle_idx=0;

    particles=std::vector<Particle>(0);





    for(int i=0;i<n_particles;++i){
        particles.push_back({comm_tax_free, n_usr ,transf_costs, obj_coefficients, n_commodities, n_tolls, i});
    }
}


void Swarm::run(double* p_init, double* v_init, double* const u_bounds, double* const l_bounds){
    std::vector<double> run_results(n_particles);

        // no idea what is this
    double d_M = 0;
    for (int i=0;i<n_tolls;++i)
        d_M += std::pow(u_bounds[i]-l_bounds[i],2);
    d_M = std::sqrt(d_M);


    #pragma omp parallel for num_threads(this->num_threads) shared(particles)
    for(int i=0;i<n_particles;++i){
        particles[i].init_values(&p_init[i+n_tolls], &v_init[i+n_tolls], u_bounds,l_bounds, d_M);
    }

    int i;
    bool new_best = false;
    bool new_glob_best=false;
    int no_update = 0;
    double random_param=0.01;
    best_val = 0;
    std::cout<<n_iterations<<std::endl;

    for(int iter=0; iter< n_iterations; iter++) {

        
        #pragma omp parallel for num_threads(this->num_threads) shared(run_results, particles) //reduction(max : run_result)//implicit(none) private(i) shared(run_results, n_particles, particles)
        for(i=0;i<n_particles;++i) {
            if(no_update>195 && no_update<205) {random_param=0.04;}
            else {random_param=0.02;};

            particles[i].update_vel(particles[best_particle_idx].p, iter, random_param);
            particles[i].update_pos();
            run_results[i] = particles[i].compute_obj_and_update_best();
            }

        for(i=0;i<n_particles;++i){
            if(run_results[i] > best_val) {
                    best_val = run_results[i];
                    best_particle_idx = particles[i].particle_idx;
                    no_update= 0;
                    new_glob_best=true;
                }
            }
        std::cout<<best_val<<std::endl;
        
        if (new_glob_best==false) no_update++;
        else no_update = 0;
    
        if (no_update>no_update_lim) stop_param=1;
        }
    }


void Swarm::print() {
    std::cout<<*this<<std::endl;
}

void Swarm::print_particles(){
    for(int i=0; i< n_particles; i++ ) particles[i].print();
}

void Swarm::save_particles() {
    std::string file_name = "data.csv";
    //std::remove(file_name);
    std::ofstream outfile;
    outfile.open(file_name);
    for (int i=0;i<n_particles;++i) {
        for (int j=0;j<2;++j) {
            outfile<<particles[i].p[j];
            if (i<2-1)
                outfile<<",";
            else
                outfile<<std::endl;
        }
    }
    outfile.close();
}

std::ostream& operator<<( std::ostream &os, Swarm& s ) {
    std::cout<<"best pos -> "<<s.particles[s.best_particle_idx]<< " best obj -> "<<std::endl;
    for(int i=0; i<s.n_particles; ++i) {
        std::cout<<s.particles[i]<<"  obj -> "<< " best obj -> ";
        if (i<(s.n_particles-1))
            std::cout<<std::endl;
    }
    return os;
}


/*
void Swarm::print_output(int iter) {
    std::cout<<"<"<<n_p<<">"<<"("<<cube_best_particle_idx[best_idx]<<") "<<cube_best_val[best_idx]<<
            "["<<particles[cube_best_particle_idx[best_idx]].current_run_val<<
            "]"<<"  iter "<< iter<< "[["<<no_update<<"]]"<<
            " w: "<<particles[cube_best_particle_idx[best_idx]].w<< " c_soc: "<<
             particles[cube_best_particle_idx[best_idx]].c_soc<< " c_cog: "
             <<particles[cube_best_particle_idx[best_idx]].c_cog<<
             " fitness: "<<particles[cube_best_particle_idx[best_idx]].fitness<<"[ "<<
             particles[cube_best_particle_idx[best_idx]].fitness_memb[0]<<", "<<
             particles[cube_best_particle_idx[best_idx]].fitness_memb[1]<<", "<<
             particles[cube_best_particle_idx[best_idx]].fitness_memb[2]<<" ]"<< 
             " sigma: "<<particles[cube_best_particle_idx[best_idx]].sigma<<"[ "<<
             particles[cube_best_particle_idx[best_idx]].sigma_memb[0]<<", "<<
             particles[cube_best_particle_idx[best_idx]].sigma_memb[1]<<", "<<
             particles[cube_best_particle_idx[best_idx]].sigma_memb[2]<<" ]"<< "vel: "<<
             compute_distance(particles[cube_best_particle_idx[best_idx]].v, std::vector<double>(n_tolls,0.));
             if (p_optimum.size()!=0)
             std::cout<<" real distance: "<<
             compute_distance(particles[cube_best_particle_idx[best_idx]].personal_best,p_optimum)<<
             "]"<<std::endl;
             else std::cout<<std::endl;
}
*/