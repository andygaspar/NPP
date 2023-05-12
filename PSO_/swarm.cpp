#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include "particle.cpp"
#include <omp.h>
#include <cstdlib>
#include "p_optimum.cpp"

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
    int no_update=0;
    int  top_param=0;
    std::vector<Particle> particles;
    short n_particles;
    
    // computation parameters
    short num_threads;
    bool verbose;

    


    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    Swarm(std::vector<std::vector<double>> p_init,double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, 
                    double* const u_bounds,double* const l_bounds, short n_comm, short n_tolls_, short n_parts, int n_iter, short num_th, int n_u_l=500);
    Swarm() {}

    void set_init_sols(double* solutions, int n_solutions);
    void lower_particles(std::vector<double>& results);
    void centering_particles(int N, double var);



    void run_and_lower(int stop);
    
    void print();
    void print_particles();
    void save_particles();
    void set_vals();
    bool check(int id) {
        for (int i=0;i<n_tolls;i++) {
            if ((particles[id].p[i]>search_ub[i]) || (particles[id].p[i]<search_lb[i])) {
                std::cout<<"("<<i<<")"<<search_lb[i]<<"|"<<particles[id].p[i]<<"|"<<search_ub[i]<<std::endl;
                return false;
            } 
        }
        return true;
    }
    void print_output(int iter);

    double get_best_val() {
        return cube_best_val[best_idx];
    }

    double* get_best(){
        std::cout<<best_idx<<std::endl;
        std::cout<<cube_p_best[best_idx][0]<<std::endl;
        return cube_p_best[best_idx];
    }
};

/*-----------------------------------------------------------------------------------------*/
/* Initialize the swarm object and its particles with random velocity and given positions. */                                                                        
/*-----------------------------------------------------------------------------------------*/
Swarm::Swarm(std::vector<std::vector<double>> p_init, double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef,double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, int n_u_l) {
    n_iterations = n_iter;
    no_update_lim=n_u_l;
    n_particles=n_parts; 
    n_particles = N_PARTS;
    n_tolls=n_to;
    n_commodities=n_comm;
    obj_coefficients=obj_coef;
    search_ub = u_bounds;
    search_lb = l_bounds;
    num_threads=num_th;
    n_subcubes=1;
    best_idx=0;
    cube_best_val = new double[n_subcubes];
    cube_p_best = new double*[n_subcubes];
    cube_best_particle_idx = new short[n_subcubes];

    cube_p_best[0] = new double[n_tolls];
    particles=std::vector<Particle>(0);

    d_M = 0;
    for (int i=0;i<n_tolls;++i)
        d_M += std::pow(u_bounds[i]-l_bounds[i],2);
    d_M = std::sqrt(d_M);

    for(int i=0;i<n_p;++i){
        particles.push_back({p_init[i], comm_tax_free, n_usr ,transf_costs,obj_coefficients, u_bounds,l_bounds, n_commodities, n_tolls, i, d_M,0});
    }

    double tmp = 0; 
    cube_best_val[0] = 0;
    for(int i=0;i<n_p;++i) {
        tmp = particles[i].compute_obj_and_update_best();
        if (tmp>cube_best_val[0])
            cube_best_particle_idx[0] = i;
            cube_best_val[0] = tmp;
    }

    for(int i=0; i< n_tolls; i++) cube_p_best[0][i]=particles[cube_best_particle_idx[0]].p[i];
}


void Swarm::run_and_lower(int stop){
    std::vector<double> run_results(n_p);
    int i;
    bool new_best = false;
    bool new_glob_best=false;
    int high=0;
    int red_done=0;
    double random_param=0.01;

    for(int iter=0; iter< n_iterations; iter++) {
        if (stop_param==1) iter=n_iterations;
        // lower particles
        if (iter==stop){
            n_p = N_PARTICLES;
            for (int k=0;k<n_particles;k++) {run_results[k] = particles[k].personal_best_val;}
            lower_particles(run_results);
            red_done=1;
        }

        if(verbose and iter%10==0) {
            print_output(iter);
        }
        
        #pragma omp parallel for num_threads(this->num_threads) shared(run_results, particles) //reduction(max : run_result)//implicit(none) private(i) shared(run_results, n_particles, particles)
        for(i=0;i<n_p;++i) {
            if(no_update>195 && no_update<205) {random_param=0.04;}
            else {random_param=0.02;};
            if(iter<stop){
                particles[i].update_params(cube_p_best[particles[i].lh_idx], cube_best_val[particles[i].lh_idx]);
                particles[i].update_vel(cube_p_best[particles[i].lh_idx], iter, stop, stop_param, cube_best_particle_idx[best_idx], random_param);
                particles[i].update_pos();
                run_results[i] = particles[i].compute_obj_and_update_best();
            }
            else{
                particles[i].update_params(cube_p_best[best_idx], cube_best_val[best_idx]);
                particles[i].update_vel(cube_p_best[best_idx], iter, stop, stop_param, cube_best_particle_idx[best_idx], random_param);
                particles[i].update_pos();
                run_results[i] = particles[i].compute_obj_and_update_best();
            }
            
        }

        for(i=0;i<n_p;++i){
            if(run_results[i] > cube_best_val[particles[i].lh_idx]) {
                if (run_results[i] > cube_best_val[best_idx]) {
                    best_idx = particles[i].lh_idx;
                    no_update= 0;
                    new_glob_best=true;
                }
                cube_best_val[particles[i].lh_idx] = run_results[i];
                cube_best_particle_idx[particles[i].lh_idx] = i;
                for(int j=0; j< n_tolls; j++) cube_p_best[particles[i].lh_idx][j]=particles[i].p[j];
            }
        }
        if (new_glob_best==false)
            no_update++;
        new_glob_best=false;
    
        if (no_update>no_update_lim) stop_param=1;
    }
}

void Swarm::print() {
    std::cout<<*this<<std::endl;
}

void Swarm::print_particles(){
    for(int i=0; i< n_p; i++ ) particles[i].print();
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
    std::cout<<"best pos -> "<<s.cube_p_best<< " best obj -> "<<std::endl;
    for(int i=0; i<s.n_p; ++i) {
        std::cout<<s.particles[i]<<"  obj -> "<< " best obj -> ";
        if (i<(s.n_p-1))
            std::cout<<std::endl;
    }
    return os;
}

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