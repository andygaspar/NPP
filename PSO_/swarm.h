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
    Params parameters;
    int n_iterations;
    int no_update_lim;
    bool no_update_lim_reached = false;
    std::vector<Particle> particles;
    short n_particles;
    
    // computation parameters
    short num_threads;
    bool verbose=false;

    double best_val;
    short best_particle_idx;


    // statistics parms
    std::vector<std::vector<double>> p_means;
    std::vector<std::vector<double>> p_stds;
    std::vector<std::vector<double>> v_means;
    std::vector<std::vector<double>> v_stds;
    int stat_frequency;
    int actual_final_iterations;



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

    void run(double* p_init, double* v_init, double* const u_bounds,double* const l_bounds, bool stats, bool verbose);
    
    void print();
    void print_particles();
    void updte_stats();

    double* get_particle_position(){
        double* positions = new double[n_tolls*n_particles];
        for(int p=0; p < n_particles; p++){
            for(int i=0; i< n_tolls; i++) positions[p*n_tolls + i] = particles[p].p[i];
        }
        return positions;
    }

    double* get_particle_velocity(){
        double* velocity = new double[n_tolls*n_particles];
        for(int p=0; p < n_particles; p++){
            for(int i=0; i< n_tolls; i++) velocity[p*n_tolls + i] = particles[p].v[i];
        }
        return velocity;
    }

    int get_actual_iteration() {return actual_final_iterations;}
    double* get_stats_array(std::vector<std::vector<double>> v);
    double* get_p_means();
    double* get_v_means();
    double* get_p_stds();
    double* get_v_stds();

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



    parameters = Params();

    for(int i=0;i<n_particles;++i){
        particles.push_back({comm_tax_free, n_usr ,transf_costs, obj_coefficients, n_commodities, n_tolls, i, parameters});
    }
}


void Swarm::run(double* p_init, double* v_init, double* const u_bounds, double* const l_bounds, bool stats, bool verbose){
    std::vector<double> run_results(n_particles);

        // no idea what is this
    double d_M = 0;
    for (int i=0;i<n_tolls;++i)
        d_M += std::pow(u_bounds[i]-l_bounds[i],2);
    d_M = std::sqrt(d_M);


    if (stats){}


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
    int iter=0;

    while((iter< n_iterations) and (!no_update_lim_reached)) {

        
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

        double speed = 0;
        for(i=0;i<n_tolls;++i) speed += particles[best_particle_idx].v[i];
        speed = speed/n_tolls;

        if(verbose and (iter%100 == 0)) std::cout<<"iter "<<iter<<"  best_val "<<best_val<<" "<<speed<<std::endl;
        
        if (new_glob_best==false) no_update++;
        else no_update = 0;
    
        if (no_update>no_update_lim) no_update_lim_reached = true;

        if(stats and (iter % parameters.stat_frequency == 0)) updte_stats();

        iter++;

        }

        actual_final_iterations = iter - 1;
    }


void Swarm::updte_stats(){
    p_means.push_back(std::vector<double>(n_tolls, 0));
    p_stds.push_back(std::vector<double>(n_tolls, 0));
    v_means.push_back(std::vector<double>(n_tolls, 0));
    v_stds.push_back(std::vector<double>(n_tolls, 0));

    double p_mean; double p_std; double v_mean; double v_std;

    for(int toll=0; toll< n_tolls; toll++) {
        p_mean = 0; p_std = 0; v_mean = 0; v_std= 0;

        for(int p=0; p< n_particles; p++) {
            p_mean += particles[p].p[toll];
            v_mean += particles[p].v[toll];
        }
        p_means[p_means.size() - 1][toll] = p_mean/n_particles;
        v_means[v_means.size() - 1][toll] = v_mean/n_particles;

        for(int p=0; p< n_particles; p++) {
            p_std += pow(particles[p].p[toll] - p_means[p_means.size() - 1][toll], 2);
            v_std += pow(particles[p].v[toll] - v_means[v_means.size() - 1][toll], 2);
        }
        p_stds[p_stds.size() - 1][toll] = sqrt(p_std/n_particles);
        v_stds[v_stds.size() - 1][toll] = sqrt(v_std/n_particles);
        std::cout<<p_means[0][toll]<<"   llllll"<<std::endl;
    }
}



double* Swarm::get_stats_array(std::vector<std::vector<double>> v){
    double* array = new double[n_tolls*actual_final_iterations];
    for(int i=0; i< actual_final_iterations; i++)
        for(int t=0; t < n_tolls; t++){
             array[i*n_tolls + t] = v[i][t];
        }
    return array;
}

double* Swarm::get_p_means(){return get_stats_array(p_means);}
double* Swarm::get_v_means(){return get_stats_array(v_means);}
double* Swarm::get_p_stds(){return get_stats_array(p_stds);}
double* Swarm::get_v_stds(){return get_stats_array(v_stds);}


void Swarm::print() {
    std::cout<<*this<<std::endl;
}

void Swarm::print_particles(){
    for(int i=0; i< n_particles; i++ ) particles[i].print();
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