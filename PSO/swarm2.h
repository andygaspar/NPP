#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include "swarm.h"

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


class Swarm2 {
    public:
    // problem related features
    short n_commodities;
    short n_tolls;
    double* search_lb;
    double* search_ub;

    //PSO parameters
    Params parameters;
    short n_iterations;
    int no_update_lim;
    bool no_update_lim_reached = false;
    std::vector<Particle> particles;
    std::vector<double> run_results;
    std::vector<double> particles_best;
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



    friend std::ostream& operator<<( std::ostream &os, Swarm2& s );

    Swarm2( double* const comm_tax_free, int* const n_usr, double* transf_costs, double* const u_bounds, double* const l_bounds,
                     short n_comm, short n_tolls_, short n_parts, short n_iter, int no_update_lim_, short num_th, short seed);
    Swarm2() {}

    double* get_best() {
        double* solution = new double[n_tolls]; 
        for(int i=0; i<n_tolls; i++) 
        {solution[i] = particles[best_particle_idx].personal_best[i];}
        return solution;
        }

    ~Swarm2 (){}

    double get_best_val() {return best_val;}
    double get_status() {return no_update_lim_reached;}

    void run(std::vector<std::vector<double>> &p_init, double* v_init, short n_run_iterations, bool verbose);
    
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

    double* get_particle_values(){
        double* values = new double[n_particles];
        for(int p=0; p < n_particles; p++) values[p] = particles[p].personal_best_val;
        return values;
    }

    double* get_particle_velocity(){
        double* velocity = new double[n_tolls*n_particles];
        for(int p=0; p < n_particles; p++){
            for(int i=0; i< n_tolls; i++) velocity[p*n_tolls + i] = particles[p].v[i];
        }
        return velocity;
    }

    int get_actual_iteration() {return actual_final_iterations;}
    int get_stats_len() {return p_means.size();}
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
Swarm2::Swarm2(double* comm_tax_free, int* n_usr, double* transf_costs, double* const u_bounds, double* const l_bounds,
                     short n_comm, short n_tolls_, 
                    short n_parts, short n_iter, int no_update_lim_, short num_th, short seed) {

    n_iterations = n_iter;
    n_particles=n_parts; 
    n_tolls=n_tolls_;
    n_commodities=n_comm;
    no_update_lim = no_update_lim_,
    num_threads=num_th;
    best_particle_idx=0;

    particles=std::vector<Particle> (n_particles);
    run_results = std::vector<double> (n_particles);
    particles_best = std::vector<double> (n_particles);



    parameters = Params();
    double d_M = 0;
    for (int i=0;i<n_tolls;++i)
        d_M += std::pow(u_bounds[i]-l_bounds[i],2);
    d_M = std::sqrt(d_M);

    for(int i=0;i<n_particles;++i){
        particles[i] = Particle{comm_tax_free, n_usr ,transf_costs, u_bounds, l_bounds,n_commodities, n_tolls, i, parameters, d_M, seed};
    }
}


void Swarm2::run(std::vector<std::vector<double>>& p_init, double* v_init, short n_run_iterations, bool verbose){

        // no idea what is this//std::cout<<"seed set to "<<seed<<std::endl;}
    int n_run_particles = p_init.size();


    #pragma omp parallel for num_threads(this->num_threads) shared(particles)
    for(int i=0;i<n_run_particles;++i){
        particles[i].init_vector_values(p_init[i], &v_init[i+n_tolls]);
    }

    int i;
    bool new_best = false;
    bool new_glob_best=false;
    int no_update = 0;
    no_update_lim_reached = false;
    double random_param=0.01;
    best_val = 0;
    int iter=0;

    double avg_velocity= 0;

    #pragma omp parallel for num_threads(this->num_threads) shared(run_results, particles) //reduction(max : run_result)//implicit(none) private(i) shared(run_results, n_particles, particles)
        for(i=0;i<n_run_particles;++i) {
            if(no_update>195 && no_update<205) {random_param=0.04;}
            else {random_param=0.02;};
            run_results[i] = particles[i].compute_obj_and_update_best();
            }
            
    for(i=0;i<n_run_particles; i++){
        if(run_results[i] > best_val) {
            best_val = run_results[i];
            best_particle_idx = particles[i].particle_idx;
            }
        }
    if(verbose) std::cout<<"first iter  best_val: "<<best_val<<"    avg vel: "<<n_run_iterations<<std::endl;

    while((iter< n_run_iterations) and (!no_update_lim_reached)) {

        
        #pragma omp parallel for num_threads(this->num_threads) shared(run_results, particles) //reduction(max : run_result)//implicit(none) private(i) shared(run_results, n_run_particles, particles)
        for(i=0;i<n_run_particles; i++) {
            if(no_update>195 && no_update<205) {random_param=0.04;}
            else {random_param=0.02;};

            particles[i].update_vel(particles[best_particle_idx].p, iter, random_param);
            particles[i].update_pos();
            run_results[i] = particles[i].compute_obj_and_update_best();
            }

        for(i=0;i<n_run_particles; i++){
            if(run_results[i] > best_val) {
                    best_val = run_results[i];
                    best_particle_idx = particles[i].particle_idx;
                    no_update= 0;
                    new_glob_best=true;
                }
            }

        if(verbose and (iter%100 == 0)){
            avg_velocity = 0;
            //for(i=0;i<n_tolls;++i) avg_velocity += particles[best_particle_idx].v[i];
            // avg_velocity = avg_velocity/n_tolls;
            if(verbose and (iter%1 == 0)) std::cout<<"iter "<<iter<<"  best_val: "<<best_val<<"    avg vel: "<<avg_velocity<<std::endl;

        }
        
        
        if (new_glob_best==false) no_update++;
        else no_update = 0;
    
        if (no_update>no_update_lim) {no_update_lim_reached = true;}

        iter++;

        }

        actual_final_iterations = iter;
        for(int i=0; i<n_run_particles; i++) 
        {particles_best[i] = particles[i].personal_best_val;}
    }


void Swarm2::updte_stats(){
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
        // std::cout<<p_means[0][toll]<<"   llllll"<<std::endl;
    }
}



double* Swarm2::get_stats_array(std::vector<std::vector<double>> v){
    double* array = new double[n_tolls*v.size()];
    for(size_t i=0; i< v.size(); i++)
        for(int t=0; t < n_tolls; t++){
             array[i*n_tolls + t] = v[i][t];
        }
    return array;
}

double* Swarm2::get_p_means(){return get_stats_array(p_means);}
double* Swarm2::get_v_means(){return get_stats_array(v_means);}
double* Swarm2::get_p_stds(){return get_stats_array(p_stds);}
double* Swarm2::get_v_stds(){return get_stats_array(v_stds);}


void Swarm2::print() {
    std::cout<<*this<<std::endl;
}

void Swarm2::print_particles(){
    for(int i=0; i< n_particles; i++ ) particles[i].print();
}


std::ostream& operator<<( std::ostream &os, Swarm2& s ) {
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
