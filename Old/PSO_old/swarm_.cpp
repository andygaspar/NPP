#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include "particle_.cpp"
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
    int  stop_param=0;
    int n_subcubes;
    std::vector<Particle> particles;
    int no_update_lim;

    // latin hypercube parameters
    int N_div; //  number of dimensions which get partitioned (divided)
    double* div_start; //intervals start
    double* div_end; 
    int* div_dim;  // dimensions getting devided
    short n_p;
    short N_PARTICLES;
    short n_particles;
    double d_M;

    short* cube_best_particle_idx;
    double* cube_best_val;
    double** cube_p_best;
    int best_idx;
    
    // computation parameters
    short num_threads;
    bool verbose;

    


    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, short n_cut, short N_div, int n_u_l=500, bool verb=false);
    Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, short n_cut, short N_div, double* past_best, double* upp_b, int n_u_l=500);
    Swarm(std::vector<std::vector<double>> p_init,double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, int n_u_l=500);
    Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, int n_u_l=500);
    Swarm() {}

    void init_div(int n_cut);
    void set_init_sols(double* solutions, int n_solutions);
    void lower_particles(std::vector<double>& results);
    void centering_particles(int N, double var);

    void LH_sampling(int i, int n_cut, double* comm_tax_free, int* n_usr,double* transf_costs, int idx_part);
    void normal_sampling(double* cube_p_best, double* comm_tax_free, int* n_usr,double* transf_costs);

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
    N_PARTICLES = N_PARTS;
    n_p = n_particles;
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

/*-----------------------------------------------------------------------------------------*/
/* Initialize the swarm object and its particles with random velocity and random positions. */                                                                        
/*-----------------------------------------------------------------------------------------*/
Swarm::Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* const obj_coef, double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, int n_u_l) {
    n_iterations = n_iter;
    no_update_lim=n_u_l;
    n_particles=n_parts; 
    N_PARTICLES = N_PARTS;
    n_p = n_particles;
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
        particles.push_back({comm_tax_free, n_usr ,transf_costs,obj_coefficients, u_bounds,l_bounds, n_commodities, n_tolls, i, d_M,search_lb,search_ub,0});
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

/*-----------------------------------------------------------------------------------------*/
/*      Initialize the swarm object and its particles with random velocity and position    */
/*      implementing the latin hypercube sampling method in the bounded space.             */
/*-----------------------------------------------------------------------------------------*/
Swarm::Swarm(double* comm_tax_free, int* n_usr,double* transf_costs, double* const obj_coef, double* const u_bounds,double* const l_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th,short N_PARTS, short n_cut, short N_div_, int n_u_l, bool verb) {
    n_iterations = n_iter;
    no_update_lim=n_u_l;
    n_particles=n_parts; 
    N_PARTICLES = N_PARTS;
    n_p = n_particles;
    n_tolls=n_to;
    n_commodities=n_comm;
    obj_coefficients=obj_coef;
    search_ub = u_bounds;
    search_lb = l_bounds;
    num_threads=num_th;
    N_div = N_div_;
    verbose=verb;

    // find the number of dimensions on which the cuts are being performed, according with the desired number of particles 
    if (N_div==0) {
        while (std::pow(n_cut,N_div)<n_p) {N_div++;}
        N_div--;
    }
    n_p = int(std::ceil(n_p/std::pow(n_cut,N_div))*std::pow(n_cut,N_div));
    n_particles = n_p;
    n_subcubes=std::pow(n_cut,N_div);
    best_idx=0;
    cube_best_val = new double[n_subcubes];
    cube_p_best = new double*[n_subcubes];
    cube_best_particle_idx = new short[n_subcubes];

    for (int i=0;i<std::pow(n_cut,N_div);i++) cube_p_best[i] = new double[n_tolls];
    particles=std::vector<Particle>(0);

    d_M = 0;
    for (int i=0;i<n_tolls;i++)
        d_M += std::pow(u_bounds[i]-l_bounds[i],2);
    d_M = std::sqrt(d_M);

    // latin hypercube sampling method
    div_start = new double[n_tolls];
    div_end = new double[n_tolls];
    div_dim = new int[N_div];
    init_div(n_cut);
    LH_sampling(0, n_cut,comm_tax_free, n_usr ,transf_costs,0);

    // update the best_* things
    double tmp = 0; 
    for (int i=0;i<n_subcubes;i++) cube_best_val[i] = 0;
    for(int i=0;i<n_p;i++) {
        tmp = particles[i].compute_obj_and_update_best();
        if (tmp>cube_best_val[particles[i].lh_idx]) {
            cube_best_val[particles[i].lh_idx] = tmp;
            cube_best_particle_idx[particles[i].lh_idx] = i;
            cube_best_val[particles[i].lh_idx] = tmp;
            if (tmp>cube_best_val[best_idx]) best_idx=particles[i].lh_idx;
        }
    }
    for (int i=0;i<n_subcubes;i++) {
        cube_p_best[i] = new double[n_tolls];
        for(int j=0; j< n_tolls; j++) {
            cube_p_best[i][j]=particles[cube_best_particle_idx[i]].p[j];
        } 
    }
}

/*-----------------------------------------------------------------------------------------*/
/* Initialize the "div_start" and "div_end" vectors according to latin hypercube sampling. */                                                                      
/*-----------------------------------------------------------------------------------------*/
void Swarm::init_div(int n_cut) {
    // create a list of two columns: [0]--> toll_idx [1]--> lenght of the toll
    std::vector<std::array<double,2>> list_dim = create_list(search_lb,search_ub,n_tolls);
    // sort it w.r.t. the toll lenght
    std::sort(list_dim.begin(),list_dim.end(),compare);

    // div_dim --> collects the toll_idx for the ones selected for the cutting
    for (int i=0;i<N_div;++i) {div_dim[i]=list_dim[i][0];}
    // div_start and div_end are inizialized for each toll with the respective search_lb and search_ub
    for (int i=0;i<n_tolls;++i) {
        div_start[i] = search_lb[i];
        div_end[i] = search_ub[i];
    }
    // in case of the tolls selected for the cutting the div_end is initialized with the first cat value --> e.g. |--------|-------|-------|
    for (int i=0;i<N_div;++i) {
        div_end[div_dim[i]] = search_lb[div_dim[i]]+(search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
    }
}

void Swarm::set_init_sols(double* solutions, int n_solutions) {
    std::cout<<"   init solutions "<< n_solutions<<"  tolls"<<n_tolls<< std::endl;
    int n_sols = n_solutions;
    if(n_sols > n_p) n_sols = n_p;

    for(int i=0; i< n_sols; i ++) {
        for(int j=0; j< n_tolls; j++) particles[i].p[j] = solutions[i*n_tolls + j];
    }

    for(int j=0; j< n_tolls; j++) std::cout<<particles[0].p[j]<< " ";
    std::cout<< std::endl; 
    std::cout<<particles[0].compute_obj_and_update_best()<<std::endl;
    std::cout<< "  ++++++  " <<particles[0].personal_best_val<< std::endl;
    for (int j=0;j<n_subcubes;j++)
        for(int i=0; i< n_tolls; i++) cube_p_best[j][i]=particles[0].p[i];
}

/*-----------------------------------------------------------------------------------------*/
/*    Mantain in the population only the first n_particles in terms of better results    */                                                                        
/*-----------------------------------------------------------------------------------------*/
void Swarm::lower_particles(std::vector<double>& results) {

    std::vector<double> tmp(n_particles,0.);
    std::vector<double> tmp_results(n_p,0.);
    std::vector<short> tmp_cube_best_particle_idx(n_subcubes,-1);
    std::vector<Particle> new_particles(n_p);

    // collect and sort the results data in an increasing order to select the best ones
    std::vector<std::array<double,2>> list = create_list(tmp.data(),results.data(),results.size());
    std::sort(list.begin(),list.end(),compare);

    // update best_idx
    best_idx=particles[list[0][0]].lh_idx;

    // update new_particles, tmp_results and tmp_cube_best_particle_idx vectors
    for (int i=0;i<n_p;i++) {
        new_particles[i] = particles[list[i][0]];
        new_particles[i].p = new_particles[i].personal_best;
        new_particles[i].v = std::vector<double>(n_tolls,0.);
        new_particles[i].idx = i;
        tmp_results[i] = list[i][1];
        if (cube_best_particle_idx[particles[list[i][0]].lh_idx]==list[i][0]){
            tmp_cube_best_particle_idx[particles[list[i][0]].lh_idx]=i;
        }
    }

    // substitute the real vectors with the new ones
    cube_best_particle_idx = tmp_cube_best_particle_idx.data();
    particles = new_particles;
}

/* change the particles generating them around the best one so far with a normal distribution */
void Swarm::centering_particles(int N, double var) {

    std::vector<std::vector<double>> p_init(0);
    p_init.push_back(particles[cube_best_particle_idx[best_idx]].personal_best);
    for (int i=1;i<N;i++) {
        // generate a random position with normal distribution around the best particle position
        std::vector<double> pos(particles[0].n_tolls);
        for (int k=0;k<n_tolls;k++) {
            pos[k]=get_normal(particles[cube_best_particle_idx[best_idx]].personal_best[k],var);
            if (pos[k]<search_lb[k]) pos[k]=search_lb[k];
            if (pos[k]>search_ub[k]) pos[k]=search_ub[k];
        }
        p_init.push_back(pos);
    }

    std::vector<double> t_cost(particles[0].n_commodities*particles[0].n_tolls);
    for (int i=0;i<particles[0].n_commodities*particles[0].n_tolls;i++) 
        t_cost[i]=particles[0].transfer_costs[std::floor(i/particles[0].n_tolls)][i%particles[0].n_tolls];

    std::vector<double> com_t_f = particles[0].commodities_tax_free;
    std::vector<int> n_u = particles[0].n_users;
    int i_lh = particles[0].lh_idx;


    n_iterations = n_iterations;
    n_particles=N; 
    N_PARTICLES = N_PARTICLES;
    n_p = n_particles;
    n_tolls=particles[0].n_tolls;
    n_commodities=particles[0].n_commodities;
    search_ub = particles[0].search_ub.data();
    search_lb = particles[0].search_lb.data();
    num_threads=num_threads;

    particles=std::vector<Particle>(0);

    for(int i=0;i<N;++i){
        particles.push_back({p_init[i], com_t_f.data(), n_u.data() ,t_cost.data(), obj_coefficients, search_ub,search_lb, n_commodities, n_tolls, i, d_M,i_lh});
    }

    double tmp = 0; 
    cube_best_val[best_idx] = 0;
    for(int i=0;i<N;++i) {
        tmp = particles[i].compute_obj_and_update_best();
        if (tmp>cube_best_val[best_idx]){
            cube_best_particle_idx[best_idx] = i;
            cube_best_val[best_idx] = tmp;
        }
    }
    std::cout<<cube_best_particle_idx[best_idx]<<" "<<cube_best_val[best_idx]<<std::endl;
    for(int i=0; i< n_tolls; i++) cube_p_best[best_idx][i]=particles[cube_best_particle_idx[best_idx]].p[i];
    no_update=0;
}


/*----------------------------------------------------------------------------------------------------------------------------------*/
/* Perform LH sampling                                                                                                              */     
/* i -> during this call of LH_sampling the i-th (according to div_dim) toll dimension's considered portion is modified             */
/* count -> takes trace of the already considered portions of the i-th toll dimension from those indicated in div_dim               */
/*                                                                                                                                  */
/* Example of functioning: N_div=2, n_cut=2 (so n_subcubes=2^2=4), n_p=6                                                            */
/*    -----------------------                                                                                                       */
/*    |   SC2    |   SC4    |  d                                                                                                    */ 
/*    -----------------------  i                                                                                                    */ 
/*    |   SC1    |   SC3    |  m                                                                                                    */ 
/*    -----------------------  1                                                                                                    */
/*             dim 0                                                                                                                */
/*                                                                                                                                  */
/* START CYCLE 1 -> i=0, count=0, idx_part=0 -> while OK -> if OK -> START CYCLE 2 with i=1, idx_part=0 in SC1                      */
/*                                                                                                                                  */          
/* CYCLE 2 with i=1, count=0, idx_part=0 in SC1 -> while OK -> if NO -> if NO -> for OK -> ADD PARTICLE with idx=0 in SC1 ->        */
/* -> idx_part=1 -> ADD PARTICLE with idx=1 in SC1 -> idx_part=2 -> count=1 -> if NO -> go in SC2 -> while OK -> if NO -> if OK ->  */ 
/* -> idx_part=2 -> for OK -> ADD PARTICLE with idx=2 in SC2 -> idx_part=3 -> ADD PARTICLE with idx=3 in SC2 -> idx=4 -> count=2 -> */ 
/* -> if OK -> go in SC1 -> END CYCLE 2                                                                                             */  
/*                                                                                                                                  */ 
/* RESTART CYCLE 1 -> count=1 -> go in SC3 -> while OK -> if OK -> START CYCLE 3 with i=1, idx_part=0 in SC3                        */
/*                                                                                                                                  */     
/* CYCLE 3 -> i=1, count=0, idx_part=0 in SC3 -> while OK -> if NO -> if OK -> idx_part=4 -> for OK ->                              */
/* -> ADD PARTICLE with idx=4 in SC3 -> idx_part=5 -> ADD PARTICLE with idx=5 in SC3 -> idx_part=6 -> count=1 -> if NO ->           */
/* -> go in SC4 -> while OK -> if NO -> if OK -> idx_part=6 -> for OK -> ADD PARTICLE with idx=6 in SC4 -> idx_part=7 ->            */
/* -> ADD PARTICLE with idx=7 in SC4 -> idx_part=8 -> count=2 -> if OK -> go in SC3 -> END CYCLE 2                                  */  
/*                                                                                                                                  */ 
/* RESTART CYCLE 1 -> count=2 -> go out of the space for dim_div[0] -> while NO -> go in SC1  -> END CYCLE 1                        */ 
/*                                                                                                                                  */ 
/*  NOTE ----> at the end the number of particles is not n_p=6, but is equal to ceil(n_p/(n_subcubes))*n_subcubes = 8               */
/*----------------------------------------------------------------------------------------------------------------------------------*/
void Swarm::LH_sampling(int i, int n_cut, double* comm_tax_free, int* n_usr,double* transf_costs,int idx_part) {
    // count mantains the information about the portion of the toll dimension we are considering
    int count=0; 
    while (count<n_cut) {
        if (i<N_div-1) {
            LH_sampling(i+1, n_cut,comm_tax_free, n_usr ,transf_costs, idx_part);
            count++;
            div_start[div_dim[i]] += (search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
            div_end[div_dim[i]] += (search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
        } 
        else {
            if (particles.size()>0) idx_part=particles[particles.size()-1].idx+1;
            for (int k=0;k<n_p/std::pow(n_cut,N_div);++k) {
                particles.push_back({comm_tax_free, n_usr ,transf_costs, obj_coefficients, search_ub,search_lb, n_commodities, n_tolls, idx_part, d_M,div_start,div_end,(int)std::floor(idx_part/(n_p/std::pow(n_cut,N_div)))});
                idx_part++;
            }
            count++;
            if (count==n_cut) {
                div_start[div_dim[i]] = search_lb[div_dim[i]];
                div_end[div_dim[i]] = search_lb[div_dim[i]] + (search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
            }
            else {
                div_start[div_dim[i]] += (search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
                div_end[div_dim[i]] += (search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
            }
        }
        //std::cout<<idx_part<<"/"<<n_p<<std::endl;
    }
    div_start[div_dim[i]] = search_lb[div_dim[i]];
    div_end[div_dim[i]] = search_lb[div_dim[i]] + (search_ub[div_dim[i]]-search_lb[div_dim[i]])/n_cut;
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