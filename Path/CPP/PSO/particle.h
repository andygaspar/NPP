#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include "params.h"



/*
Particle class implements a particle object such that:
- ndim: dimensionality of the space
- p: coordinates of the particle's position in the space
- v: particle's velocity vector
- personal_best: coordinates associated with the best position found so far
          in the particle's history according to a given objective function

The particle's position can be initialized by hand or randomly in the space,
setting the limits.
The methods implemented aim to:
- update the position of the particle according PSO method
- update the velocity of the particle according PSO method
*/

class Particle {
    public:
    // particle identifiers
    int particle_idx;
    int lh_idx;
    double init_commodity_val;

    //magic parameters
    Params params; 

    // problem related
    short n_commodities;
    short n_tolls;
    double commodity_cost;
    double toll_cost;
    std::vector<std::vector<double>> transfer_costs;
    std::vector<double> search_ub;
    std::vector<double> search_lb;
    std::vector<double> search_range;
    std::vector<double> commodities_tax_free;
    std::vector<int> n_users;

    // PSO++
    double c_soc=1.49445;
    double c_cog=1.49445;
    double w=0.9;
    double fitness;
    std::vector<double> fitness_memb {0,0,0}; //[better, same, worse]
    std::vector<double> sigma_memb {0,0,0}; //[same, near, far]
    double sigma;
    double sigma_max;
    double L;
    double U;
    double random_velocity_adjustment;

    // running features
    std::vector<double> p;
    std::vector<double> p_past;
    std::vector<double> v;
    std::vector<double> personal_best;
    double personal_best_val;
    double current_run_val;
    double past_run_val;
    int count_iter;

    short seed;
    std::default_random_engine generator;
    
    
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    Particle() {}
    // search_ub, search_lb: search space bounds
    // init_ub, init_lb: initialization space bounds
    Particle(
            double* comm_tax_free, int* n_usr, double* trans_costs, double* search_ub_, double* search_lb_,
                    short n_comm, short n_to, int part_idx,  Params parameters, double d_max, short seed);
    ~Particle() {
    }

    void init_values(double* p_init, double* v_init);
    void init_vector_values(std::vector<double>& p_init, std::vector<double>& v_init);
    void update_fitness(double best);
    void update_sigma(double* g) {sigma = compute_distance(p,std::vector<double>(g, g + n_tolls));}
    void update_pos();
    void update_vel(std::vector<double> g, int iter, double random_component_dump);
    void update_best(double new_personal_best_val) {personal_best_val = new_personal_best_val; personal_best=p;}
    void update_inertia();
    void update_c_soc();
    void update_c_cog();
    void update_L();
    void update_U();
    void update_params(double* g, double best);

    void evaluate_fitness_memb();
    void evaluate_sigma_memb();

    double compute_obj_and_update_best();

    void print();
};


/*-----------------------------------------------------------------------------------*/
/*      Initialize the particle object with random velocity  and given position      */
/*-----------------------------------------------------------------------------------*/

Particle::Particle(double* comm_tax_free, int* n_usr, double* trans_costs, double* search_ub_, double* search_lb_,
                    short n_comm, short n_tolls_, int part_idx, Params parameters, double d_max, short seed) {

  
        params = parameters;
        particle_idx = part_idx;
        n_commodities=n_comm;
        n_tolls=n_tolls_;
        count_iter=0;
        search_ub = std::vector<double> (n_tolls);
        search_lb = std::vector<double> (n_tolls);
        sigma_max = d_max;
        
        init_commodity_val = params.init_commodity_val;
        L=0;
        U=0.01;

        v = std::vector<double> (n_tolls);
        p = std::vector<double> (n_tolls);

        std::mt19937 o;

    //         generators = std::vector<std::default_random_engine> (0);
    // for (int i = 0; i <  n_threads; i++) {
    //     std::default_random_engine engine;
    //     if(seed>=0) engine.seed(i);
    //     else engine = std::default_random_engine(r());
    //     generators.emplace_back(engine);

    // }
        
        if(seed >= 0) {
            generator.seed(seed);
        }
        else{
            generator = std::default_random_engine (o());
        }
        

        commodities_tax_free = std::vector<double> (n_commodities);
        for(int j=0; j< n_commodities; j++) commodities_tax_free[j] = comm_tax_free[j];

        transfer_costs = std::vector<std::vector<double>>(n_commodities);
        for(int j =0; j<n_commodities; j++)  {
            transfer_costs[j] = std::vector<double>(n_tolls);
            for(int k=0; k< n_tolls; k++) transfer_costs[j][k]=trans_costs[j*n_tolls + k];
        }
        
        n_users = std::vector<int> (n_commodities);
        for(int j=0; j< n_commodities; j++) {n_users[j] = n_usr[j];}
            for(int j=0; j< n_tolls; j++) {
                search_ub[j] = search_ub_[j];
                search_lb[j] = search_lb_[j];
            }
        search_range = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) search_range[j] = search_ub[j] - search_lb[j];
        
}

void Particle::init_values(double* p_init, double* v_init){

    v = std::vector<double> (n_tolls);
    p = std::vector<double> (n_tolls);

    for(int j=0; j< n_tolls; j++) {
            p[j] = p_init[j];
            v[j] = v_init[j];
        }

    p_past=p;

    personal_best = p;
    personal_best_val = 0;

}


void Particle::init_vector_values(std::vector<double>& p_init, std::vector<double>& v_init){


    p = p_init;
    v = v_init;
    

    p_past=p;

    personal_best = p;
    personal_best_val = 0;

}

void Particle::update_fitness(double best) {
        if (best!=0) fitness = (compute_distance(p,p_past)/sigma_max)*(std::max(best,past_run_val)-std::max(best,current_run_val))/(std::abs(personal_best_val));
        else fitness = 1;
}

void Particle::update_pos() {

    for (int i=0; i < n_tolls; i ++) {
        p_past[i] = p[i];
        p[i] = p[i] + v[i];
        if (p[i] >= search_ub[i]) {
            p[i]= search_ub[i];
            v[i] = -v[i];
            }
        if (p[i] <= search_lb[i]) {
            p[i]= search_lb[i];
            v[i] = -v[i];
            }
    }
}

void Particle::update_vel(std::vector<double> g, int iter, double random_component_dump=0.01) {
    count_iter++;
    double r;
    std::uniform_real_distribution<double> distribution;
    for(int i=0; i<n_tolls; i++) {
        random_velocity_adjustment = std::max((fitness_memb[1]+sigma_memb[0]),0.)*random_component_dump;

        distribution = std::uniform_real_distribution<double> (-0.2*(search_range[i]),0.5*(search_range[i]));
        r = distribution(generator);
        // double r = get_rand(-0.2*(search_range[i]),0.5*(search_range[i]));
        v[i] = w*v[i] + c_soc*((g[i] - p[i])) + c_cog*((personal_best[i] - p[i])) + random_velocity_adjustment*r;
        if (std::abs(v[i])>U*(search_range[i]))
            v[i] = U*(search_range[i])*v[i]/std::abs(v[i]);
        if (std::abs(v[i])<L*(search_range[i])){
            if (v[i]==0)
                v[i]=L*(search_range[i]);
            else
                v[i]=L*(search_range[i])*v[i]/std::abs(v[i]);
        }
    }
}

void Particle::update_inertia() {
    w = 0;
    w += (fitness_memb[2] + sigma_memb[0])*params.w1; 
    w += (fitness_memb[1] + sigma_memb[1])*params.w2;
    w += (fitness_memb[0] + sigma_memb[2])*params.w3;
}

void Particle::update_c_soc() {
    c_soc = 0;
    c_soc += (fitness_memb[0] + sigma_memb[1])*params.c_soc1;
    c_soc += (fitness_memb[1] + sigma_memb[0])*params.c_soc2;
    c_soc += (fitness_memb[2] + sigma_memb[2])*params.c_soc3;
}

void Particle::update_c_cog() {
    c_cog = 0;
    c_cog += (sigma_memb[2])*params.c_cog1;
    c_cog += (fitness_memb[2] + fitness_memb[1] + sigma_memb[0] + sigma_memb[1])*params.c_cog2;
    c_cog += (fitness_memb[0])*params.c_cog3;
}

void Particle::update_L() {
    L = 0;
    L += (fitness_memb[1] + fitness_memb[0] + sigma_memb[2])*params.L1;
    L += (sigma_memb[0] + sigma_memb[1])*params.L2;
    L += (fitness_memb[2])*params.L3;
}

void Particle::update_U() {
    U = 0;
    U += (sigma_memb[0])*params.U1;
    U += (fitness_memb[1] + fitness_memb[0] + sigma_memb[1])*params.U2;
    U += (fitness_memb[2] + sigma_memb[2])*params.U3;
}

void Particle::update_params(double* g, double best) {
    update_sigma(g);
    update_fitness(best);
    evaluate_sigma_memb();
    evaluate_fitness_memb();
    update_c_cog();
    update_c_soc();
    update_L();
    update_U();
    update_inertia();
}

void Particle::evaluate_fitness_memb() {
    if (fitness<=0){
        fitness_memb[0] = -fitness;
        fitness_memb[1] = 1+fitness;
        fitness_memb[2] = 0;
    }
    else{
        fitness_memb[0] = 0;
        fitness_memb[1] = 1-fitness;
        fitness_memb[2] = fitness;
    }
}

void Particle::evaluate_sigma_memb() {
    double l1 = params.limit_sigma_1;
    double l2 = params.limit_sigma_2;
    double l3 = params.limit_sigma_3;
    if (sigma<=l1*sigma_max) {
        sigma_memb[0] = 1;
        sigma_memb[1] = 0;
        sigma_memb[2] = 0;
    }
    if (sigma>l1*sigma_max && sigma<=l2*sigma_max) {
        sigma_memb[0] = (l2*sigma_max - sigma)/((l2-l1)*sigma_max);
        sigma_memb[1] = (sigma - l1*sigma_max)/((l2-l1)*sigma_max);
        sigma_memb[2] = 0;
    }
    if (sigma>l2*sigma_max && sigma<=l3*sigma_max) {
        sigma_memb[0] = 0;
        sigma_memb[1] = (l3*sigma_max - sigma)/((l3-l2)*sigma_max);
        sigma_memb[2] = (sigma - l2*sigma_max)/((l3-l2)*sigma_max);
    }
    if (sigma>l3*sigma_max) {
        sigma_memb[0] = 0;
        sigma_memb[1] = 0;
        sigma_memb[2] = 1;
    }
}

double Particle::compute_obj_and_update_best(){
    past_run_val = current_run_val;

    /* compute objective value */
    current_run_val=0;
    int i,j,cheapest_path_idx;
    cheapest_path_idx = -1;

    // std::cout<<"commodities "<<n_commodities<<std::endl;
    for(i=0; i<n_commodities; i++) {
        commodity_cost=init_commodity_val;
        bool found = false;
        for(j=0; j< n_tolls; j++) {
            //std::cout<<p[j]<<", ";
            toll_cost = p[j] + transfer_costs[i][j];
            if(toll_cost <= commodity_cost) {
                if (toll_cost < commodity_cost) {
                    commodity_cost = toll_cost;
                    cheapest_path_idx = j;
                }
                else {
                    if ( p[j] > p[cheapest_path_idx]) {
                        commodity_cost = toll_cost;
                        cheapest_path_idx = j;
                    }
                }
            }
        }
//        std::cout<<"["<<i<<"]"<<commodities_tax_free[i]<<">="<<commodity_cost<<std::endl;
        if(commodities_tax_free[i] >= commodity_cost) {
            found = true;
            // std::cout<<obj_coefficients[cheapest_path_idx]<<" "<<n_users[i]<<" "<< p[cheapest_path_idx]<<std::endl;
            // std::cout<<std::endl;
            current_run_val += p[cheapest_path_idx]*n_users[i];
        }

    }
    
    
    /* update personal_best and personal_best_val */
    if(current_run_val> personal_best_val){
        for(int i=0; i<n_tolls; i++) personal_best[i] = p[i];
        personal_best_val = current_run_val;
        count_iter=0;
    }
/*     std::cout<<current_run_val<<" personal"<<std::endl;
    print(); */
    return current_run_val;
}

void Particle::print() {
    std::cout<<"Transfer"<<std::endl;
    for(int i=0; i<n_commodities; i++) {
        for(int j=0; j<n_tolls; j++) std::cout<<transfer_costs[i][j]<<' ';
        std::cout<<std::endl;
    }
    std::cout<<std::endl<<"upper bounds"<<std::endl<<search_ub[0]<<std::endl;
    std::cout<<std::endl<<"n users"<<std::endl<<n_users[0]<<std::endl;
    std::cout<<std::endl<<"p"<<std::endl<<p[0]<<std::endl;
    
}

std::ostream& operator<<( std::ostream &os, Particle& v ) {
    std::cout<<"pos -> "<<v.p[0]<<" vel -> "<<v.v[0];
    return os;
}


