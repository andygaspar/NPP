#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include "utils_.cpp"

double INIT_COMMODITY_VAL = pow(10, 5);



struct MagicNumbers{
    double w1; 
    double w2;
    double w3;
    double c_soc1;
    double c_soc2;
    double c_soc3;
    double c_cog1;
    double c_cog2;
    double c_cog3;
    double L1;
    double L2;
    double L3;
    double U1;
    double U2;
    double U3;
    double limit_sigma_1;
    double limit_sigma_2;
    double limit_sigma_3;

    MagicNumbers(double ww1 = 0.1, 
    double ww2 = 0.5,
    double ww3 = 0.9,
    double cc_soc1 = 0.2,
    double cc_soc2 = 0.3,
    double cc_soc3 =1.5,
    double  cc_cog1 =0.2,
    double cc_cog2 =0.9,
    double cc_cog3 =1.1,
    double lL1 =0,
    double lL2 =0.001,
    double lL3 =0.01,
    double  uU1 =0.08,
    double uU2 =0.25,
    double uU3=0.5,
    double ll1 = 0.2,
    double ll2 = 0.4,
    double ll3 = 0.7){
    w1 = ww1;  w2 = ww2; w3 = ww3; c_soc1 = cc_soc1; c_soc2 = cc_soc2; c_soc3 =cc_soc3; c_cog1 = cc_cog1; c_cog2 = cc_cog2; c_cog3 =cc_cog3;
    L1 =lL1; L2 = lL2; L3 =lL3; U1 = uU1; U2 =uU2; U3=uU3; limit_sigma_1 = ll1; limit_sigma_2 = ll2; limit_sigma_3 = ll3;

    }


};


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
    int idx;
    int lh_idx;
    double init_commodity_val;

    //magic parameters
    MagicNumbers magic_numbers; 

    // problem related
    short n_commodities;
    short n_tolls;
    double commodity_cost;
    double toll_cost;
    std::vector<std::vector<double>> transfer_costs;
    std::vector<double> obj_coefficients;
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
    
    
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    Particle() {}
    // search_ub, search_lb: search space bounds
    // init_ub, init_lb: initialization space bounds
    Particle(double* comm_tax_free, int* n_usr, double* trans_costs, double* obj_coef, double* search_ub,double* search_lb, short n_comm, short n_to, int i, double d_max,double* const init_lb, double* const init_ub, int lh_id, MagicNumbers mag_num);
    Particle(std::vector<double> p_init, double* comm_tax_free, int* n_usr, double* trans_costs, double* obj_coef, double* search_ub,double* search_lb, short n_comm, short n_to, int i, double d_max, int lh_id, MagicNumbers mag_num);
    ~Particle() {}
    void set_values(std::vector<double> pos, double* comm_tax_free, int* n_usr, double* trans_costs, double* obj_coef, double* search_ub_,double* search_lb_, short n_comm, short n_to, int i, double d_max, int lh_id,  MagicNumbers mag_num);
    void update_fitness(double best);
    void update_sigma(double* g) {sigma = compute_distance(p,std::vector<double>(g, g + n_tolls));}
    void update_pos();
    void update_vel(double* g, int iter, int stop, int& stop_param, int best_idx, double random_component_dump);
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

/*-----------------------------------------------------------------------------------------*/
/*      Initialize the particle object with random velocity and position depending on      */
/*      the "start" and "end" vectors passed, which limit the space of initialization      */
/*-----------------------------------------------------------------------------------------*/
Particle::Particle(double* comm_tax_free, int* n_usr, double* trans_costs, double* obj_coef, double* search_ub_,double* search_lb_, short n_comm, short n_to, int i, double d_max,double* const init_lb, double* const init_ub, int lh_id, MagicNumbers mag_num = MagicNumbers()) {
        
        std::vector<double> pos(n_to);
        for(int j=0; j< n_to; j++) {
            pos[j] = get_rand(init_lb[j], init_ub[j]);
        }
        set_values(pos, comm_tax_free, n_usr, trans_costs, obj_coef, search_ub_, search_lb_, n_comm, n_to, i, d_max, lh_id, mag_num);
}

/*-----------------------------------------------------------------------------------*/
/*      Initialize the particle object with random velocity  and given position      */
/*-----------------------------------------------------------------------------------*/

Particle::Particle(std::vector<double> p_init, double* comm_tax_free, int* n_usr, double* trans_costs, double* obj_coef, double* search_ub_,double* search_lb_, short n_comm, short n_to, int i, double d_max, int lh_id,MagicNumbers mag_num = MagicNumbers()) {
    set_values(p_init, comm_tax_free, n_usr, trans_costs, search_ub_, search_lb_, obj_coef, n_comm, n_to, i, d_max, lh_id, mag_num);
}

void Particle::set_values(std::vector<double> p_init, double* comm_tax_free, int* n_usr, double* trans_costs, double* obj_coef, double* search_ub_,double* search_lb_, short n_comm, short n_to, int i, double d_max, int lh_id, MagicNumbers mag_num ) {
        magic_numbers = mag_num;
        idx = i;
        n_commodities=n_comm;
        n_tolls=n_to;
        sigma_max = d_max;
        count_iter=0;
        lh_idx=lh_id;

        commodities_tax_free = std::vector<double> (n_commodities);
        for(int j=0; j< n_commodities; j++) commodities_tax_free[j] = comm_tax_free[j];

        transfer_costs = std::vector<std::vector<double>>(n_commodities);
        for(int j =0; j<n_commodities; j++)  {
            transfer_costs[j] = std::vector<double>(n_tolls);
            for(int k=0; k< n_tolls; k++) transfer_costs[j][k]=trans_costs[j*n_tolls + k];
        }
        
        n_users = std::vector<int> (n_commodities);
        for(int j=0; j< n_commodities; j++) n_users[j] = n_usr[j];

        obj_coefficients = std::vector<double> (n_tolls);
        search_ub = std::vector<double> (n_tolls);
        search_lb = std::vector<double> (n_tolls);

        for(int j=0; j< n_tolls; j++) {
            obj_coefficients[j] = obj_coef[j];
            search_ub[j] = search_ub_[j];
            search_lb[j] = search_lb_[j];
        }
        search_range = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) search_range[j] = search_ub[j] - search_lb[j];

        // init positions
        p = p_init;
        p_past=p;

        v= std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) v[j] = get_rand(0, 0.) * (search_ub[j]-search_lb[j]);

        personal_best = p;
        personal_best_val = 0;
        init_commodity_val = pow(10, 5);
        L=0;
        U=10;
        
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

void Particle::update_vel(double* g, int iter, int stop, int& stop_param, int best_idx, double random_component_dump=0.01) {
    count_iter++;
    for(int i=0; i<n_tolls; i++) {
        random_velocity_adjustment = std::max((fitness_memb[1]+sigma_memb[0]),0.)*random_component_dump;
        //if (iter<stop || count_iter>200) random_velocity_adjustment = (fitness_memb[1]+sigma_memb[0])*0.09;
        //if ((count_iter>1500) && (idx==best_idx)) stop_param=1;
        double r = get_rand(-0.2*(search_range[i]),0.5*(search_range[i]));
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
    w += (fitness_memb[2] + sigma_memb[0])*magic_numbers.w1; 
    w += (fitness_memb[1] + sigma_memb[1])*magic_numbers.w2;
    w += (fitness_memb[0] + sigma_memb[2])*magic_numbers.w3;
}

void Particle::update_c_soc() {
    c_soc = 0;
    c_soc += (fitness_memb[0] + sigma_memb[1])*magic_numbers.c_soc1;
    c_soc += (fitness_memb[1] + sigma_memb[0])*magic_numbers.c_soc2;
    c_soc += (fitness_memb[2] + sigma_memb[2])*magic_numbers.c_soc3;
}

void Particle::update_c_cog() {
    c_cog = 0;
    c_cog += (sigma_memb[2])*magic_numbers.c_cog1;
    c_cog += (fitness_memb[2] + fitness_memb[1] + sigma_memb[0] + sigma_memb[1])*magic_numbers.c_cog2;
    c_cog += (fitness_memb[0])*magic_numbers.c_cog3;
}

void Particle::update_L() {
    L = 0;
    L += (fitness_memb[1] + fitness_memb[0] + sigma_memb[2])*magic_numbers.L1;
    L += (sigma_memb[0] + sigma_memb[1])*magic_numbers.L2;
    L += (fitness_memb[2])*magic_numbers.L3;
}

void Particle::update_U() {
    U = 0;
    U += (sigma_memb[0])*magic_numbers.U1;
    U += (fitness_memb[1] + fitness_memb[0] + sigma_memb[1])*magic_numbers.U2;
    U += (fitness_memb[2] + sigma_memb[2])*magic_numbers.U3;
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
    double l1 = magic_numbers.limit_sigma_1;
    double l2 = magic_numbers.limit_sigma_2;
    double l3 = magic_numbers.limit_sigma_3;
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
    for(i=0; i<n_commodities; i++) {
        commodity_cost=init_commodity_val;
        bool found = false;
        for(j=0; j< n_tolls; j++) {
            //std::cout<<p[j]<<", ";
            toll_cost = p[j]*obj_coefficients[j] + transfer_costs[i][j];
            if(toll_cost <= commodity_cost) {
                if (toll_cost < commodity_cost) {
                    commodity_cost = toll_cost;
                    cheapest_path_idx = j;
                }
                else {
                    if ( p[j]*obj_coefficients[j] > p[cheapest_path_idx]*obj_coefficients[cheapest_path_idx]) {
                        commodity_cost = toll_cost;
                        cheapest_path_idx = j;
                    }
                }
            }
        }
        //std::cout<<"["<<i<<"]"<<commodities_tax_free[i]<<">="<<commodity_cost<<std::endl;
        if(commodities_tax_free[i] >= commodity_cost) {
            found = true;
            current_run_val += p[cheapest_path_idx]*obj_coefficients[cheapest_path_idx]*n_users[i];
        }
    }
    //std::cout<<current_run_val<<std::endl;
    
    /* update personal_best and personal_best_val */
    if(current_run_val> personal_best_val){
        for(int i=0; i<n_tolls; i++) personal_best[i] = p[i];
        personal_best_val = current_run_val;
        count_iter=0;
    }
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


