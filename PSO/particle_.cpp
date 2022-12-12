#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
//#include "utils.cpp"

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
    short n_commodities;
    short n_tolls;
    int idx;
    std::vector<std::vector<double>> transfer_costs;
    std::vector<double> upper_bounds;
    std::vector<double> commodities_tax_free;
    std::vector<int> n_users;
    std::vector<double> p;
    std::vector<double> v;
    std::vector<double> personal_best;
    double personal_best_val;
    double run_cost;
    double commodity_cost;
    double init_commodity_cost;
    double toll_cost;
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    Particle() {}

    Particle(double* comm_tax_free, int* n_usr, double* trans_costs, double* u_bounds, short n_comm, short n_to, int i) {
        idx = i;
        n_commodities=n_comm;
        n_tolls=n_to;

        commodities_tax_free = std::vector<double> (n_commodities);
        for(int i=0; i< n_commodities; i++) commodities_tax_free[i] = comm_tax_free[i];

        transfer_costs = std::vector<std::vector<double>>(n_commodities);
        for(int i =0; i<n_commodities; i++)  {
            transfer_costs[i] = std::vector<double>(n_tolls);
            for(int j=0; j< n_tolls; j++) transfer_costs[i][j]=trans_costs[i*n_tolls + j];
        }

        n_users = std::vector<int> (n_commodities);
        for(int i=0; i< n_commodities; i++) n_users[i] = n_usr[i];

        upper_bounds = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) upper_bounds[j] = u_bounds[j];

        p = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) p[j] = get_rand(0, 1) * upper_bounds[j];

        v= std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) v[j] = 0;

        personal_best = p;
        personal_best_val = 0;
        init_commodity_cost = pow(10, 5);

    }
    ~Particle() {}
    double get_personal_best_val() {return personal_best_val;}
    void update_pos();
    void reflection();
    void update_vel(double w, double c_soc, double c_cog, double* g);
    void update_best(double new_personal_best_val) {personal_best_val = new_personal_best_val; personal_best=p;}

    double get_rand(double start, double end) {
        std::default_random_engine generator(std::rand());
        std::uniform_real_distribution<double> distribution(start, end);
        return distribution(generator);
        }

    double compute_obj_and_update_best();
    void print();
};

void Particle::update_pos() {

    for (int i=0; i < n_tolls; i ++) {
        p[i] = p[i] + v[i];
        if (p[i] >= upper_bounds[i]) {
            std::default_random_engine generator(std::rand());
            std::uniform_real_distribution<double> distribution(0,0.8);
            p[i]= upper_bounds[i];
            v[i] -= distribution(generator)*v[i];
            }
        if (p[i] <= 0) {
            std::default_random_engine generator(std::rand());
            std::uniform_real_distribution<double> distribution(0,0.8);
            p[i]= 0.;
            v[i] -= distribution(generator)*v[i];
            }
    }
}

void Particle::update_vel(double w, double c_soc, double c_cog, double* g) {

    for(int i=0; i<n_tolls; i++) v[i] = w*v[i] + c_soc*((g[i] - p[i])) + c_cog*((personal_best[i] - p[i]));
}

double Particle::compute_obj_and_update_best(){
    run_cost=0;
    int i,j,cheapest_path_idx;
    for(i=0; i<n_commodities; i++) {

        commodity_cost=init_commodity_cost;
        bool found = false;
        //std::cout<<i<<"   "<<commodities_tax_free[i]<<std::endl;
        for(j=0; j< n_tolls; j++) {
            
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
        if(commodities_tax_free[i] >= commodity_cost) {
            found = true;
            run_cost += p[cheapest_path_idx]*n_users[i];
        }
        /*
        std::string not_free;
        if(found) not_free = "  True ";
        else  not_free = " False ";
        std::cout.precision(17);
        std::cout<<"comm " <<i<<"   p "<<cheapest_path_idx<<"   not free "<<not_free<<"   n users "<<n_users[i] <<"   transf "<< transfer_costs[i][cheapest_path_idx]<<
        "   p "<<p[cheapest_path_idx]<<"   cost "
           <<p[cheapest_path_idx] + transfer_costs[i][cheapest_path_idx] << "   free "<< commodities_tax_free[i]<<"   gain "
           <<p[cheapest_path_idx]*n_users[i]<<std::endl; */

    }


    if(run_cost> personal_best_val){
        for(int i=0; i<n_tolls; i++) personal_best[i] = p[i];
        personal_best_val = run_cost;
    }
    return run_cost;
}

void Particle::print() {
    std::cout<<"Transfer"<<std::endl;
    for(int i=0; i<n_commodities; i++) {
        for(int j=0; j<n_tolls; j++) std::cout<<transfer_costs[i][j]<<' ';
        std::cout<<std::endl;
    }
    std::cout<<std::endl<<"upper bounds"<<std::endl<<upper_bounds[0]<<std::endl;
    std::cout<<std::endl<<"n users"<<std::endl<<n_users[0]<<std::endl;
    std::cout<<std::endl<<"p"<<std::endl<<p[0]<<std::endl;
    
}

std::ostream& operator<<( std::ostream &os, Particle& v ) {
    std::cout<<"pos -> "<<v.p[0]<<" vel -> "<<v.v[0];
    return os;
}


