#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "utils.cpp"

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
    private:
    short n_commodities;
    short n_tolls;
    int idx;
    double** transfer_costs;
    Vector<double> upper_bounds;
    Vector<double> commodities_tax_free;
    Vector<int> n_users;
    Vector<double> p;
    Vector<double> v;
    Vector<double> personal_best;
//    Vector<double> actual_costs;
    double personal_best_val;
    double run_cost;
    double commodity_cost;
    double init_commodity_cost;
    double toll_cost;
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    public:
    Particle() {}

    Particle(double* comm_tax_free, int* n_usr, double* trans_costs, double* u_bounds, short n_comm, short n_to, int i) {
        idx = i;
        n_commodities=n_comm;
        n_tolls=n_to;

        std::vector<double> vect_com(comm_tax_free, comm_tax_free + n_commodities);
        commodities_tax_free = Vector<double> {vect_com};
        int size = sizeof(n_users[0]);
        transfer_costs = new double*[n_comm];
        for(int i =0; i<n_commodities; i++) transfer_costs[i]=&trans_costs[i*n_tolls];

        std::vector<int> vect_n_usr(n_usr, n_usr + n_commodities);
        n_users = Vector<int> {vect_n_usr};        

        std::vector<double> vect_up(u_bounds, u_bounds + n_tolls);
        upper_bounds = Vector<double> {vect_up};

        p = Vector<double>(n_tolls, 0, 1);
        p = p * upper_bounds;
        v= Vector<double> {n_commodities};
        personal_best = p;
        personal_best_val = pow(10, 6);
        init_commodity_cost = pow(10, 5);

    }
    ~Particle() {}
    double get_personal_best_val() {return personal_best_val;}
    void update_pos();
    void reflection();
    void update_vel(double w, double c_soc, double c_cog, Vector<double> g);
    void update_best(double new_personal_best_val) {personal_best_val = new_personal_best_val; personal_best=p;}
    void set_pos(Vector<double> new_pos) {p=new_pos;}
    void set_vel(Vector<double> new_vel) {v=new_vel;}
    Particle(const Particle& new_particle) {
        n_commodities = new_particle.n_commodities;
        n_tolls = new_particle.n_tolls;
        idx = new_particle.idx;
        transfer_costs = new_particle.transfer_costs;
        upper_bounds = new_particle.upper_bounds;
        commodities_tax_free = new_particle.commodities_tax_free;
        n_users = new_particle.n_users;
        p = new_particle.p;
        v = new_particle.v;
        personal_best = new_particle.personal_best;
    //    Vector<double> actual_costs = new_particle.n_tolls;
        personal_best_val = new_particle.personal_best_val;
        run_cost = new_particle.run_cost;
        commodity_cost = new_particle.commodity_cost;
        init_commodity_cost = new_particle.init_commodity_cost;
        toll_cost = new_particle.toll_cost;
    }
    Vector<double> p_b() {return personal_best;}
    Vector<double> pos() {return p;}
    Vector<double> vel() {return v;}
    double compute_obj_and_update_best();
    void print();
};

void Particle::update_pos() {

    for (int i=0; i < p.size(); i ++) {
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

void Particle::update_vel(double w, double c_soc, double c_cog, Vector<double> g) {
    Vector<double> r1{p.size(), 0.0, 1.0};
    Vector<double> r2{p.size(), 0.0, 1.0};
    v = w*v + c_soc*(r1*(g - p)) + c_cog*(r2*(personal_best - p));
}

double Particle::compute_obj_and_update_best(){
    run_cost=0;
    int i,j,cheapest_path_idx;
    for(i=0; i<n_commodities; i++) {
        commodity_cost=init_commodity_cost;
        for(j=0; j< n_tolls; j++) {
            toll_cost = p[j] + transfer_costs[i][j];
            if(toll_cost < commodity_cost) {
                commodity_cost = toll_cost;
                cheapest_path_idx = j;
                }
        }
        if(commodities_tax_free[i] >= commodity_cost) run_cost += p[cheapest_path_idx]*n_users[i];
    }

    if(run_cost< personal_best_val){
        personal_best = p;
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
    std::cout<<std::endl<<"upper bounds"<<std::endl<<upper_bounds<<std::endl;
    std::cout<<std::endl<<"n users"<<std::endl<<n_users<<std::endl;
    std::cout<<std::endl<<"p"<<std::endl<<p<<std::endl;
    
}

std::ostream& operator<<( std::ostream &os, Particle& v ) {
    std::cout<<"pos -> "<<v.p<<" vel -> "<<v.v;
    return os;
}


