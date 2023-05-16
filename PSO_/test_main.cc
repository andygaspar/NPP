#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include "swarm.h"
#include <jsoncpp/json/json.h>
#include <fstream>




int main() {
    std::ifstream people_file("params.json", std::ifstream::binary);
    Json::Value people;
    people_file >> people;
    std::cout<<"mmm"<<std::endl;

    std::cout<<people<<std::endl;

    std::cout<<"mmm"<<std::endl;

    // int age = people["Anna"]["age"].asInt();

    short n_comm = 4;
    short n_tolls_=10;
    short n_parts = 5;
    int n_iter = 100;
    int no_update_lim_ = 20; 
    short num_th = 4;

    double* comm_tax_free = new double[n_comm];
    int* n_usr = new int[n_comm];
    double* transf_costs = new double[n_tolls_*n_comm];
    double* obj_coef = new double[n_tolls_];

    for(int i = 0; i < n_comm; i++) {
        comm_tax_free[i] = get_rand(0,10);
        n_usr[i] = i;
    }

    for(int i = 0; i < n_tolls_; i++) {
        transf_costs[i] = get_rand(0,10);
        obj_coef[i] = get_rand(0,10);
        for(int j=0; j<n_comm; j++) transf_costs[j*n_tolls_ + i] = get_rand(0,10);
    }
    std::cout<<"ciao cicco"<<std::endl;

    Swarm* swarm = new Swarm{comm_tax_free, n_usr, transf_costs, obj_coef, n_comm, n_tolls_, n_parts, n_iter, no_update_lim_, num_th};

    double* p_init = new double[n_tolls_*n_parts];
    double* v_init = new double[n_tolls_*n_parts];
    double* u_bounds = new double[n_tolls_];
    double* l_bounds = new double[n_tolls_];


    for(int i = 0; i < n_tolls_; i++) {
        u_bounds[i] = 1;
        l_bounds[i] = 0;
        for(int j=0; j<n_parts; j++){
            p_init[i + j*n_tolls_] = get_rand(0,1);
            v_init[i + j*n_tolls_] = get_rand(-0.1,0.1);
        }
    }

    
    swarm->run( p_init, v_init, u_bounds, l_bounds, true, false);
    std::cout<<""<<std::endl;
    for(int i = 0; i < n_tolls_; i++){
        std::cout<<swarm->particles[swarm->best_particle_idx].p[i]<<" ";
    }
    std::cout<<"ciao cicco"<<std::endl;


    delete[] comm_tax_free;
    delete[] n_usr;
    delete[] transf_costs;
    delete[] obj_coef;
    delete[] p_init;
    delete[] v_init;
    delete[] u_bounds;
    delete[] l_bounds;


}