#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include "swarm.h"


class GeneticOperators {
    public:
    // problem related features
    short pop_size;
    short n_paths;
    short off_size;
    double** upper_bounds;
    short offs_size;
    std::vector<int> a_combs;
    std::vector<int> b_combs;
    double mutation_rate;
    short recombination_size;
    std::vector<int> random_order;
    std::vector<std::vector<int>> random_element_order;
    short n_threads;
    short start_index;
    short n_commodities;

    double current_run_val;
    double commodity_cost;
    double init_commodity_val;

    double** commodities_tax_free;
    double*** transfer_costs;
    short** n_users;




    GeneticOperators(double* upper_bounds_, double* comm_tax_free, short* n_usr, double* trans_costs, short n_commodities_, short pop_size_, short off_size_, short n_paths_, double mutation_rate_, short recombination_size_, short num_threads_){

    pop_size = pop_size_;
    n_paths = n_paths_;
    off_size = off_size_;
    mutation_rate = mutation_rate_;
    recombination_size = recombination_size_;
    n_threads = num_threads_;
    start_index = pop_size * n_paths;


    n_commodities = n_commodities_;
    init_commodity_val = init_commodity_val=pow(10, 5);
    std::random_device rnd_device;
    
    commodities_tax_free = new double*[n_threads];
    for(int i=0; i< n_threads; i++){
        commodities_tax_free[i] = new double[n_commodities];
        for(int j=0; j< n_commodities; j++) commodities_tax_free[i][j] = comm_tax_free[j];
        }


    transfer_costs = new double**[n_threads];
    for(int i=0; i< n_threads; i++){
        transfer_costs[i] = new double*[n_commodities];
        for(int j =0; j<n_commodities; j++)  {
                transfer_costs[i][j] = new double[n_paths];
                for(int k=0; k< n_paths; k++) transfer_costs[i][j][k]=trans_costs[j*n_paths + k];
            }
        }

    n_users = new short*[n_threads];
    for(int i=0; i< n_threads; i++){
        n_users[i] = new short[n_commodities];
        for(int j=0; j< n_commodities; j++) n_users[i][j] = n_usr[j];
    }

    a_combs = std::vector<int> (0);
    b_combs = std::vector<int> (0);
    for(short i=0; i < pop_size - 1; i++){
        for(short j=i + 1; j < pop_size; j++) {
            a_combs.push_back(i);
            b_combs.push_back(j);
        }
    }
    random_order = std::vector<int> (0);
    for(int i=0; i<a_combs.size(); i++) random_order.push_back(i);

    random_element_order = std::vector<std::vector<int>> (n_threads);
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_paths; j++) random_element_order[i].push_back(i);
        }

    upper_bounds = new double*[n_threads];
    for(int i=0; i<n_threads; i++){
        upper_bounds[i] = new double[n_paths];
        for(int j=0; j<n_paths; j++) upper_bounds[i][j] = upper_bounds_[j];
        }
    }


    ~GeneticOperators(){
        for(int i=0; i<n_threads; i++) {
            delete[] upper_bounds[i];
            delete[] n_users[i];
            delete[] commodities_tax_free[i];
            for(int j =0; j<n_commodities; j++) delete[] transfer_costs[i][j];

        }
        delete[] n_users;
        delete[] transfer_costs;
        delete[] commodities_tax_free;
        delete[] upper_bounds;
    }

    void reshuffle_element_order(std::vector<int> &vect){
        int idx, temp;
        for(int i=0; i<vect.size(); i++) {
            // std::cout<<"size combs "<<vect.size()<<std::endl;
            idx = get_rand_idx(0, vect.size() - 1);
            // std::cout<<"idx "<<idx<<std::endl;
            // std::cout<<"val idx "<<vect[idx]<<std::endl;
            if(idx!=i){
                vect[i] += vect[idx];
                vect[idx] = vect[i] - vect[idx];
                vect[i] -= vect[idx];
            }

        }
    }

    void generate(double* a_parent, double* b_parent, double* child, double* u_bounds, std::vector<int> element_order){
        int i;
        for(i=0; i<n_paths; i++) child[i] = a_parent[i];
        reshuffle_element_order(element_order);
        for(i=0; i < recombination_size; i++) child[element_order[i]] = b_parent[element_order[i]];
        
        for(i=0; i<n_paths; i++) {
            if(get_rand(0., 1.) < mutation_rate) child[i] = get_rand(0., u_bounds[i]);
        }
        // print_vect(child, n_paths);

    }

    double* generation(double* population){
        double* children = new double[off_size * n_paths];
        reshuffle_element_order(random_order);
        #pragma omp parallel for num_threads(n_threads) schedule(static) shared(population)
        for(short i=0; i < off_size; i++) {
            // std::cout<<a_combs[random_order[i]]<<" "<<b_combs[random_order[i]]<<" "<<population[a_combs[random_order[i]] * n_paths]<<std::endl;
            short th = omp_get_thread_num();

        // print_vect(&population[n_paths*i], n_paths);

            generate(&population[a_combs[random_order[i]] * n_paths], &population[b_combs[random_order[i]] * n_paths], &children[i * n_paths], 
                        upper_bounds[th], random_element_order[th]);
            }
        return children;

    }

    double* eval_parallel(double* new_population) {
        double* vals = new double[off_size];
        #pragma omp parallel for num_threads(n_threads) shared(new_population, vals)
        for(int i=0; i < off_size; i++) {
            short th = omp_get_thread_num();
            vals[i] = eval(&new_population[ i * n_paths], transfer_costs[th], commodities_tax_free[th], n_users[th]);
        }
        return vals;
    }


    double eval(double* p, double** trans_costs, double* comm_tax_free, short* n_usr){

    /* compute objective value */
    double current_run_val=0;
    double toll_cost;
    int i,j,cheapest_path_idx;

    for(i=0; i<n_commodities; i++) {
        commodity_cost=init_commodity_val;
        bool found = false;
        for(j=0; j< n_paths; j++) {

            toll_cost = p[j] + trans_costs[i][j];
            //std::cout<<"commodities "<<p[j]<< " " <<trans_costs[i][j]<<std::endl;
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

        if(comm_tax_free[i] >= commodity_cost) {
            found = true;
            current_run_val += p[cheapest_path_idx]*n_usr[i];
        }

    }
    


    return current_run_val;
}


};









