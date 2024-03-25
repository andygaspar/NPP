#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include "swarm.h"


class Genetic {
    public:
    // problem related features
    short pop_size;
    short n_paths;
    short off_size;
    short pop_total_size;
    short pso_size;
    double** upper_bounds;
    short n_commodities;

    std::vector<std::vector<double>> population;
    std::vector<double> vals;
  
    std::vector<int> a_combs;
    std::vector<int> b_combs;
    double mutation_rate;
    short recombination_size;
    std::vector<int> random_order;
    std::vector<std::vector<int>> random_element_order;
    short n_threads;
    std::vector<int> indices;
    short start_index;
    

    double current_run_val;
    double commodity_cost;
    double init_commodity_val;

    double** commodities_tax_free;
    double*** transfer_costs;
    int** n_users;
    double* t_costs;

    std::vector<std::default_random_engine> generators;
    





    Genetic(double* upper_bounds_, double* comm_tax_free, short* n_usr, double* trans_costs, short n_commodities_, short pop_size_, short off_size_, short n_paths_, double mutation_rate_, short recombination_size_, short num_threads_){

    pop_size = pop_size_;
    n_paths = n_paths_;
    off_size = off_size_;
    pso_size = 8;
    pop_total_size = pop_size + off_size + pso_size;

    mutation_rate = mutation_rate_;
    recombination_size = recombination_size_;
    n_threads = num_threads_;
    start_index = pop_size * n_paths;
    t_costs = trans_costs;

    std::random_device r;
    generators = std::vector<std::default_random_engine> (0);
    for (int i = 0; i <  n_threads; i++) {
        generators.emplace_back(std::default_random_engine(r()));
    }

    population = std::vector<std::vector<double>> (pop_total_size, std::vector<double>(n_paths, 0));
    
    vals = std::vector<double> (pop_total_size, 0);

    n_commodities = n_commodities_;
    init_commodity_val = init_commodity_val=pow(10, 5);
    
    indices = std::vector<int> (pop_total_size);
    std::iota(indices.begin(), indices.begin() + pop_total_size, 0);
    
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

    n_users = new int*[n_threads];
    for(int i=0; i< n_threads; i++){
        n_users[i] = new int[n_commodities];
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


    ~Genetic(){
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
            idx = get_rand_idx(0, vect.size() - 1);
            if(idx!=i){
                vect[i] += vect[idx];
                vect[idx] = vect[i] - vect[idx];
                vect[i] -= vect[idx];
            }

        }
    }


    void generate(const std::vector<double> &a_parent, const std::vector<double> &b_parent, std::vector<double> &child, const double* u_bounds, std::vector<int> &element_order, double m_rate, short num_paths){
        int i,iter, idx;
        short r_size = recombination_size;
        int th = omp_get_thread_num();
        for(i=0; i<num_paths; i++) child[i] = a_parent[i];
        // reshuffle_element_order(element_order);

        for(i=0; i < r_size; i++) {
            std::uniform_int_distribution<int> distribution(i, num_paths - 1);
            idx = distribution(generators[th]);
            element_order[i] += element_order[idx];
            element_order[idx] = element_order[i] - element_order[idx];
            element_order[i] -= element_order[idx];
            child[element_order[idx]] = b_parent[element_order[idx]];
        }
        std::uniform_real_distribution<double> distribution_(0., 1.);
        for(i=0; i<num_paths; i++) {
            
            if(distribution_(generators[omp_get_thread_num()]) < m_rate) {
                std::uniform_real_distribution<double> distribution_(0., u_bounds[i]);
                child[i] = distribution_(generators[omp_get_thread_num()]);
                }
        }
        // print_vect(child, n_paths);

    }

    void init_population(double* init_pop){
        short th;
        #pragma omp parallel for num_threads(n_threads) schedule(static) private(th) shared(population, init_pop)
        for(short i=0; i < pop_size; i++){
            th = omp_get_thread_num();
            for(short j=0; j < n_paths; j++) population[i][j] = init_pop[i*n_paths + j];

            vals[i] = eval(population[i], transfer_costs[th], commodities_tax_free[th], n_users[th]);
        }
        for(short i=pop_size; i< pop_total_size; i++) population[i] = std::vector<double> (n_paths);
    }

    void run(double* init_pop, int iterations){
        init_population(init_pop);
        for(int iter= 0; iter<iterations; iter ++) {

            double* lower_bounds = new double[n_paths];
            for(short i=0; i < n_paths; i++) lower_bounds[i] = 0;
        
            short th;
            reshuffle_element_order(random_order);
            #pragma omp parallel for num_threads(n_threads) default(none) private(th) shared(upper_bounds, population, a_combs, b_combs, random_order, random_element_order,n_paths)
            for(short i=0; i < off_size; i++) {
                th = omp_get_thread_num();
                generate(population[indices[a_combs[random_order[i]]]], 
                        population[indices[b_combs[random_order[i]]]], 
                        population[indices[pop_size + i]], 
                        upper_bounds[th], random_element_order[th], mutation_rate, n_paths);
                vals[indices[pop_size + i]] = eval(population[indices[pop_size + i]], transfer_costs[th], commodities_tax_free[th], n_users[th]);
                }
            argsort(vals, pop_size + off_size);
            if(iter%100 == 0) {
            Swarm* swarm = new Swarm{commodities_tax_free[0], n_users[0], t_costs, upper_bounds[0], lower_bounds, n_commodities, n_paths, 128, 1000, 100, 1};
            swarm -> run(population, )
            std::cout<<vals[indices[0]]<<std::endl;
            }
            // print_vector(indices);
            // print_vector(vals);std::cout<<std::endl;
        }
        std::cout<<vals[indices[0]]<<std::endl;
        
    }


    double eval(std::vector<double> &p, double** trans_costs, double* comm_tax_free, int* n_usr){

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


    void argsort(const std::vector<double> &array, int end) {
        std::iota(indices.begin(), indices.begin() + end, 0);
        std::sort(indices.begin(), indices.begin() + end,
                [&array](int left, int right) -> bool {
                    // sort indices according to corresponding array element
                    return array[left] > array[right];
                });
    }

    void print_stuff(){
        std::cout<<"hahaha"<<std::endl;
}

};









