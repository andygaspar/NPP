#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>

#include "genetic_operators.h"
#include "read_file.h"


class Genetic {
    public:
    // problem related features
    short pop_size;
    short n_paths;
    short off_size;
    short pop_total_size;
    
    short n_commodities;

    std::vector<std::vector<double>> population;
    std::vector<double> vals;
  
    std::vector<int> a_combs;
    std::vector<int> b_combs;
    double mutation_rate;
    short recombination_size;
    std::vector<int> random_order;
    std::vector<std::vector<int>> random_element_order;


    Swarm2 swarm;     
    std::vector<std::vector<double>> pso_population;
    std::vector<int> pso_selection_idx;
    std::vector<int> pso_selection_order;
    std::vector<int> pso_selection_random_order;

    short n_threads;
    std::vector<int> indices;
    short start_index;
    

    double current_run_val;
    double init_commodity_val;

    std::vector<std::vector<double>> commodities_tax_free;
    std::vector<std::vector<std::vector<double>>> transfer_costs;
    std::vector<std::vector<int>> n_users;
    std::vector<std::vector<double>> upper_bounds;

    short pso_size;
    short pso_every;
    short pso_selection;
    short pso_iterations;
    short pso_final_iterations;
    short pso_no_update_lim;

    bool verbose;
    bool provisional_mutation_rate;

    std::vector<std::default_random_engine> generators;

    double best_val;



    Genetic(double* upper_bounds_, double* comm_tax_free, int* n_usr, double* trans_costs, short n_commodities_, short n_paths_, 
    short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
    short pso_size_, short pso_selection_, short pso_every_, short pso_iterations_, short pso_final_iterations_, short pso_no_update_lim_,
    short num_threads_, bool verbose_, short seed){

    pop_size = pop_size_;
    n_paths = n_paths_;
    off_size = off_size_;
    pso_size = pso_size_;
    pso_selection = pso_selection_;
    pso_every = pso_every_;
    pso_no_update_lim = pso_no_update_lim_;
    pso_iterations = pso_iterations_;
    pso_final_iterations =  pso_final_iterations_;

    pop_total_size = pop_size + off_size + pso_selection;

    provisional_mutation_rate = false;


    

    mutation_rate = mutation_rate_;
    recombination_size = recombination_size_;
    n_threads = num_threads_;
    start_index = pop_size * n_paths;

    verbose = verbose_;

    std::random_device r;
    generators = std::vector<std::default_random_engine> (0);
    for (int i = 0; i <  n_threads; i++) {
        std::default_random_engine engine;
        if(seed>=0) engine.seed(i);
        else engine = std::default_random_engine(r());
        generators.emplace_back(engine);

    }

    population = std::vector<std::vector<double>> (pop_total_size, std::vector<double>(n_paths, 0));
    vals = std::vector<double> (pop_total_size, 0);

    n_commodities = n_commodities_;
    init_commodity_val = init_commodity_val=pow(10, 5);
    
    indices = std::vector<int> (pop_total_size);
    std::iota(indices.begin(), indices.begin() + pop_total_size, 0);
    
    commodities_tax_free = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_commodities, 0));;
    for(int i=0; i< n_threads; i++){
        for(int j=0; j< n_commodities; j++) commodities_tax_free[i][j] = comm_tax_free[j];
        }


    transfer_costs = std::vector<std::vector<std::vector<double>>> 
                    (n_threads, std::vector<std::vector<double>>(n_commodities, std::vector<double> (n_paths, 0)));
    for(int i=0; i< n_threads; i++){
        for(int j =0; j<n_commodities; j++)  {
            for(int k=0; k< n_paths; k++) transfer_costs[i][j][k]=trans_costs[j*n_paths + k];
            }
        }

    n_users = std::vector<std::vector<int>> (n_threads, std::vector<int>(n_commodities, 0));
    for(int i=0; i< n_threads; i++){
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

    random_order = std::vector<int> (a_combs.size());
    for(size_t i=0; i<a_combs.size(); i++) random_order[i] = i;

    pso_selection_idx = std::vector<int> (pop_size);
    for(int i=0; i<pop_size; i++) pso_selection_idx[i] = i;

    pso_selection_order = std::vector<int> (pso_size);
    for(int i=0; i<pso_size; i++) pso_selection_order[i] = i;

    pso_selection_random_order = std::vector<int> (pso_size);
    for(int i=0; i<pso_size; i++) pso_selection_random_order[i] = i;

    random_element_order = std::vector<std::vector<int>> (n_threads, std::vector<int> (n_paths));
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_paths; j++) random_element_order[i][j] = j;
        }

    upper_bounds = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_paths, 0));;
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_paths; j++) upper_bounds[i][j] = upper_bounds_[j];
        }

    pso_population = std::vector<std::vector<double>> (pso_size, std::vector<double>(n_paths, 0));
    
    double* lower_bounds = new double[n_paths];
    for(short i=0; i < n_paths; i++) lower_bounds[i] = 0;

    short n_particles = pop_size;
    swarm = Swarm2{comm_tax_free, n_usr, trans_costs, upper_bounds_, lower_bounds, n_commodities, n_paths, n_particles, pso_iterations, pso_no_update_lim, n_threads, seed};
    delete[] lower_bounds;
    }


    ~Genetic(){
    }


    double get_best_val() {return best_val;}

    void reshuffle_element_order(std::vector<int> &vect){
        size_t idx;
        std::uniform_int_distribution<size_t> distribution;
        for(size_t i=0; i<vect.size(); i++) {
            distribution = std::uniform_int_distribution<size_t> (i, vect.size() - 1);
            idx = distribution(generators[0]);
            vect[i] += vect[idx];
            vect[idx] = vect[i] - vect[idx];
            vect[i] -= vect[idx];
        }
    }

    void fill_random_velocity_vect(double* v, int size, double start, double end) {
        std::uniform_real_distribution<double> distribution(start, end);
        for(int i=0; i < size; i++){
            v[i] = distribution(generators[0]);
        }
    }       
   

    void init_population(double* init_pop){
        short th;
        #pragma omp parallel for num_threads(n_threads) schedule(static) private(th) shared(population, init_pop)
        for(short i=0; i < pop_size; i++){
            th = omp_get_thread_num();
            for(short j=0; j < n_paths; j++) population[i][j] = init_pop[i*n_paths + j];

            vals[i] = eval(population[i], transfer_costs[th], commodities_tax_free[th], n_users[th], n_commodities, init_commodity_val, n_paths);
        }
        for(short i=pop_size; i< pop_total_size; i++) population[i] = std::vector<double> (n_paths);

        double maxval = 0;
        for(short i=0; i < pop_size; i++) if(vals[i] > maxval) {maxval = vals[i];}
    }


    void generate(const std::vector<double> &a_parent, const std::vector<double> &b_parent, std::vector<double> &child, std::vector<double> & u_bounds, 
                    std::vector<int> &element_order, double m_rate, short num_paths, short recomb_size, std::default_random_engine gen){
            int i,iter, idx;
            short r_size = recomb_size;
            // int th = omp_get_thread_num();
            for(i=0; i<num_paths; i++) child[i] = a_parent[i];
            std::uniform_int_distribution<int> distribution;
            reshuffle_element_order(element_order);
            // randomly select component from b_parent
            for(i=0; i < r_size; i++) {
                // distribution = std::uniform_int_distribution<int> (i, num_paths - 1);
                // idx = distribution(gen);
                // idx = get_rand_idx(i, num_paths - 1);
                // element_order[i] += element_order[idx];
                // element_order[idx] = element_order[i] - element_order[idx];
                // element_order[i] -= element_order[idx];
                // child[element_order[idx]] = b_parent[element_order[idx]];
                child[element_order[i]] = b_parent[element_order[i]];

            }
            std::uniform_real_distribution<double> distribution_;
            std::uniform_real_distribution<double> distribution_mutation(0., 1.);
            for(i=0; i<num_paths; i++) {
                
                if(distribution_mutation(gen) < m_rate) {
                    std::uniform_real_distribution<double> distribution_(0., u_bounds[i]);
                    child[i] = distribution_(gen);
                    }
            }
            // print_vect(child, n_paths);

        }



    void run(double* init_pop, int iterations){
        init_population(init_pop);

        size_t k;
        int j, p;
        double std;
        
        double* init_vel = new double[pso_size * n_paths];
        double m_rate = mutation_rate;

        for(int iter= 0; iter<iterations; iter ++) {
        
            short th;
            reshuffle_element_order(random_order);
            // print_pop();
            #pragma omp parallel for num_threads(n_threads) default(none) private(th) shared(upper_bounds, population, a_combs, b_combs, random_order, random_element_order, n_commodities, init_commodity_val, n_paths, recombination_size, generators)
            for(short i=0; i < off_size; i++) {
                th = omp_get_thread_num();
                generate(population[indices[a_combs[random_order[i]]]], 
                        population[indices[b_combs[random_order[i]]]], 
                        population[indices[pop_size + i]], 
                        upper_bounds[th], random_element_order[th], mutation_rate, n_paths, recombination_size, generators[th]);
                vals[indices[pop_size + i]] = eval(population[indices[pop_size + i]], transfer_costs[th], commodities_tax_free[th], n_users[th], n_commodities, init_commodity_val, n_paths);
                }
            

            if(iter % pso_every == 0 and iter > 0) {

                argsort(vals, indices, pop_total_size);

                // print_vector(pso_selection_idx);
                for(k=0; k< pso_population.size(); k ++){
                    for(j=0; j<n_paths; j++) pso_population[k][j] = population[indices[pso_selection_idx[k]]][j]; 
                }

                fill_random_velocity_vect(init_vel, pso_size* n_paths, -5., 5.);
                swarm.run(pso_population, init_vel, pso_iterations, false);

                
                argsort(swarm.particles_best, pso_selection_order, pso_size);
                for(p=0; p< pso_selection; p ++) {
                    // print_vector_and_val(swarm.particles[pso_selection_order[p]].personal_best, swarm.particles[pso_selection_order[p]].personal_best_val);
                    
                    population[indices[pop_size + off_size + p]] = swarm.particles[pso_selection_order[p]].personal_best;
                    vals[indices[pop_size + off_size + p]] = swarm.particles[pso_selection_order[p]].personal_best_val;
                }
                // std::cout<<std::endl;
                reshuffle_element_order(pso_selection_random_order);
                // for(p=0; p< pso_selection; p ++) {
                //     print_vector_and_val(swarm.particles[pso_selection_random_order[p]].personal_best, swarm.particles[pso_selection_random_order[p]].personal_best_val);
                // }
                // std::cout<<std::endl;
                // std::cout<<std::endl;

                
            }
            argsort(vals, indices, pop_total_size);
            // if(iter%100 == 0 and iter > 0) print_pop();
            // for(k=0; k< indices.size(); k ++) std::cout<<vals[indices[k]]<<" ";
            // std::cout<<std::endl;
            
            //if(iter%100 == 0 and iter > 0) std::cout<<"iteration "<< iter<<"    "<<vals[indices[0]]<<std::endl;
            std = get_var(vals);
            if(verbose and  iter%250 == 0 and iter > 0) std::cout<<"iteration "<< iter<<"    "<<vals[indices[0]]<<"   mean " <<get_mean(vals)<<"   std " <<std<<std::endl;
            if(std < 1) {restart_population(); std::cout<<"restarted"<<std::endl;}   
        }
        
        std::vector<std::vector<double>> final_run_population = std::vector<std::vector<double>> (pop_size, std::vector<double> (n_paths)); 
        for(p=0; p< pop_size; p ++) final_run_population[p] = population[indices[p]];
        double* final_init_vel = new double[pop_size*n_paths];
        fill_random_velocity_vect(final_init_vel, pop_size*n_paths, -5., 5.);
        swarm.run(final_run_population, final_init_vel, pso_final_iterations, verbose);
        if(verbose) std::cout<<"final iteration "<<swarm.best_val<<std::endl;
        
        best_val = swarm.best_val;

        delete [] init_vel;
        delete [] final_init_vel;
        
    }

    double eval(const std::vector<double> &p,const std::vector<std::vector<double>> & trans_costs, const std::vector<double> &comm_tax_free, const std::vector<int> n_usr, const short n_comm, const double init_comm_val, const short n_paths){

        /* compute objective value */
        double current_run_val=0;
        double toll_cost;
        int i,j,cheapest_path_idx;
        cheapest_path_idx = -1;
        double comm_cost;

        for(i=0; i<n_comm; i++) {
            comm_cost=init_comm_val;
            bool found = false;
            for(j=0; j< n_paths; j++) {

                toll_cost = p[j] + trans_costs[i][j];
                //std::cout<<"commodities "<<p[j]<< " " <<trans_costs[i][j]<<std::endl;
                if(toll_cost <= comm_cost) {
                    if (toll_cost < comm_cost) {
                        comm_cost = toll_cost;
                        cheapest_path_idx = j;
                    }
                    else {
                        if ( p[j] > p[cheapest_path_idx]) {
                            comm_cost = toll_cost;
                            cheapest_path_idx = j;
                        }
                    }
                }
            }

            if(comm_tax_free[i] >= comm_cost) {
                found = true;
                current_run_val += p[cheapest_path_idx]*n_usr[i];
            }
        }
        


        return current_run_val;
    }



    void argsort(const std::vector<double> &array, std::vector<int> &indeces, int end) {
        std::iota(indices.begin(), indices.begin() + end, 0);
        std::sort(indices.begin(), indices.begin() + end,
                [&array](int left, int right) -> bool {
                    // sort indices according to corresponding array element
                    return array[left] > array[right];
                });
    }


    void restart_population() { 
        std::uniform_real_distribution<double> distribution;       
        for(int i=1; i < pop_size; i++){
            for(int j=0; j < n_paths; j++) {
                distribution = std::uniform_real_distribution<double> (0., upper_bounds[0][j]);
                population[indices[i]][j] = distribution(generators[0]);
            }
        }
    }


    void print_pop(){
        for(int p=0; p<pop_total_size; p++){
            for(int j =0; j<n_paths; j++)std::cout<<population[indices[p]][j]<<" ";
            std::cout<<"     "<<vals[indices[p]]<<std::endl;
        }
        std::cout<<std::endl;
    }

};


   









