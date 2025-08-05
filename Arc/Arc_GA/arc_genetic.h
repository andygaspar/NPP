#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include "utils_.h"


class ArcGenetic {
    public:
    // problem related features
    short pop_size;
    short n_tolls;
    short off_size;
    short pop_total_size;
    
    short n_commodities;
    double tolerance;
    int adj_size;

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
    double init_commodity_val;

    std::vector<std::vector<std::vector<double>>> adj;
    std::vector<std::vector<int>> n_users;
    std::vector<std::vector<double>> upper_bounds;
    std::vector<std::vector<double>> lower_bounds;
    std::vector<std::vector<int>> origins;
    std::vector<std::vector<int>> destinations;
    std::vector<std::vector<std::vector<int>>> tolls_idxs;


    std::vector<std::vector<std::vector<double>>> adj_solution;
    std::vector<std::vector<std::vector<double>>> prices_mat;
    std::vector<std::vector<double>> dist;
    std::vector<std::vector<bool>> visited;
    std::vector<std::vector<double>> profit;


    double START_VAL = pow(10, 6);

    bool verbose;
    std::vector<std::default_random_engine> generators;

    double best_val;

    int no_improvement;



    ArcGenetic(double* upper_bounds_, double* lower_bounds_, double* adj_, int adj_size_, int* tolls_idxs_, int* n_usr, int* origins_, int* destinations_, short n_commodities_, short n_tolls_, 
    short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
    short num_threads_, bool verbose_, short seed){

    pop_size = pop_size_;
    n_tolls = n_tolls_;
    off_size = off_size_;
  

    pop_total_size = pop_size + off_size;

    no_improvement = 0;
    best_val = 0;
    tolerance = pow(10, -9);

    

    mutation_rate = mutation_rate_;
    recombination_size = recombination_size_;
    n_threads = num_threads_;
    start_index = pop_size * n_tolls;

    adj_size = adj_size_;

    verbose = verbose_;

    std::mt19937 r;
    generators = std::vector<std::default_random_engine> (0);
    for (int i = 0; i <  n_threads; i++) {
        std::default_random_engine engine;
        if(seed>=0) engine.seed(i);
        else engine = std::default_random_engine(r());
        generators.emplace_back(engine);

    }

    population = std::vector<std::vector<double>> (pop_total_size, std::vector<double>(n_tolls, 0));
    vals = std::vector<double> (pop_total_size, 0);

    n_commodities = n_commodities_;
    init_commodity_val = init_commodity_val=pow(10, 5);
    
    indices = std::vector<int> (pop_total_size);
    std::iota(indices.begin(), indices.begin() + pop_total_size, 0);


    adj = std::vector<std::vector<std::vector<double>>> (n_threads, std::vector<std::vector<double>> (adj_size, std::vector<double> (adj_size, 0)));
    adj_solution = std::vector<std::vector<std::vector<double>>> (n_threads, std::vector<std::vector<double>> (adj_size, std::vector<double> (adj_size, 0)));

    for(int th=0; th< n_threads; th++){
        for(int i=0; i< adj_size; i++){
            for(int j=0; j< adj_size; j++) {
                adj[th][i][j]=adj_[i*adj_size + j];
                adj_solution[th][i][j]=adj_[i*adj_size + j];
            }

        }
    }

    tolls_idxs = std::vector<std::vector<std::vector<int>>> (n_threads, std::vector<std::vector<int>> (n_tolls, std::vector<int> (2, 0)));
    for(int th=0; th< n_threads; th++){
        for(int i=0; i< n_tolls; i++){
            tolls_idxs[th][i][0] = tolls_idxs_[i];
            tolls_idxs[th][i][1] = tolls_idxs_[n_tolls + i];
        }
    }
    

    prices_mat = std::vector<std::vector<std::vector<double>>> (n_threads, std::vector<std::vector<double>> (adj_size, std::vector<double> (adj_size, 0)));
    dist = std::vector<std::vector<double>> (n_threads, std::vector<double> (adj_size, 0));
    profit = std::vector<std::vector<double>> (n_threads, std::vector<double> (adj_size, 0));
    visited = std::vector<std::vector<bool>> (n_threads, std::vector<bool> (adj_size, false));

    n_users = std::vector<std::vector<int>> (n_threads, std::vector<int>(n_commodities, 0));
    for(int th=0; th< n_threads; th++){
        for(int j=0; j< n_commodities; j++) n_users[th][j] = n_usr[j];
    }

    origins = std::vector<std::vector<int>> (n_threads, std::vector<int>(n_commodities, 0));
    for(int th=0; th< n_threads; th++){
        for(int j=0; j< n_commodities; j++) origins[th][j] = origins_[j];
    }

    destinations = std::vector<std::vector<int>> (n_threads, std::vector<int>(n_commodities, 0));
    for(int th=0; th< n_threads; th++){
        for(int j=0; j< n_commodities; j++) destinations[th][j] = destinations_[j];
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

    random_element_order = std::vector<std::vector<int>> (n_threads, std::vector<int> (n_tolls));
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_tolls; j++) random_element_order[i][j] = j;
        }

    upper_bounds = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_tolls, 0));;
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_tolls; j++) upper_bounds[i][j] = upper_bounds_[j];
        }


    lower_bounds = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_tolls, 0));;
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_tolls; j++) lower_bounds[i][j] = lower_bounds_[j];
        }
    }

    ~ArcGenetic(){
    }


    double get_best_val() {return best_val;}

    double* get_population () {
        double* out_pop = new double[pop_size*n_tolls];
        for(short i=0; i < pop_size; i++) {
            for(short j=0; j< n_tolls; j++) out_pop[i*n_tolls + j] = population[indices[i]][j];
        }
        return out_pop;
    }

    double* get_vals () {
        double* out_vals = new double[pop_size];
        for(short j=0; j< pop_size; j++) out_vals[j] = vals[indices[j]];
        return out_vals;
    }

    void reshuffle_element_order(std::vector<int> &vect){
        size_t idx, temp;
        std::uniform_int_distribution<size_t> distribution = std::uniform_int_distribution<size_t> (0, vect.size() - 1);
        int th = omp_get_thread_num();
        for(size_t i=0; i<vect.size(); i++) {
            idx = distribution(generators[th]);
            temp = vect[idx];
            vect[idx] = vect[i];
            vect[i] = temp;
        }
    }      
   

    void init_population(double* init_pop){
        short th;
        // #pragma omp parallel for num_threads(n_threads) schedule(static) private(th) shared(population, init_pop)
        for(short i=0; i < pop_size; i++){
            th = omp_get_thread_num();
            for(short j=0; j < n_tolls; j++) population[i][j] = init_pop[i*n_tolls + j];

            vals[i] = eval(population[i], n_users[th], adj[th], adj_solution[th], tolls_idxs[th], prices_mat[th], dist[th], profit[th], 
                                                    visited[th], origins[th], destinations[th], n_commodities, n_tolls, START_VAL, tolerance);
            // std::cout<<vals[i]<<std::endl;
        }
        double maxval = 0;
        for(short i=0; i < pop_size; i++) if(vals[i] > maxval) {maxval = vals[i];}
    }


    void generate(const std::vector<double> &a_parent, const std::vector<double> &b_parent, std::vector<double> &child, 
                    std::vector<double> & u_bounds, std::vector<double> & l_bounds,
                    std::vector<int> &element_order, double m_rate, short num_paths, short recomb_size, std::default_random_engine gen){
            int i,iter, idx;
            short r_size = recomb_size;

            for(i=0; i<num_paths; i++) child[i] = a_parent[i];

            // randomly select component from b_parentn_users[th], adj[th], adj_solution[th], tolls_idxs[th], prices_mat[th], dist[th], profit[th], 
            reshuffle_element_order(element_order);
            for(i=0; i < r_size; i++) child[element_order[i]] = b_parent[element_order[i]];


            int th = omp_get_thread_num();
            std::uniform_real_distribution<double> distribution_;
            std::uniform_real_distribution<double> distribution_mutation(0., 1.);
            for(i=0; i<num_paths; i++) {
                if(u_bounds[i] > 0){
                    if(distribution_mutation(generators[th]) < m_rate) {
                        std::uniform_real_distribution<double> distribution_(l_bounds[i], u_bounds[i]);
                        child[i] = distribution_(generators[th]);
                        }
                }
            }

        }



    void run(double* init_pop, int iterations){
        init_population(init_pop);
        size_t k;
        int j, p;
        double std;
        // print_pop();
        // print_mat_vector(adj[3]);
        double m_rate = mutation_rate;

        for(int iter= 0; iter<iterations; iter ++) {
        
            short th;
            reshuffle_element_order(random_order);
            // print_pop();
            #pragma omp parallel for num_threads(n_threads) private(th) 
            for(short i=0; i < off_size; i++) {
                th = omp_get_thread_num();
                generate(population[indices[a_combs[random_order[i]]]], 
                        population[indices[b_combs[random_order[i]]]], 
                        population[indices[pop_size + i]], 
                        upper_bounds[th], lower_bounds[th], random_element_order[th], mutation_rate, n_tolls, recombination_size, generators[th]);
                vals[indices[pop_size + i]] = eval(population[indices[pop_size + i]], n_users[th], adj[th], adj_solution[th], tolls_idxs[th], prices_mat[th], dist[th], profit[th], 
                                                    visited[th], origins[th], destinations[th], n_commodities, n_tolls, START_VAL, tolerance);
                }
            

            argsort(vals, indices, pop_total_size);
            if(vals[indices[0]] > best_val) {
                best_val = vals[indices[0]];
                no_improvement = 0;
            }

            else no_improvement ++;
            if(no_improvement >= 500) {
                restart_population();
                //std::cout<<"restarted"<<std::endl;                
                no_improvement = 0;
            }

            std = get_std(vals, indices, pop_size);
            if(verbose and  iter%1000 == 0) std::cout<<"iteration "<< iter<<"    "<<vals[indices[0]]<<"   mean " <<get_mean(vals, indices, pop_size)<<"   std " <<std<<"   no impr " <<no_improvement<<std::endl;
            if(std < 0.0000001) {restart_population(); } // std::cout<<"restarted "<<std<<std::endl;}
        }

        // print_vector(population[indices[12]]);
        
        best_val = vals[indices[0]];
        if(verbose) std::cout<<"final iteration "<<best_val<<std::endl;
        // print_pop();
        
    }

    void npp_dijkstra(std::vector<std::vector<double>> &adj_sol, std::vector<std::vector<double>> &price,  std::vector<double>& distance, std::vector<double>& profit_, std::vector<bool>& visited_, 
    const int origin, const int comm_n_users, const double start_max_val, const double tol){

        for (size_t i = 0; i < distance.size(); i++) {
            distance[i] = START_VAL;
            visited_[i] = false;
            profit_[i] = 0;

        }

    
        // Distance of source vertex from itself is always 0
        distance[origin] = 0;
        size_t j; int min_index = 0;
        double min_distance, max_profit;
    
        // Find shortest path for all vertices
        for (size_t count = 0; count < distance.size() - 1; count++) {

            min_distance = start_max_val; 
            max_profit = 0;
            for (j = 0; j < distance.size(); j++)
                if (visited_[j] == false && distance[j] <= min_distance + tol) {
                    if (distance[j] < min_distance - tol) {
                        min_distance = distance[j], 
                        min_index = j;
                        max_profit = profit_[j];  
                    }
                    else{
                        if(profit_[j] > max_profit){
                            min_distance = distance[j], 
                            min_index = j;
                            max_profit = profit_[j];  
                        }
                    }
                    
                }
                    
    
            // Mark the picked vertex as processed
            visited_[min_index] = true;
    
            // Update distance value of the adjacent vertices of the
            // picked vertex.
            for (j = 0; j < distance.size(); j++)

                if (!visited_[j] && adj_sol[min_index][j]
                    && distance[min_index] != START_VAL
                    && distance[min_index] + adj_sol[min_index][j] <= distance[j] + tol) {

                        if (distance[min_index] + adj_sol[min_index][j] < distance[j] - tol) {
                            distance[j] = distance[min_index] + adj_sol[min_index][j]; 
                            profit_[j] = profit_[min_index] + price[min_index][j] * comm_n_users;  
                        }
                        else{
                            if(profit_[min_index] + price[min_index][j] * comm_n_users > profit_[j]){
                                distance[j] = distance[min_index] + adj_sol[min_index][j]; 
                                profit_[j] = profit_[min_index] + price[min_index][j] * comm_n_users;  
                            }
                        }
                            
                    }
                    
        }
    
    }

    double eval(const std::vector<double> &p, const std::vector<int>& n_usr, 
                const std::vector<std::vector<double>> &adj_,
                std::vector<std::vector<double>> &adj_sol, 
                const std::vector<std::vector<int>> &t_idxs,
                std::vector<std::vector<double>> &price,  
                std::vector<double>& distance, 
                std::vector<double>& profit_, 
                std::vector<bool>& visited_, 
                const std::vector<int>& origins_,
                const std::vector<int>& destinations_,
                const int n_comm, 
                const int n_t,
                const double start_max_val, const double tol){

        /* compute objective value */
        double current_run_val=0;
        double path_price;
        int i, j;
        double minimal_cost;


        for(i=0; i<n_t; i++){
            adj_sol[t_idxs[i][0]][t_idxs[i][1]] = adj_[t_idxs[i][0]][t_idxs[i][1]] + p[i];
            price[t_idxs[i][0]][t_idxs[i][1]] = p[i];
        }

        for(i=0; i<n_comm; i++) {
            npp_dijkstra(adj_sol, price, distance, profit_, visited_, origins_[i], n_usr[i], start_max_val, tol);
            current_run_val += profit_[destinations_[i]];
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
        short th;
        #pragma omp parallel for num_threads(n_threads)  private(th) 
        for(int i=1; i < pop_size; i++){
            std::uniform_real_distribution<double> distribution;       
            th = omp_get_thread_num();
            for(int j=0; j < n_tolls; j++) {
                if(upper_bounds[th][j] >0) {
                    distribution = std::uniform_real_distribution<double> (lower_bounds[th][j], upper_bounds[th][j]);
                    population[indices[i]][j] = distribution(generators[th]);
                }

            }
            vals[indices[i]] = eval(population[indices[i]], n_users[th], adj[th], adj_solution[th], tolls_idxs[th], prices_mat[th], dist[th], profit[th], 
                                                    visited[th], origins[th], destinations[th], n_commodities, n_tolls, START_VAL, tolerance);

        }
    }


    void print_pop(){
        for(int p=0; p<pop_total_size; p++){
            for(int j =0; j<n_tolls; j++)std::cout<<population[indices[p]][j]<<" ";
            std::cout<<"     "<<vals[indices[p]]<<std::endl;
        }
        std::cout<<std::endl;
    }

};


   









