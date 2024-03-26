#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <jsoncpp/json/json.h>
#include <fstream>
#include "genetic.h"


double* get_random(int size, double start, double end) {
    double* vect = new double[size];
    std::random_device r;
    std::default_random_engine generator(std::rand());
    std::uniform_real_distribution<double> distribution(start, end);
    for(int i=0; i < size; i++){
        vect[i] = distribution(generator);
        //std::cout<<vect[i]<<std::endl;
    }
    return  vect;
}


double* get_random_population(double* upper_bounds, int pop_size, int n_paths) {
    double* vect = new double[pop_size*n_paths];
    std::random_device r;
    std::default_random_engine generator(std::rand());
    
    for(int i=0; i < pop_size; i++){
        for(int j=0; j < n_paths; j++) {
            std::uniform_real_distribution<double> distribution(0., upper_bounds[j]);
            vect[i*n_paths + j] = distribution(generator);
        }

        //std::cout<<vect[i]<<std::endl;
    }
    return  vect;
}


int* get_random_int(short size, double start, double end) {
    int* vect = new int[size];
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<int> distribution(start, end);
    for(int i=0; i < size; i++){
        vect[i] = distribution(generator);
        //std::cout<<vect[i]<<std::endl;
    }
    return vect;
}

double* get_upper_bounds(double* comm_tax_free, double* trans_costs, short n_commodities, short n_paths){
    double* upper_bounds = new double[n_paths];
    for(int p=0; p < n_paths; p++){
        double bound = 0;
        for(int c=0; c < n_commodities; c++){
            if(comm_tax_free[c] + trans_costs[p*n_paths + c] > bound) bound = comm_tax_free[c] + trans_costs[p*n_paths + c];
        }
        upper_bounds[p] = bound;
    }
    return upper_bounds;
}
    

int main(){
    using namespace std::chrono;

    // After function call
    
    // std::srand(std::time(0));
    short n_paths = 10;
    short n_commodities = 10;
    short pop_size = 128;
    short off_size = pop_size/2;
    int iterations = 500;
    short recombination_size = n_paths/2;
    double mutation_rate = 0.5;


    short pso_every = 50;
    short pso_size = pop_size/4;
    short pso_selection = pso_size/2;
    short pso_iterations = 1000;
    short pso_no_update = 1000;

    short num_threads = 12;



    
    double* comm_tax_free = get_random(n_commodities, 20, 30);
    int* n_usr = get_random_int(n_commodities, 1, 10);
    double* trans_costs = get_random(n_paths*n_commodities, 5, 15);
    double* upper_bounds = get_upper_bounds(comm_tax_free, trans_costs, n_commodities, n_paths);
    

    double* population = get_random_population(upper_bounds, pop_size, n_paths);
    double* children = new double[n_paths * off_size];


    // Genetic gg(upper_bounds, comm_tax_free, n_usr, trans_costs, n_commodities, n_paths, 
    //             pop_size, off_size,  mutation_rate, recombination_size, 
    //             pso_size, pso_selection, pso_every, pso_iterations, pso_no_update, 1);
    // auto start1 = high_resolution_clock::now();
    // gg.run(population,iterations);
    // auto stop1 = high_resolution_clock::now();
    // auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    // std::cout << "duration th "<<1<<" "<<duration1.count()/1000. << std::endl;

    population = get_random(n_paths*pop_size + n_paths * off_size, 10, 20);
    Genetic g(upper_bounds, comm_tax_free, n_usr, trans_costs, n_commodities, n_paths,
                pop_size, off_size, mutation_rate, recombination_size, 
                pso_size, pso_selection, pso_every,pso_iterations, pso_no_update, num_threads);
    auto start = high_resolution_clock::now();
    g.run(population,iterations);
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "duration th "<<num_threads<<" "<<duration.count()/1000. << std::endl;




}
//     double * rand_vect = new double[4];
// for(int k=0; k < 10; k++) {
//     #pragma omp parallel for num_threads(4) schedule(static) 
//     for(int i=0; i < 4; i++) {
//         rand_vect[i] = get_rand(0., 1.);
//     }

//     print_vect(rand_vect, 4);
// }
