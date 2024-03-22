#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <jsoncpp/json/json.h>
#include <fstream>
#include "genetic_operators.h"


double* get_random(short size, double start, double end) {
    double* vect = new double[size];
    std::default_random_engine generator(std::rand());
    std::uniform_real_distribution<double> distribution(start, end);
    for(int i=0; i < size; i++){
        vect[i] = distribution(generator);
        std::cout<<vect[i]<<std::endl;
    }
    return  vect;
}


short* get_random_int(short size, double start, double end) {
    short* vect = new short[size];
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<int> distribution(start, end);
    for(int i=0; i < size; i++){
        vect[i] = distribution(generator);
        std::cout<<vect[i]<<std::endl;
    }
    return vect;
}

int* make_combs(short pop_size){
    int* combs = new int[pop_size*2];
     for(int i=0; i < pop_size - 1; i++){
        combs[i] = 0;
        combs[pop_size + i] = i;
     }
    return combs;

}
    

int main(){


    short n_paths = 6;
    short n_commodities = 4;
    short pop_size = 4;

    double* upper_bounds = get_random(n_paths, 10, 20);
    int* combs = make_combs(pop_size);
    double* comm_tax_free = get_random(n_commodities, 10, 20);
    short* n_usr = get_random_int(n_commodities, 10, 20);
    double* trans_costs = get_random(n_paths*n_commodities, 2, 6);
    
    short num_threads = 1;
    short recombination_size = 2;
    short off_size = 3;
    short mutation_rate = 0.1;

    double* population = get_random(n_paths*pop_size + n_paths * off_size, 10, 20);
    double* vals = new double[recombination_size];

    Genetic g(upper_bounds, combs, comm_tax_free, n_usr, trans_costs, n_commodities, pop_size, off_size, n_paths, mutation_rate, recombination_size, num_threads);
    g.generation(population, vals);
}