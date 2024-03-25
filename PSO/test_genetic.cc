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


short* get_random_int(short size, double start, double end) {
    short* vect = new short[size];
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<int> distribution(start, end);
    for(int i=0; i < size; i++){
        vect[i] = distribution(generator);
        //std::cout<<vect[i]<<std::endl;
    }
    return vect;
}
    

int main(){
    using namespace std::chrono;

    // After function call
    
    // std::srand(std::time(0));
    short n_paths = 180;
    short n_commodities = 180;
    short pop_size = 256;
    short off_size = pop_size/2;

    double* upper_bounds = get_random(n_paths, 10, 20);
    double* comm_tax_free = get_random(n_commodities, 20, 30);
    short* n_usr = get_random_int(n_commodities, 10, 20);
    double* trans_costs = get_random(n_paths*n_commodities, 10, 20);
    
    short num_threads = 8;
    short recombination_size = n_paths/2;
    
    double mutation_rate = 0.02;

    double* population = get_random(n_paths*pop_size + n_paths * off_size, 10, 20);
    double* children = new double[n_paths * off_size];
    int iterations = 1000;

    Genetic gg(upper_bounds, comm_tax_free, n_usr, trans_costs, n_commodities, pop_size, off_size, n_paths, mutation_rate, recombination_size, 1);
    auto start1 = high_resolution_clock::now();
    gg.run(population,iterations);
    auto stop1 = high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    std::cout << "duration th "<<1<<" "<<duration1.count()/1000. << std::endl;

    Genetic g(upper_bounds, comm_tax_free, n_usr, trans_costs, n_commodities, pop_size, off_size, n_paths, mutation_rate, recombination_size, num_threads);
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
