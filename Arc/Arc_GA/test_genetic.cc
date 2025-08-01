#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <jsoncpp/json/json.h>
#include <fstream>
#include "arc_genetic_h2.h"
#include "arc_genetic.h"
// #include "instance.h"


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

int read_size(const std::string& filename) { 
        std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }
    int size;
    file >> size;
    return size;
}



template <typename T>
T* readMatrixFromFile(const std::string& filename, int row_size, int col_size=1) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }

    // Read the number of rows and columns from the first line of the file

    // Allocate memory for the matrix
    T* matrix  = new T[row_size*col_size];


    // Read the matrix elements from the file
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j) {
            file >> matrix[i*col_size + j];
        }
    }

    return matrix;
}

template <typename T>
void printMatrix(T* matrix, int row_size, int col_size) {
    std::cout << "Matrix elements:" << std::endl;
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j) {
            std::cout << matrix[i*col_size + j] << " ";
        }
        std::cout << std::endl;
    }
}

    

int main(){
    using namespace std::chrono;


    std::string file_name = "test_dijkstra";
    short n_commodities = read_size("Problems/" + file_name + "/n_com.csv");;
    short n_tolls = read_size("Problems/" + file_name + "/n_tolls.csv");;
    int adj_size = read_size("Problems/" + file_name + "/adj_size.csv");
    double*  adj = readMatrixFromFile<double> ("Problems/" + file_name + "/adj.csv", adj_size, adj_size);
    double* upper_bounds = readMatrixFromFile<double> ("Problems/" + file_name + "/ub.csv", n_tolls);
    int* toll_idxs= readMatrixFromFile<int> ("Problems/" + file_name + "/toll_idxs.csv", 2*n_tolls);
    int* origins = readMatrixFromFile<int> ("Problems/" + file_name + "/origins.csv", n_commodities);
    int* destinations = readMatrixFromFile<int> ("Problems/" + file_name + "/destinations.csv", n_commodities);
    int* n_users = readMatrixFromFile<int> ("Problems/" + file_name + "/n_users.csv", n_commodities);
    // for(short i=0; i < n_tolls; i++) upper_bounds[i] *= 2;
    printMatrix(upper_bounds, 1, n_tolls);

    short pop_size = 5;
    short off_size = pop_size/2;
    int iterations = 1000;
    int heuristic_every = 2000;
    short recombination_size = n_tolls/2;
    double mutation_rate = 0.2;


    short num_threads = 1;

    bool verbose = true;

    short seed = -1; 




    double* lower_bounds = new double[n_tolls];
    for(int p=0; p < n_tolls; p++) lower_bounds[p] = 0;


 
    //fr.print_problem();

    // double* population = get_random_population(upper_bounds, pop_size, n_tolls);
    double* population = readMatrixFromFile<double> ("Problems/" + file_name + "/population.csv", pop_size, n_tolls);


    // ArcGenetic g(upper_bounds, lower_bounds, comm_tax_free, n_usr, trans_costs, n_commodities, n_paths,
    //             pop_size, off_size, mutation_rate, n_paths/2, 
    //             100, num_threads, verbose, seed);
    
    // ArcGenetic gg(upper_bounds, lower_bounds, adj, adj_size, toll_idxs, n_users, origins, destinations, n_commodities, n_tolls, 
    //                 pop_size, off_size, mutation_rate, recombination_size, 
    //                 num_threads, verbose, seed);

    // gg.run(population,iterations);


    ArcGeneticHeuristic g(upper_bounds, lower_bounds, adj, adj_size, toll_idxs, n_users, 
                origins, destinations, n_commodities, n_tolls, 
                pop_size, off_size, mutation_rate, recombination_size, heuristic_every,
                num_threads, verbose, seed);
    
    
    // double* population = get_random_population(fr.upper_bounds, pop_size, n_paths);
    // double* children = new double[n_paths * off_size];

    // double* new_population = get_random(n_paths*pop_size + n_paths * off_size, 10, 20);
    // Genetic g(fr.upper_bounds, fr.commodities_tax_free, fr.n_users, fr.transfer_costs, fr.n_commodities, fr.n_paths,
    //             pop_size, off_size, mutation_rate, fr.n_paths/2, 
    //             pso_size, pso_selection, pso_every,pso_iterations, pso_final_iterations, pso_no_update, num_threads, verbose, seed);
    // auto start = high_resolution_clock::now();
    g.run(population,iterations);
    // auto stop = high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // std::cout << "duration th "<<num_threads<<" "<<duration.count()/1000. << std::endl;






// delete[] comm_tax_free;
// delete[] n_usr;
// delete[] trans_costs;;
// delete[] upper_bounds; n_paths;
    

delete[] population;



}
//     double * rand_vect = new double[4];
// for(k=0; k < 10; k++) {
//     #pragma omp parallel for num_threads(4) schedule(static) 
//     for(int i=0; i < 4; i++) {
//         rand_vect[i] = get_rand(0., 1.);
//     }

//     print_vect(rand_vect, 4);
// }
