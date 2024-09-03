#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <jsoncpp/json/json.h>
#include <fstream>
#include "arc_genetic_h.h"
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
T* readMatrixFromFile(const std::string& filename, int size) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }

    // Read the number of rows and columns from the first line of the file

    // Allocate memory for the matrix
    T* matrix  = new T[size*size];


    // Read the matrix elements from the file
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file >> matrix[i*size + j];
        }
    }

    return matrix;
}

template <typename T>
void printMatrix(T* matrix, int size) {
    std::cout << "Matrix elements:" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i*size + j] << " ";
        }
        std::cout << std::endl;
    }
}

    

int main(){
    using namespace std::chrono;
    // FileReader fr{"TestCases/comm_10_tolls_10"};
    // After function call
    
    // std::srand(std::time(0));
    


    // static double adj_[] = {0.0,26.455680991172585,0.0,21.464405117819744,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,26.455680991172585,0.0,21.346495489906907,0.0,23.082901282149315,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,21.346495489906907,0.0,0.0,0.0,8.854821990083572,0.0,0.0,0.0,0.0,0.0,0.0,21.464405117819744,0.0,0.0,0.0,18.127616337880774,0.0,24.376823391999682,0.0,0.0,0.0,0.0,0.0,0.0,23.082901282149315,0.0,18.127616337880774,0.0,33.90988281503088,0.0,31.753190023462395,0.0,0.0,0.0,0.0,0.0,0.0,8.854821990083572,0.0,33.90988281503088,0.0,0.0,0.0,8.251622782386665,0.0,0.0,0.0,0.0,0.0,0.0,24.376823391999682,0.0,0.0,0.0,20.866847592587135,0.0,28.75175114247994,0.0,0.0,0.0,0.0,0.0,0.0,31.753190023462395,0.0,20.866847592587135,0.0,32.76789914877983,0.0,22.04133683281797,0.0,0.0,0.0,0.0,0.0,0.0,8.251622782386665,0.0,32.76789914877983,0.0,0.0,0.0,7.1310817459366085,0.0,0.0,0.0,0.0,0.0,0.0,28.75175114247994,0.0,0.0,0.0,7.613878991046221,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,22.04133683281797,0.0,7.613878991046221,0.0,5.606551923209771,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,7.1310817459366085,0.0,5.606551923209771,0.0};
    // int adj_size = 12;
    // double upper_bounds_[] = {90.17934920488645,90.17934920488645,69.48445759700354,69.48445759700354};
    // int toll_idxs_[]= {5,8,2,5,8,5,5,2};
    // int origins_[]= {5,5,2,1,4,0,9,3,1,2,1,3,0,5,0,6,2,0,7,4,0,7,6,1,3,0,8,2,3,0,1,0,3,4,7,3,5,4,1,1,1,0,6,1,2,4,8,4,0,6};
    // int destinations_[]= {8,11,11,10,8,2,10,9,2,8,4,8,1,7,8,10,5,7,11,11,4,10,9,9,4,3,9,4,11,11,3,10,6,10,8,10,10,6,7,5,11,5,11,6,3,7,10,5,6,7};
    // int n_users_[]= {1,4,4,3,1,2,4,4,1,4,3,3,1,2,3,1,4,2,3,1,2,2,3,3,4,3,2,2,4,3,4,1,4,4,4,2,2,2,2,1,3,2,1,1,3,2,4,4,3,3};

    short n_commodities = read_size("Problems/test1/n_com.csv");;
    short n_tolls = read_size("Problems/test1/n_tolls.csv");;
    int adj_size = read_size("Problems/test1/adj_size.csv");
    double*  adj = readMatrixFromFile<double> ("Problems/test1/adj.csv", adj_size);
    double* upper_bounds = readMatrixFromFile<double> ("Problems/test1/ub.csv", n_tolls);
    int* toll_idxs= readMatrixFromFile<int> ("Problems/test1/toll_idxs.csv", 2*n_tolls);
    int* origins = readMatrixFromFile<int> ("Problems/test1/origins.csv", n_commodities);
    int* destinations = readMatrixFromFile<int> ("Problems/test1/destinations.csv", n_commodities);
    int* n_users = readMatrixFromFile<int> ("Problems/test1/n_users.csv", n_commodities);



    // int*  toll_idxs = new int[n_tolls*2];
    // for(int i=0; i < n_tolls*2; i++){
    //     toll_idxs[i] = toll_idxs_[i];
    // }


    // int*  origins = new int[n_commodities];
    // for(int i=0; i < n_commodities; i++){
    //     origins[i] = origins_[i];
    // }


    // int*  destinations = new int[n_commodities];
    // for(int i=0; i < n_commodities; i++){
    //     destinations[i] = destinations_[i];
    // }

    // int*  n_users = new int[n_commodities];
    // for(int i=0; i < n_commodities; i++){
    //     n_users[i] = n_users_[i];
    // }


    short pop_size = 64;
    short off_size = pop_size/2;
    int iterations = 100;
    short recombination_size = n_tolls/2;
    double mutation_rate = 0.02;


    short num_threads = 16;

    bool verbose = true;

    short seed = 1; 




    double* lower_bounds = new double[n_tolls];
    for(int p=0; p < n_tolls; p++) lower_bounds[p] = 0;


 
    //fr.print_problem();

    double* population = get_random_population(upper_bounds, pop_size, n_tolls);


    // ArcGenetic g(upper_bounds, lower_bounds, comm_tax_free, n_usr, trans_costs, n_commodities, n_paths,
    //             pop_size, off_size, mutation_rate, n_paths/2, 
    //             100, num_threads, verbose, seed);


    ArcGeneticHeuristic g(upper_bounds, lower_bounds, adj, adj_size, toll_idxs, n_users, 
                origins, destinations, n_commodities, n_tolls, 
                pop_size, off_size, mutation_rate, recombination_size, 
                num_threads, verbose, seed);

    // double* population = get_random_population(fr.upper_bounds, pop_size, n_paths);
    // double* children = new double[n_paths * off_size];

    // double* new_population = get_random(n_paths*pop_size + n_paths * off_size, 10, 20);
    // Genetic g(fr.upper_bounds, fr.commodities_tax_free, fr.n_users, fr.transfer_costs, fr.n_commodities, fr.n_paths,
    //             pop_size, off_size, mutation_rate, fr.n_paths/2, 
    //             pso_size, pso_selection, pso_every,pso_iterations, pso_final_iterations, pso_no_update, num_threads, verbose, seed);
    auto start = high_resolution_clock::now();
    g.run(population,iterations);
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "duration th "<<num_threads<<" "<<duration.count()/1000. << std::endl;






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
