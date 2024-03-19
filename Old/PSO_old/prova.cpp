//#include "swarm_.cpp"
#include "run_utils_.cpp"
//#include "read_file.cpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>


int main() {

    clock_t start1, end1, start2, end2;
    
    /*--- path test cases directory ---*/
    std::string directory_path = "../TestCases/";

    /*--- collect all the files of tipe "loc*" in the test directory ---*/
    std::vector<std::string> files{};

    files.push_back("loc_50_comm_50_toll_10");
  
    /*--- set PSO parameters ---*/
    short n_particles = 15000;
    short N_PARTICLES = 128;
    int n_iterations = 10000;
    int n_first_iterations=100;
    int N_div=3;
    int n_cut=5;
    int N=10;
    short num_th = 1;
    double tmp_delta;
    int no_update_lim=2000;
    double restriction_rate=0.6;

    /*--- iterate over all the files collected ---*/
    for (int i=0;i<files.size();++i) {

        /*--- read the current TestCase information ---*/
        FileReader file_reader{directory_path+files[i]};
        std::vector<double> zeros_(file_reader.n_tolls,1.); 
        
        std::vector<double> final_solution = recurrent_run_PSO_on_smaller_domain(file_reader, n_particles, N_PARTICLES, n_iterations, n_first_iterations,N_div,n_cut,N,restriction_rate,no_update_lim);

    }
};