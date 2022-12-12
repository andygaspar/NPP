#include "../PSO/swarm_.cpp"
#include "../PSO/read_file.cpp"

#include <filesystem>
#include <chrono>
#include <unistd.h>
#include <inttypes.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>


namespace fs = std::filesystem;
const fs::path pathToShow{"../TestCases"};


int main() {

    clock_t start1, end1, start2, end2;

    /*--- create and open a new file in order to write there useful outputs ---*/
    std::string file_name = "pso_benchmark_times.txt";
    std::ofstream outfile;
    outfile.open(file_name);

     /*--- set PSO parameters ---*/
    short n_particles = 128;
    int n_iterations = 5000;
    short num_th = 16;
    
    /*--- path test cases directory ---*/
    std::string directory_path = "../TestCases/";
    std::size_t instance_counter = 0;

    for (const auto& entry : fs::directory_iterator(pathToShow)) {
        std::cout << "Solving instance " << instance_counter << " : " << entry << std::endl;
        /*--- read the current TestCase information ---*/
        FileReader file_reader{entry.path()};

        /*--- initialize the problem ---*/
        start1=clock();
        Swarm s{file_reader.commodities_tax_free,
            file_reader.n_users,file_reader.transfer_costs, 
            file_reader.upper_bounds, 
            file_reader.n_commodities, 
            file_reader.n_tolls, 
            n_particles, 
            n_iterations, 
            num_th}; 
        end1=clock();

        /*--- run ---*/
        start2=clock();
        s.run();
        end2=clock();

        /*--- write outputs on file ---*/
        outfile << entry << ": time init = "<<((float)end1 - (float)start1)/CLOCKS_PER_SEC<<"s time run = "<<((float)end2 - (float)start2)/CLOCKS_PER_SEC<<"  obj = "<<s.get_best_val()<<std::endl;
    }

    /*--- close file ---*/
    outfile.close();
}