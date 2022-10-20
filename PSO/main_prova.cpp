#include "swarm.cpp"
#include "read_file.cpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

using string=std::string;

int main() {
    /*
    
    clock_t start, end;
    Swarm s{10,2, &obj};
    start=clock();
    s.update(100);
    end=clock();
    std::cout<<s<<std::endl<<std::endl;
    std::cout<<"time: "<<((float) end - start)/CLOCKS_PER_SEC<<" s"<<std::endl;
    */

    string file_name = "../TestCases/comm_20_tolls_10";
    FileReader file_reader{file_name};
    std::cout<<std::endl<<std::endl;
    short n_particles = 512;
    int n_iterations = 10000;
    short num_th = 16;

    Swarm s{file_reader.commodities_tax_free, file_reader.n_users,file_reader.transfer_costs, file_reader.upper_bounds, 
                  file_reader.n_commodities, file_reader.n_tolls, n_particles, n_iterations, num_th};
    //s.print_particles();  
    s.run();
};