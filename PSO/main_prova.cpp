#include "../PSO/swarm_.cpp"
#include "../PSO/read_file.cpp"
#include <chrono>
#include <io.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>


int main() {

    clock_t start1, end1, start2, end2;
    
    /*--- path test cases directory ---*/
    std::string directory_path = "C:/Users/Isabella/Desktop/BILEVEL/NPP/TestCases/";

    /*--- collect all the files of tipe "loc*" in the test directory ---*/
    std::vector<std::string> files{};
    _finddata_t data;
    int ff = _findfirst((char*)(directory_path+"loc*").c_str(), &data);
    if (ff != -1) {
        int res = 0;
        while (res != -1) {
            files.push_back(data.name);
            res = _findnext(ff, &data);
        }
        _findclose(ff);
    }

    /*--- create and open a new file in order to write there useful outputs ---*/
    std::string file_name = "pso_benchmark_times.txt";
    std::string solution_name = "pso_benchmark_solution.txt";
    std::ofstream outfile;
    std::ofstream outfile_sol;
    outfile.open(file_name);
    outfile_sol.open(solution_name);
  
    /*--- set PSO parameters ---*/
    short n_particles = 1012;
    int n_iterations = 1500;
    short num_th = 1;

    /*--- iterate over all the files collected ---*/
    for (int i=0;i<files.size();++i) {

        /*--- read the current TestCase information ---*/
        FileReader file_reader{directory_path+files[i]};

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
        std::cout<<"distance: "<<s.compute_distance()<<std::endl;
    
        /*--- compute n_loc and n_highways ---*/
        int n_loc = (files[i][5]>=48 && files[i][5]<=57) ? std::stoi(files[i].substr(4,2)):std::stoi(files[i].substr(4,1));
        int n_highway = 1/2+std::sqrt(1/4+2*file_reader.n_tolls);

        /*--- write outputs on file ---*/
        outfile <<"loc_"<<n_loc<<"_comm_"<<file_reader.n_commodities<<"_toll_"<<n_highway<<
        ": time init = "<<((float)end1 - (float)start1)/CLOCKS_PER_SEC<<"s time run = "<<
        ((float)end2 - (float)start2)/CLOCKS_PER_SEC<<"  obj = "<<s.get_best_val()<<std::endl;

        outfile_sol <<"loc_"<<n_loc<<"_comm_"<<file_reader.n_commodities<<"_toll_"<<n_highway<<
        ": solution = "<<"[ ";
        for (int k=0;k<file_reader.n_tolls;k++)
            outfile_sol<< s.get_best()[k]<<", ";
        outfile_sol<<"]"<<std::endl;
    }

    /*--- close file ---*/
    outfile.close();
    outfile_sol.close();
};