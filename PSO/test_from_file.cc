#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include "swarm.h"
#include <jsoncpp/json/json.h>
#include <fstream>
#include "read_file.h"

int main(){
    std::cout<<"ddd"<<std::endl;
    FileReader fr{"TestCases/loc_10_comm_10_toll_5"};
    fr.print_problem();
    std::default_random_engine generator(2);
    std::uniform_real_distribution<double> distribution(0, 1);
    for(int i=0; i < 10; i++){
        std::cout<<distribution(generator)<<std::endl;
    }


/*     Swarm* swarm = new Swarm{fr.commodities_tax_free, fr.n_users, fr.transfer_costs, fr., n_comm, n_tolls_, n_parts, n_iter, no_update_lim_, num_th};
 */}