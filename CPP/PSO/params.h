#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include <fstream>
#include "../Utils/utils_.h"


struct Params{
    double w1; 
    double w2;
    double w3;
    double c_soc1;
    double c_soc2;
    double c_soc3;
    double c_cog1;
    double c_cog2;
    double c_cog3;
    double L;
    double U;
    double L1;
    double L2;
    double L3;

    double U1;
    double U2;
    double U3;
    double limit_sigma_1;
    double limit_sigma_2;
    double limit_sigma_3;
    double init_commodity_val;

    int stat_frequency;

    Params(){
        // std::cout << "Current path is " << std::filesystem::current_path() << '\n';
        std::ifstream params_file("pso_params.json", std::ifstream::binary);
        Json::Value params;
        params_file >> params;
        w1 = params["w1"].asDouble();  w2 = params["w2"].asDouble(); w3 = params["w3"].asDouble(); 
        c_soc1 = params["c_soc1"].asDouble(); c_soc2 = params["c_soc2"].asDouble(); c_soc3 =params["c_soc3"].asDouble(); 
        c_cog1 = params["c_cog1"].asDouble();  c_cog2 = params["c_cog2"].asDouble(); c_cog3 =params["c_cog3"].asDouble();
        L =params["L"].asDouble(); U = params["U"].asDouble();
        L1 =params["L1"].asDouble(); L2 = params["L2"].asDouble(); L3 =params["L3"].asDouble(); 
        U1 = params["U1"].asDouble(); U2 =params["U2"].asDouble(); U3=params["U3"].asDouble(); 
        limit_sigma_1 = params["limit_sigma_1"].asDouble(); limit_sigma_2 = params["limit_sigma_2"].asDouble(); limit_sigma_3 = params["limit_sigma_3"].asDouble();
        init_commodity_val=pow(10, 5);
        stat_frequency = params["stats_frequency"].asInt();
        }
    };