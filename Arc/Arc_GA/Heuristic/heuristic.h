#include <list>
#include <tuple>
#include <cstdlib>
#include <functional>
#include <vector>
#include <random>
#include <map>
#include "commodity.h"



class Heuristic{
    public:
    size_t n_commodities;
    size_t n_tolls;
    std::vector<std::vector<double>> init_adj;
    std::vector<std::vector<double>> adj_sol;
    std::vector<std::vector<double>> prices;
    double tolerance = pow(10, -16);
    
    std::list<Commodity> commodities;
    std::vector<std::vector<int>> toll_idxs;
    std::map<std::vector<int>, std::list<Commodity*>> toll_dict;
    std::list<Path*>::iterator it; 
    std::list<Path*>::iterator prev; 

    double d_cost, sd_cost;

    Heuristic(const std::vector<int> origins, const std::vector<int> destinations, 
        const std::vector<int> n_users, const std::vector<std::vector<double>> adj, const std::vector<std::vector<int>> t_idx) 
        {   
            n_commodities = origins.size();
            n_tolls = t_idx.size();
            toll_idxs = t_idx;
            init_adj = adj;
            adj_sol = adj;
            prices = std::vector<std::vector<double>>  (adj.size(), std::vector<double> (adj.size(), 0));
            std::map<std::vector<std::vector<int>>, std::tuple<Commodity*>> dict = std::map<std::vector<std::vector<int>>, std::tuple<Commodity*>> ();
            for(size_t i=0; i < n_commodities; i++) {
                commodities.push_back(Commodity(origins[i], destinations[i], n_users[i], adj, toll_idxs, toll_dict));

            }
        }

    Heuristic() {}

    void run(std::vector<double>& p, double* val){

        for(size_t i=0; i<n_tolls; i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + p[i];
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = p[i];
        }
        bool improving = true;
        // std::cout<<"ii "<<*val<<std::endl;
        for(auto & com: commodities) {
            com.add_solution(adj_sol, prices, init_adj, toll_idxs, toll_dict);
            std::cout<<com.paths.size()<< " ";
            if(com.paths.size() == 7){
                int i=0;
            }
        }
        

        double diff_cost, second_diff;
        double compare_diff_cost, second_c_diff_cost;
        double initial_profit=0;
        for(auto & com: commodities) initial_profit += com.profit;
        std::cout<<initial_profit<<std::endl;
        // double max_cost=0;
        // for(auto & com: commodities) max_cost += com.c_od;
        // std::cout<<"init "<<max_cost<<std::endl;

        while(improving){
            improving = false;
            for(auto & com: commodities) {
                for(auto & toll: get_path(com.paths, 0) -> tolls){
                    diff_cost = get_diff(com, toll);
                    if(diff_cost > tolerance + 0.00001) {
                        for(auto & other_commodity: toll_dict[toll]){
                            if(not(com == *other_commodity) and (get_path(other_commodity->paths, 0)->contains(toll))){
                                compare_diff_cost = get_diff(*other_commodity, toll);
                                if(compare_diff_cost < diff_cost - tolerance){
                                    diff_cost = compare_diff_cost;
                                }
                            }
                        }
                    }
                    if(diff_cost > tolerance + 0.001){
                        improving = true;
                        std::cout<<"dddd "<<toll[0]<<"-"<<toll[1]<<" "<<diff_cost<<std::endl;
                        prices[toll[0]][toll[1]] += diff_cost - 0.0001;
                        update_paths_costs(prices, toll);
                        double max_cost = 0;
                        for(auto & c: commodities) {
                            max_cost += c.profit; 
                            for(auto & ttt: get_path(c.paths, 0)->tolls) std::cout<<ttt[0] <<" ";
                            std::cout<<"profit "<<c.profit<<std::endl;
                            }
                        std::cout<<"partial "<<max_cost<<std::endl;
                    }
                }
            }
        }

        for(size_t i=0; i<toll_idxs.size(); i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + prices[toll_idxs[i][0]][toll_idxs[i][1]] ;
        }



        double new_profit = 0;
        for(auto & com: commodities) {
            com.add_solution(adj_sol, prices, init_adj, toll_idxs, toll_dict);
            new_profit += com.profit;
            std::cout<<com.paths.size()<< " ";


        }
                

        std::cout<<new_profit<<std::endl;std::cout<<std::endl;



        if(new_profit > initial_profit) {
            for(size_t i = 0; i<p.size(); i++) {
                p[i] = prices[toll_idxs[i][0]][toll_idxs[i][1]];
            }
            *val = new_profit;
        }
    }


    double get_diff(Commodity& com, std::vector<int> & toll){
        d_cost = get_path(com.paths, 1)->current_cost - get_path(com.paths, 0)->current_cost;
        sd_cost = first_without_toll_t(com.paths, toll)->current_cost - last_with_toll_t(com.paths, toll)->current_cost;
        if (d_cost < sd_cost) return d_cost;
        else return sd_cost;        
    }

    Path* get_path(std::list<Path*>& path, int i){
       it  = path.begin();
       std::advance(it, i);
       return *it;
    }

    Path* first_without_toll_t(std::list<Path*>& paths, std::vector<int>& toll){
       it  = paths.begin();
       size_t i = 0;
       while(i < paths.size()){
            std::advance(it, 1);
            if(not ((*it)->contains(toll))) i += paths.size();
       }
       return *it;
    }

    Path* last_with_toll_t(std::list<Path*>& paths, std::vector<int>& toll){
       it  = paths.begin();
       size_t i = 0;
       while(i < paths.size()){
            std::advance(it, 1);
            if(not ((*it)->contains(toll))) {i += paths.size(); std::advance(it, -1);}
       }
       return *it;
    }

    void update_paths_costs(const std::vector<std::vector<double>>& new_prices, const std::vector<int>& toll){
        
        for(auto & com: toll_dict[toll]){
            com -> update(prices, toll);
            
           }
    }

    void print_glob_map(const std::map<std::vector<int>, std::list<Commodity*>> & map) {
        for(auto it = map.cbegin(); it != map.cend(); ++it)
        {
            std::cout << it->first[0] << " " << it->first[1] << "  com key \n";
            for(auto & com: it->second) {
                std::cout << com ->origin<<"  "<< com -> destination <<std::endl;
            }
        }
        std::cout <<std::endl;
    }
    

};


