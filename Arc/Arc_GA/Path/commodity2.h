#include <list>
#include <tuple>
#include <cstdlib>
#include <functional>
#include <vector>
#include <random>
#include <map>
#include <iostream>
#include <gurobi_c++.h>
#include <numeric>
#include "path.h"


double START_DIJKSTRA_VAL = 100000;



class Dij_results{
    public:
    double cost;
    double profit; 
    std::vector<std::vector<int>> path_vector;
    
    Dij_results(double c, double p, std::vector<std::vector<int>>& pv) {
        cost = c;
        profit = p;
        path_vector = std::move(pv);
    }

};


bool compare_paths_pointers(const Path* first, const Path* second){
    return (*first) < (*second);
}




class Commodity{

    public:

    int origin;
    int destination;
    double c_od;
    int n_users;
    double profit;

    double current_run_val;
    double min_cost;

    std::list<Path*> paths;
    std::list<Path*>::iterator it; 
    std::map<std::vector<std::vector<int>>, Path> path_dict;
    std::map<std::vector<int>, std::list<Path*>> toll_dict;
    std::vector<double> cost;
    std::vector<double> profits;
    std::vector<bool> visited;
    std::vector<int> previous_vertex;
    std::vector<std::vector<double>> init_adj;
    std::vector<std::vector<double>> adj_sol;
    std::vector<std::vector<double>> prices;
    std::vector<std::vector<int>> toll_idxs;
    double tolerance = std::pow(10, -9);


    Commodity(int o, int d, int n_usr, const std::vector<std::vector<double>> &init_adj_,
            const std::vector<std::vector<int>>& t_idxs) {
            
            origin = o;
            destination = d;
            n_users = n_usr;
            profit = 0;
            toll_idxs = t_idxs;
            cost = std::vector<double> (init_adj_.size());
            profits = std::vector<double> (init_adj_.size());
            visited = std::vector<bool> (init_adj_.size());
            previous_vertex = std::vector<int> (init_adj_.size());

            init_adj = init_adj_;
            adj_sol = init_adj;
            prices = std::vector<std::vector<double>> (init_adj.size(), std::vector<double> (init_adj.size(), 0));
            add_initial_path();



            }
    Commodity () {};

    bool operator ==(const Commodity & other) {
        if((origin == other.origin) and (destination == other.destination)) return true;
        else return false;
    }

    bool operator < (const Commodity & other) {
        return profit > other.profit; // to get reverse order
    }

    std::ostream& operator <<(std::ostream& os){
        os << origin << " " << destination;
        return os;
    }

    void add_paths(std::map<std::vector<std::vector<int>>, Path>& new_path_dict){
        path_dict = new_path_dict;
        paths.clear();
        for(auto & entry: path_dict) paths.push_back(&entry.second);
    }


    void add_initial_path() {

        for(size_t i=0; i<toll_idxs.size(); i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + START_DIJKSTRA_VAL;
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = START_DIJKSTRA_VAL;
        }

        Dij_results res = dijkstra(adj_sol, prices);
        Path path = Path(res.path_vector, init_adj, prices, toll_idxs);
        path_dict[path.path] = path;
        c_od = path_dict[res.path_vector].toll_free_cost;
        paths.push_back(&path_dict[path.path]);
    }

    double run_dijkstra(std::vector<double>& p){

        for(size_t i=0; i<toll_idxs.size(); i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + p[i];
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = p[i];
        }
        Dij_results res = dijkstra(adj_sol, prices);
        if(path_dict.find(res.path_vector)!=path_dict.end()){
            path_dict[res.path_vector].update(prices);
        }
        else{
            Path path = Path(res.path_vector, init_adj, prices, toll_idxs);
            path_dict[path.path] = path;
            paths.push_back(&path_dict[path.path]);
            for(const auto & toll : path_dict[res.path_vector].tolls) {

                toll_dict[toll].push_back(&path_dict[path.path]);

            }
        }

        paths.sort(compare_paths_pointers);
        return res.profit;
       }

    void update(const std::vector<std::vector<double>> &prices, const std::vector<int> & toll) {
        for(auto & path: toll_dict[toll]){
            path -> update(prices);
        }
        paths.sort(compare_paths_pointers);
        profit = get_path(0)-> profit * n_users;
    }

    Path* get_path(int i){
        it = paths.begin();
        std::advance(it, i);
        return *it;
    }



    Dij_results dijkstra(const std::vector<std::vector<double>> &a_sol, const std::vector<std::vector<double>> &pr){

        for (size_t i = 0; i < cost.size(); i++) {
            cost[i] = START_DIJKSTRA_VAL;
            visited[i] = false;
            profits[i] = 0;

        }

    
        // cost of source vertex from itself is always 0
        cost[origin] = 0;
        size_t j; int min_index = 0;
        double max_profit;
        min_cost = START_DIJKSTRA_VAL; 
    
        // Find shortest path for all vertices
        for (size_t count = 0; count < cost.size() - 1; count++) {

            min_cost = START_DIJKSTRA_VAL; 
            max_profit = 0;
            for (j = 0; j < cost.size(); j++)
                if (visited[j] == false && cost[j] <= min_cost + tolerance) {
                    if (cost[j] < min_cost - tolerance) {
                        min_cost = cost[j], 
                        min_index = j;
                        max_profit = profits[j];  
                    }
                    else{
                        if(profits[j] > max_profit){
                            min_cost = cost[j], 
                            min_index = j;
                            max_profit = profits[j];  
                        }
                    }
                    
                }
                    
    
            // Mark the picked vertex as processed
            visited[min_index] = true;
    
            // Update cost value of the adjacent vertices of the
            // picked vertex.
            for (j = 0; j < cost.size(); j++)

                if (!visited[j] && a_sol[min_index][j]
                    && cost[min_index] != START_DIJKSTRA_VAL
                    && cost[min_index] + a_sol[min_index][j] <= cost[j] + tolerance) {

                        if (cost[min_index] + a_sol[min_index][j] < cost[j] - tolerance) {
                            cost[j] = cost[min_index] + a_sol[min_index][j]; 
                            profits[j] = profits[min_index] + pr[min_index][j] ; 
                            previous_vertex[j] = min_index; 
                        }
                        else{
                            if(profits[min_index] + pr[min_index][j]  > profits[j]){
                                cost[j] = cost[min_index] + a_sol[min_index][j]; 
                                profits[j] = profits[min_index] + pr[min_index][j];  
                                previous_vertex[j] = min_index; 
                            }
                        }    
                    }
                    
        }

        size_t size = 0;
        int current_node = destination;
        while(current_node != origin) {
            current_node = previous_vertex[current_node];
            size += 1;
        }

        std::vector<std::vector<int>> path = std::vector<std::vector<int>>  (size, std::vector<int> (2));
        current_node = destination;
        int k = size - 1;
        while(current_node != origin) {
            path[k][0] = previous_vertex[current_node];
            path[k][1] = current_node;
            current_node = previous_vertex[current_node];
            k--;
        }
        return Dij_results (cost[destination], profits[destination] * n_users, path);
    }

    void print_map(const std::map<std::vector<int>, std::list<Path*>> & map) {
        for(auto it = map.cbegin(); it != map.cend(); ++it)
        {
            std::cout << it->first[0] << " " << it->first[1] << "  path key \n";
            for(auto & path: it->second) {
                std::cout << path->path[0][0]<< " ";
                    for(size_t i=1; i < path->path.size(); i++)
                        std::cout << path->path[i][1] << " ";
            }
        }
                std::cout <<std::endl;
    }



    double eval(std::vector<double>& p){
        

        for(size_t i=0; i<toll_idxs.size(); i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + p[i];
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = p[i];
        }
        for(auto & path : paths) path->update(prices);

        current_run_val = 0;
        min_cost = c_od;
        for(auto & path : paths) {
            if(path -> current_cost < min_cost - tolerance) {
                current_run_val = path -> profit;
                min_cost = path -> current_cost;
            }
            else{
                if((path -> current_cost < min_cost + tolerance) and (path -> profit > current_run_val)) {
                    current_run_val = path -> profit;
                    min_cost = path -> current_cost; 
                }
            }
        }

        return current_run_val*n_users;
    }


    void set_cost(std::vector<double>& p){
        for(size_t i=0; i<toll_idxs.size(); i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + p[i];
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = p[i];
        }
        for(auto & path : paths) path->update(prices);
    }

};



void solveModel(std::vector<Commodity>& commodities, int n_tolls, std::vector<double>& p, std::vector<double>& ub, double val, GRBEnv & env) {
    try {
        // GRBEnv env;
        env.set(GRB_IntParam_OutputFlag, 0);
        env.set(GRB_IntParam_Threads, 1);
        // env.set(GRB_IntParam_ConcurrentJobs, 1);  
        env.set(GRB_IntParam_Presolve, 0);   
        GRBModel model = GRBModel(env);
        model.set(GRB_IntParam_Threads, 1);

        // std::cout<<" val "<<val<<std::endl;
        


        // Creazione delle variabili T_var
        GRBVar* T_var = model.addVars(n_tolls, GRB_CONTINUOUS);
        for (int k = 0; k < n_tolls; ++k) {
            T_var[k].set(GRB_DoubleAttr_UB, ub[k]);
        }

        // Vettore per total_bool
        std::vector<int> total_bool(n_tolls, 0);
        std::list<Path*>::iterator first_element;
        std::list<Path*>::iterator it;

        for (auto& commodity : commodities) {
            commodity.set_cost(p);
            commodity.paths.sort(compare_paths_pointers);
            
            // Aggiorna total_bool con il primo path
            first_element = commodity.paths.begin();
            it = commodity.paths.begin();
            if(not (*first_element)->is_toll_free) {

                for (int k = 0; k < n_tolls; ++k) {
                    total_bool[k] += (*first_element)->toll_bool[k];
                }

                
                // Aggiungi i vincoli per gli altri paths
                for (size_t j = 1; j < commodity.paths.size(); j++) {

                    std::advance(it, 1);

                    GRBLinExpr lhs = (*first_element)->toll_free_cost;
                    GRBLinExpr rhs = (*it)->toll_free_cost;
                    
                    // Calcola (T_var * commodity.paths[0].toll_bool).sum()
                    for (int k = 0; k < n_tolls; ++k) {
                        if ((*first_element)->toll_bool[k] != 0) {
                            lhs += T_var[k];
                        }
                    }
                    
                    // Calcola (T_var * commodity.paths[j].toll_bool).sum()
                    for (int k = 0; k < n_tolls; ++k) {
                        if ((*it)->toll_bool[k] != 0) {
                            rhs += T_var[k];
                        }
                    }
                    
                    model.addConstr(lhs <= rhs);
                    
                    // Aggiorna total_bool con il path corrente
                    for (int k = 0; k < n_tolls; ++k) {
                        total_bool[k] += (*it)->toll_bool[k];
                    }
                }
            }
        }

        // Imposta la funzione obiettivo
        GRBLinExpr objective = 0;
        for (auto& commodity : commodities) {
            first_element = commodity.paths.begin();
            for (int k = 0; k < n_tolls; ++k) {
                if ((*first_element)->toll_bool[k] != 0) {
                    objective += T_var[k] * commodity.n_users;
                }
            }
        }


        
        model.setObjective(objective, GRB_MAXIMIZE);

        // Risolvi il modello
        model.optimize();

        for (int k = 0; k < n_tolls; ++k) {
            if(total_bool[k] >0 ) p[k] = T_var[k].get(GRB_DoubleAttr_X);
            // std::cout << total_bool[k]<<" "<< p[k] <<" "<< value << std::endl;
        }   

        // if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        //     double objVal = model.get(GRB_DoubleAttr_ObjVal);
        //     std::cout << "Optimal objective value: " << objVal << std::endl;
        // } else {
        //     std::cout << "Model did not solve to optimality. Status: "<< model.get(GRB_IntAttr_Status) << std::endl;
        
        // }
              

        // Pulizia della memoria
        delete[] T_var;

    } catch (GRBException& e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    } catch (...) {
        std::cout << "Exception during optimization" << std::endl;
    }
}