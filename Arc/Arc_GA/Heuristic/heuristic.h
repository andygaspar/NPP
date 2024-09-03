#include <list>
#include <tuple>
#include <cstdlib>
#include <functional>
#include <vector>
#include <random>
#include <map>

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

class Path{

    public:
    std::vector<std::vector<int>> path;
    double toll_free_cost;
    std::vector<std::vector<int>> tolls;
    double current_cost;
    double profit;
    double tolerance = std::pow(10, -9);
    
    Path() {}

    Path(std::vector<std::vector<int>> &p, const std::vector<std::vector<double>> &init_adj, 
            const std::vector<std::vector<double>> &prices, const std::vector<std::vector<int>> &toll_idxs) {
        // path = std::move(p);
        path = p;

        toll_free_cost = 0;
        current_cost = 0;
        for(size_t i = 0; i < p.size(); i ++) toll_free_cost += init_adj[path[i][0]][path[i][1]];

        short toll_count = 0;
        size_t k = 0;
        for (size_t i= 0; i< path.size(); i++) {
            k = 0;
            toll_free_cost += init_adj[path[i][0]][path[i][1]];
            while (k < toll_idxs.size()) {
                if((path[i][0] == toll_idxs[k][0]) and (path[i][1] == toll_idxs[k][1])) {
                    toll_count += 1;
                    k += toll_idxs.size();
                }
                else k+=1;
            }
        }

        tolls = std::vector(toll_count + 1, std::vector<int>(2));
        tolls[0][0] = 0; tolls[0][1] = 0; //null toll
        for (int i=0; i< toll_count; i++) {
            k = 0;
            while (k < toll_idxs.size()) {
                if((path[i][0] == toll_idxs[k][0]) and (path[i][1] == toll_idxs[k][1])) {
                    tolls[i + 1][0] == path[i][0];
                    tolls[i + 1][1] == path[i][1];
                    profit += prices[path[i][0]][path[i][1]];
                    k += toll_idxs.size();
                }
                else k+=1;
            }
        }

        current_cost = toll_free_cost + profit;
    }

    void update(const std::vector<std::vector<double>> &prices) {
        profit = 0;
        for(size_t i= 0; i < tolls.size(); i ++) profit += prices[tolls[i][0]][tolls[i][1]];
        current_cost = toll_free_cost + profit;
    }

    bool operator < (const Path& other){
        if(current_cost < other.current_cost - tolerance) return true;
        else{
            if((current_cost < other.current_cost + tolerance) and (profit > other.profit)) return true;
            else return false;
        }
    }
};

bool compare_paths_pointers(const Path* first, const Path* second){
    return &first > &second;
}


class Commodity{

    public:

    int origin;
    int destination;
    double c_od;
    int n_users;
    double profit;
    std::list<Path*> paths;
    std::list<Path*>::iterator it; 
    std::map<std::vector<std::vector<int>>, Path> path_dict;
    std::map<std::vector<int>, Path*> toll_dict;
    std::vector<double> cost;
    std::vector<double> profits;
    std::vector<bool> visited;
    std::vector<int> previous_vertex;
    double tolerance = std::pow(10, -9);


    Commodity(int o, int d, int n_usr, const std::vector<std::vector<double>> &init_adj,
            const std::vector<std::vector<int>>& toll_idxs, 
            std::map<std::vector<int>, std::list<Commodity*>> &global_toll_dict) {
            
            origin = o;
            destination = d;
            n_users = n_usr;
            profit = 0;
            cost = std::vector<double> (init_adj.size());
            profits = std::vector<double> (init_adj.size());
            visited = std::vector<bool> (init_adj.size());
            previous_vertex = std::vector<int> (init_adj.size());
            add_initial_path(init_adj, toll_idxs, global_toll_dict);

            }
    Commodity () {};

    bool operator < (const Commodity & other) {
        return profit < other.profit;
    }

    Dij_results dijkstra(const std::vector<std::vector<double>> &adj_sol, const std::vector<std::vector<double>> &price){

        for (size_t i = 0; i < cost.size(); i++) {
            cost[i] = START_DIJKSTRA_VAL;
            visited[i] = false;
            profits[i] = 0;

        }

    
        // cost of source vertex from itself is always 0
        cost[origin] = 0;
        size_t j, min_index;
        double min_cost, max_profit;
    
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

                if (!visited[j] && adj_sol[min_index][j]
                    && cost[min_index] != START_DIJKSTRA_VAL
                    && cost[min_index] + adj_sol[min_index][j] <= cost[j] + tolerance) {

                        if (cost[min_index] + adj_sol[min_index][j] < cost[j] - tolerance) {
                            cost[j] = cost[min_index] + adj_sol[min_index][j]; 
                            profits[j] = profits[min_index] + price[min_index][j] * n_users; 
                            previous_vertex[j] = min_index; 
                        }
                        else{
                            if(profits[min_index] + price[min_index][j] * n_users > profits[j]){
                                cost[j] = cost[min_index] + adj_sol[min_index][j]; 
                                profits[j] = profits[min_index] + price[min_index][j] * n_users;  
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
        return Dij_results (cost[destination], profits[destination], path);
    }

    void add_initial_path(const std::vector<std::vector<double>> &init_adj, const std::vector<std::vector<int>> & toll_idxs, 
                        std::map<std::vector<int>, std::list<Commodity*>> &global_toll_dict) {
        std::vector<std::vector<double>> adj_sol = init_adj;
        std::vector<std::vector<double>> prices = std::vector<std::vector<double>>  (init_adj.size(), std::vector<double> (init_adj.size(), 0));

        for(size_t i=0; i<toll_idxs.size(); i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + START_DIJKSTRA_VAL;
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = START_DIJKSTRA_VAL;
        }

        Dij_results res = dijkstra(adj_sol, prices);
        Path path = Path(res.path_vector, adj_sol, prices, toll_idxs);
        path_dict[path.path] = path;
        for (const auto& entry : path_dict) {
                std::cout << entry.first[0][0]<<std::endl ;
            }
        c_od = path_dict[res.path_vector].toll_free_cost;
        paths.push_back(&path_dict[path.path]);

        for(const auto toll : path_dict[res.path_vector].tolls) {
            toll_dict[toll] = &path_dict[res.path_vector];
            if(global_toll_dict.find(toll) == global_toll_dict.end()) global_toll_dict[toll] = std::list{this};
        }
    }

    void update_from_price(const std::vector<std::vector<double>> &adj, const std::vector<std::vector<double>> &prices,
                            const std::vector<std::vector<double>> &init_adj, const std::vector<std::vector<int>> &toll_idxs,
                            std::map<std::vector<int>, std::list<Commodity*>> &global_toll_dict){
        Dij_results res = dijkstra(adj, prices);
        if(path_dict.find(res.path_vector)!=path_dict.end()){
            path_dict[res.path_vector].update(prices);
        }
        else{
            Path path = Path(res.path_vector, init_adj, prices, toll_idxs);
            // std::vector<std::vector<int>> path_key = path.path;
            path_dict[path.path] = path;
            for (const auto& entry : path_dict) {
                std::cout << entry.first[0][0]<<std::endl ;
            }
            paths.push_back(&path_dict[path.path]);
            for(const auto toll : path_dict[res.path_vector].tolls) {
                if(toll_dict.find(toll) == toll_dict.end()) toll_dict[toll] = &path_dict[res.path_vector];
                if(global_toll_dict.find(toll) == global_toll_dict.end()) global_toll_dict[toll] = std::list{this};
            }
            paths.sort(compare_paths_pointers);
        }
        profit = res.profit;
       }

    std::vector<std::vector<int>> get_path(int i){
        it = paths.begin();
        std::advance(it, i);
        return (*it)->path;
    }

};


class Heuristic{
    public:
    size_t n_commodities;
    size_t n_tolls;
    std::vector<std::vector<double>> init_adj;
    std::vector<std::vector<double>> adj_sol;
    std::vector<std::vector<double>> prices;
    double tolerance = pow(10, -9);
    
    std::vector<Commodity> commodities;
    std::vector<std::vector<int>> toll_idxs;
    std::map<std::vector<int>, std::list<Commodity*>> toll_dict;
    std::list<Path*>::iterator it; 

    Heuristic(const std::vector<int> origins, const std::vector<int> destinations, 
        const std::vector<int> n_users, const std::vector<std::vector<double>> adj, const std::vector<std::vector<int>> t_idx) 
        {   
            n_commodities = origins.size();
            n_tolls = t_idx.size();
            toll_idxs = t_idx;
            init_adj = adj;
            adj_sol = adj;
            prices = std::vector<std::vector<double>>  (adj.size(), std::vector<double> (adj.size(), 0));
            commodities = std::vector<Commodity> (n_commodities);
            std::map<std::vector<std::vector<int>>, std::tuple<Commodity*>> dict = std::map<std::vector<std::vector<int>>, std::tuple<Commodity*>> ();
            for(size_t i=0; i < n_commodities; i++) {
                commodities[i] = Commodity(origins[i], destinations[i], n_users[i], adj, toll_idxs, toll_dict);

            }
        }

    Heuristic() {}

    void run(std::vector<double>& p){

        for(size_t i=0; i<n_tolls; i++){
            adj_sol[toll_idxs[i][0]][toll_idxs[i][1]] = init_adj[toll_idxs[i][0]][toll_idxs[i][1]] + p[i];
            prices[toll_idxs[i][0]][toll_idxs[i][1]] = p[i];
        }
        bool improving = true;

        for(size_t i=0; i<n_commodities; i++) {
            commodities[i].update_from_price(adj_sol, prices, init_adj, toll_idxs, toll_dict);
        }

        std::sort(commodities.begin(), commodities.end());
        double diff_cost;
        while(improving){
            improving = false;
            for(auto com: commodities) {
                for(auto toll: com.paths){
                    diff_cost = get_path(com.paths, 1)->current_cost - get_path(com.paths, 0)->current_cost;
                    if(diff_cost > tolerance) {
                        
                    }
                }
            }
            
        }

        int i;


    }

    Path* get_path(std::list<Path*>& path, int i){
       it  = path.begin();
       std::advance(it, i);
       return *it;
    }
    

};