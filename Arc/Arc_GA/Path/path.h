#include <list>
#include <tuple>
#include <cstdlib>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <map>


class Path{

    public:
    std::vector<std::vector<int>> path;
    double toll_free_cost;
    std::vector<std::vector<int>> tolls;
    double current_cost;
    double profit;
    double tolerance = std::pow(10, -16);;
    
    Path() {}

    Path(std::vector<std::vector<int>> &p, const std::vector<std::vector<double>> &init_adj, 
            const std::vector<std::vector<double>> &prices, const std::vector<std::vector<int>> &toll_idxs) {
        // path = std::move(p);
        path = p;

        toll_free_cost = 0;
        current_cost = 0;
        // for(size_t i = 0; i < p.size(); i ++) toll_free_cost += init_adj[path[i][0]][path[i][1]];

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


        tolls = std::vector(toll_count, std::vector<int>(2, 0));
        int t = 0;
        profit = 0;
        for (size_t i=0; i< path.size(); i++) {
            k = 0;
            while (k < toll_idxs.size()) {
                if((path[i][0] == toll_idxs[k][0]) and (path[i][1] == toll_idxs[k][1])) {
                    tolls[t][0] = path[i][0];
                    tolls[t][1] = path[i][1];
                    profit += prices[path[i][0]][path[i][1]];
                    k += toll_idxs.size();
                    t += 1;
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

    bool contains(const std::vector<int>& toll){
        for(auto & t: tolls) if((toll[0] == t[0]) and (toll[1]== t[1])) return true;
        return false;
    }

    bool operator == (const Path& other){
        return path == other.path;
    }

    bool operator < (const Path& other) const {
        if(current_cost < other.current_cost - tolerance) {
            return true;
            }
        else{
            if((current_cost < other.current_cost + tolerance) and (profit > other.profit)) return true;
            else return false;
        }
    }


};

    std::ostream& operator <<(std::ostream& os, const Path& p){
        os << p.path[0][0]<< " ";
        for(size_t i=1; i < p.path.size(); i++)
            os << p.path[i][1] << " ";
        os<<p.current_cost<<" "<<p.profit;
        return os;
    }