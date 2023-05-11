#include "swarm_.cpp"


/* Initialize a large number of particles in the space with latin hypercube, letting them evolve */
/* without limitations in the searching space but letting them communicate only between subgroups. */
/* Then after a certain number of iterations the particles number is reduced mantaining only the best particles so far. */
/* This whole process is repeated N times reducing each time the searching space around the best solution found. */
std::vector<double> recurrent_run_PSO_on_smaller_domain(FileReader file_reader,short n_particles,short N_PARTICLES, int n_iterations, short n_first_iterations, short N_div, short n_cut, int N, double lowering_rate=0.35, int n_update_lim=500, short num_th=1) {

    std::vector<double> new_u_bound(file_reader.n_tolls);
    std::vector<double> new_l_bound(file_reader.n_tolls);
    std::vector<double> solution(file_reader.n_tolls);
    double solution_val;
    double tmp_delta;
    bool ok = true;

    Swarm s{file_reader.commodities_tax_free, 
                file_reader.n_users,file_reader.transfer_costs, 
                file_reader.upper_bounds, 
                file_reader.lower_bounds, 
                file_reader.n_commodities, 
                file_reader.n_tolls, 
                n_particles, 
                n_iterations, 
                num_th, N_PARTICLES,N_div,n_cut, n_update_lim}; 
    s.run_and_lower(n_first_iterations);
    
    for (int i=0;i<file_reader.n_tolls;i++) solution[i]=s.cube_p_best[s.best_idx][i];
    solution_val=s.cube_best_val[s.best_idx];

    /*--- initialize the problem ---*/
    for(int j=0;j<N;j++){

        for (int i=0;i<file_reader.n_tolls;i++) {
            tmp_delta = file_reader.upper_bounds[i]*lowering_rate*(2*N-j)/(2*N);
            new_u_bound[i] = std::min(file_reader.upper_bounds[i], solution[i]+tmp_delta);
            new_l_bound[i] = std::max(file_reader.lower_bounds[i], solution[i]-tmp_delta);
        }
        std::cout<<"[";
        for (int i=0;i<file_reader.n_tolls;i++) {
            if ((new_u_bound[i]<p_optimum[i] && std::abs(new_u_bound[i]-p_optimum[i])>0.000000001) || (new_l_bound[i]>p_optimum[i]&& std::abs(new_l_bound[i]-p_optimum[i])>0.000000001)){ 
                ok=false;
                std::cout<<"("<<i<<") "<<new_l_bound[i]<<"|"<<p_optimum[i]<<"|"<<new_u_bound[i]<<", ";
            }
        }
        std::cout<<"]"<<std::endl;
        if(ok) std::cout<<"-------------------- OK ------------------------"<<std::endl;
        else{ ok=true; std::cout<<"-------------------- OUT ------------------------"<<std::endl;}

        Swarm s_{file_reader.commodities_tax_free, 
            file_reader.n_users,file_reader.transfer_costs, 
            new_u_bound.data(), 
            new_l_bound.data(), 
            file_reader.n_commodities, 
            file_reader.n_tolls, 
            n_particles, 
            n_iterations, 
            num_th, N_PARTICLES,N_div,n_cut, n_update_lim}; 
        s_.run_and_lower(n_first_iterations);

        if (s_.cube_best_val[s_.best_idx]>solution_val) {
            solution_val = s_.cube_best_val[s_.best_idx];
            for (int k=0;k<file_reader.n_tolls;k++) solution[k]=s_.cube_p_best[s_.best_idx][k];
        }       
    }
    std::cout<<"------------------------------ MAX FOUND: "<<solution_val<<"------------------------------"<<std::endl;
    return solution;
}