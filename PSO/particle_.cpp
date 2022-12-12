#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
//#include "utils.cpp"

/*
Particle class implements a particle object such that:
- ndim: dimensionality of the space
- p: coordinates of the particle's position in the space
- v: particle's velocity vector
- personal_best: coordinates associated with the best position found so far
          in the particle's history according to a given objective function

The particle's position can be initialized by hand or randomly in the space,
setting the limits.
The methods implemented aim to:
- update the position of the particle according PSO method
- update the velocity of the particle according PSO method
*/

class Particle {
    public:
    short n_commodities;
    short n_tolls;
    int idx;
    std::vector<std::vector<double>> transfer_costs;
    std::vector<double> upper_bounds;
    std::vector<double> commodities_tax_free;
    std::vector<int> n_users;
    std::vector<double> p;
    std::vector<double> p_past;
    std::vector<double> v;
    std::vector<double> personal_best;
    double w=0.9;
    double c_soc=1.49445;
    double c_cog=1.49445;
    double d_g;
    double fitness;
    std::vector<double> fitness_memb {0,0,0}; //[better, same, worse]
    double sigma;
    std::vector<double> sigma_memb {0,0,0}; //[same, near, far]
    double L;
    double U;
    double personal_best_val;
    double run_cost;
    double run_cost_past;
    double commodity_cost;
    double init_commodity_cost;
    double toll_cost;
    double sigma_max;
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    Particle() {}

    Particle(double* comm_tax_free, int* n_usr, double* trans_costs, double* u_bounds, short n_comm, short n_to, int i, double d_max) {
        idx = i;
        n_commodities=n_comm;
        n_tolls=n_to;
        sigma_max = d_max;

        commodities_tax_free = std::vector<double> (n_commodities);
        for(int i=0; i< n_commodities; i++) commodities_tax_free[i] = comm_tax_free[i];

        transfer_costs = std::vector<std::vector<double>>(n_commodities);
        for(int i =0; i<n_commodities; i++)  {
            transfer_costs[i] = std::vector<double>(n_tolls);
            for(int j=0; j< n_tolls; j++) transfer_costs[i][j]=trans_costs[i*n_tolls + j];
        }

        n_users = std::vector<int> (n_commodities);
        for(int i=0; i< n_commodities; i++) n_users[i] = n_usr[i];

        upper_bounds = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) upper_bounds[j] = u_bounds[j];

        p = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) p[j] = get_rand(0, 1) * upper_bounds[j];
        //p = std::vector<double> {4.7705059076017395, 5.751548443390286, 6.123415429538003, 5.876942404728085, 6.628613111030518, 5.169450847114579, 6.00787280243814, 5.867648207951801, 6.5696984249220804, 7.4662265455815255, 6.487424887555925, 5.8454109651643265, 5.1018414189363614, 6.191383747824933, 5.1868217232199605, 6.507918711696853, 5.488096235349984, 4.433967178416594, 5.257151673359806, 2.7032893589721123, 3.8229179024349342, 3.890758470694619, 5.11888579827513, 2.947795436783654, 4.8241387642744975, 4.988454194786504, 3.8634870073702334, 4.654514280689021, 3.4983219903094067, 4.705107273280182, 1.839732590000109, 3.509639149559174, 3.193196647142532, 4.0964714824193855, 4.187276178945115, 3.2936634865324486, 3.740232267758783, 4.092052976034404, 3.8946761835025043, 5.134643870662844, 2.9775072332720924, 5.805181300063042, 5.969496730575051, 4.84452954315878, 4.777349985513416, 4.479364526097951, 5.392945290777055, 2.8207751257886553, 3.1327388144927575, 1.8893121693037518, 2.812121775065071, 5.16831871473366, 2.132745134366049, 3.1811563811813013, 5.014304726965326, 5.5065108568105625, 4.097135776734916, 6.177048286210761, 6.341363716722768, 5.216396529306499, 5.806731078225644, 4.85123151224567, 5.764812276924774, 3.4423459515804797, 4.531888280469053, 3.5273262558640806, 3.946703327818895, 5.540185700881379, 2.5046121205137677, 3.597656206003924, 6.31027262280552, 4.164976344994599, 4.744991759110919, 5.569562806044409, 4.644868505212369, 5.845901105219411, 4.333046712343965, 5.896494097810574, 2.4401207123646897, 4.701025974089564, 4.384583471672922, 5.2878583069497775, 5.169755747124986, 4.485050311062841, 4.931619092289175, 5.39310367257511, 6.682245967703276, 6.846561398215282, 5.95279385219464, 6.849321972854083, 5.870520314828484, 6.270009958417287, 4.484936846208919, 5.574479175097492, 4.56991715049252, 4.9892942224473344, 6.045383382373894, 4.080599605251212, 4.873043750898869, 5.223083703787337, 5.3873991342993435, 4.2624319468830745, 4.928732154989003, 3.897266929822246, 4.9793251475801625, 2.238677529512948, 3.783857023859156, 3.467414521442512, 4.370689356719367, 4.586221118457955, 3.5678813608324305, 4.014450142058763, 6.2258210896229045, 6.623331281594838, 7.519859402254282, 6.541057744228683, 5.649269649824911, 5.155474275609118, 6.245016604497691, 5.2404545798927185, 5.659831651847533, 5.5417290920227416, 3.589715989717284, 5.310784530032562, 6.787646712106845, 7.68417483276629, 6.705373174740689, 5.390281822652099, 5.319789706121126, 6.409332035009697, 5.404770010404725, 5.82414708235954, 5.706044522534748, 3.602410120275776, 5.47509996054457, 6.559207645350019, 5.58040598732442, 6.211095272308851, 4.194822518704855, 5.284364847593428, 4.279802822988456, 4.699179894943271, 5.986468696265456, 3.7356202634207225, 4.3501327731282995, 6.194042628289191, 7.107623392968295, 4.535453227979895, 5.217185289948064, 4.212623265343092, 4.632000337297907, 6.8829968169249, 3.8474232365572885, 4.895834483372541, 6.128821734942695, 3.829657501644027, 4.9191998305326, 3.9146378059276277, 4.538759724068731, 5.9041951588993005, 3.735951728181794, 4.182520509408128, 4.7432382663231305, 5.832780595211704, 4.828218570606731, 5.247595642561546, 5.129493082736754, 3.025858680477782, 4.898548520746575, 3.260610430223302, 2.25604840561833, 2.718850030252422, 4.5186116902797355, 1.4830381099121244, 2.531449356727377, 3.6226461026631753, 4.525920937940031, 5.608154019168309, 3.7231129420530937, 4.169681723279428, 2.2473950548947474, 4.603591994563336, 1.5680184141957252, 2.947354599766271, 5.022969066518151, 2.2528881788033583, 3.8506294350431265, 2.988312515758622, 4.67392194470318, 3.0478214391561895};
        //std::cout<<"("<<idx<<") "<<"p[0] = "<< p[0]<<std::endl;
        p_past = std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) p_past[j] = p[j];

        v= std::vector<double> (n_tolls);
        for(int j=0; j< n_tolls; j++) v[j] = get_rand(0, 0.4) * upper_bounds[j];

        personal_best = p;
        personal_best_val = 0;
        init_commodity_cost = pow(10, 5);
        L=0;
        U=10;
    }
    ~Particle() {}
    //double get_personal_best_val() {return personal_best_val;}
    void update_fitness(double best) {
        if (best!=0)
            fitness = (compute_sigma(p_past)/sigma_max)*(std::max(best,run_cost_past)-std::max(best,run_cost))/(std::abs(personal_best_val));
        else
            fitness = 1;
        }
    void update_sigma(double* g) {sigma = compute_sigma(std::vector<double>(g, g + n_tolls));}
    void evaluate_fitness_memb();
    void evaluate_sigma_memb();
    void update_pos();
    void reflection();
    void update_vel(double* g,double* u_bounds);
    void update_best(double new_personal_best_val) {personal_best_val = new_personal_best_val; personal_best=p;}

    double get_rand(double start, double end) {
        std::default_random_engine generator(std::rand());
        std::uniform_real_distribution<double> distribution(start, end);
        return distribution(generator);
        }

    double compute_obj_and_update_best();
    double compute_sigma(std::vector<double> p_);
    void update_inertia();
    void update_c_soc();
    void update_c_cog();
    void update_L();
    void update_U();
    void update_params(double* g, double best);
    void print();
};

double Particle::compute_sigma(std::vector<double> p_) {
    double sum=0;
    for (int i=0; i<n_tolls; i++) {
        sum += std::pow((p[i]-p_[i]),2);
    }
    return std::sqrt(sum);
}

void Particle::evaluate_fitness_memb() {
    if (fitness<=0){
        fitness_memb[0] = -fitness;
        fitness_memb[1] = 1+fitness;
        fitness_memb[2] = 0;
    }
    else{
        fitness_memb[0] = 0;
        fitness_memb[1] = 1-fitness;
        fitness_memb[2] = fitness;
    }
}

void Particle::evaluate_sigma_memb() {
    double l1 = 0.2;
    double l2 = 0.4;
    double l3 = 0.7;
    if (sigma<=l1*sigma_max) {
        sigma_memb[0] = 1;
        sigma_memb[1] = 0;
        sigma_memb[2] = 0;
    }
    if (sigma>l1*sigma_max && sigma<=l2*sigma_max) {
        sigma_memb[0] = (l2*sigma_max - sigma)/((l2-l1)*sigma_max);
        sigma_memb[1] = (sigma - l1*sigma_max)/((l2-l1)*sigma_max);
        sigma_memb[2] = 0;
    }
    if (sigma>l2*sigma_max && sigma<=l3*sigma_max) {
        sigma_memb[0] = 0;
        sigma_memb[1] = (l3*sigma_max - sigma)/((l3-l2)*sigma_max);
        sigma_memb[2] = (sigma - l2*sigma_max)/((l3-l2)*sigma_max);
    }
    if (sigma>l3*sigma_max) {
        sigma_memb[0] = 0;
        sigma_memb[1] = 0;
        sigma_memb[2] = 1;
    }
}

void Particle::update_inertia() {
    w = 0;
    w += (fitness_memb[2] + sigma_memb[0])*0.2; 
    w += (fitness_memb[1] + sigma_memb[1])*0.6;
    w += (fitness_memb[0] + sigma_memb[2])*1.6;
}

void Particle::update_c_soc() {
    c_soc = 0;
    c_soc += (fitness_memb[0] + sigma_memb[1])*0.3;
    c_soc += (fitness_memb[1] + sigma_memb[0])*0.6;
    c_soc += (fitness_memb[2] + sigma_memb[2])*1;
}

void Particle::update_c_cog() {
    c_cog = 0;
    c_cog += (sigma_memb[2])*0.2;
    c_cog += (fitness_memb[2] + fitness_memb[1] + sigma_memb[0] + sigma_memb[1])*1;
    c_cog += (fitness_memb[0])*1.8;
}

void Particle::update_L() {
    L = 0;
    L += (fitness_memb[1] + fitness_memb[0] + sigma_memb[2])*0;
    L += (sigma_memb[0] + sigma_memb[1])*0.001;
    L += (fitness_memb[2])*0.01;
}

void Particle::update_U() {
    U = 0;
    U += (sigma_memb[0])*0.08;
    U += (fitness_memb[1] + fitness_memb[0] + sigma_memb[1])*0.25;
    U += (fitness_memb[2] + sigma_memb[2])*0.5;
}

void Particle::update_params(double* g, double best) {
    //std::cout<<"("<<idx<<") distancefromGlobalBest = "<< compute_sigma(std::vector<double>(g, g + n_tolls))<<std::endl;
    update_sigma(g);
    update_fitness(best);
    evaluate_sigma_memb();
    evaluate_fitness_memb();
    update_c_cog();
    update_c_soc();
    update_L();
    update_U();
    update_inertia();
    /*std::cout<<"("<<idx<<") "<<
            " w: "<<w<< " c_soc: "<<
             c_soc<< " c_cog: "
             <<c_cog<<
             " fitness: "<<fitness<<"[ "<<
             fitness_memb[0]<<", "<<
             fitness_memb[1]<<", "<<
             fitness_memb[2]<<" ]"<< 
             " sigma: "<<sigma<<"[ "<<
             sigma_memb[0]<<", "<<
             sigma_memb[1]<<", "<<
             sigma_memb[2]<<" ] L: "<< L<<" U: "<<U<<
             std::endl;*/
}

void Particle::update_pos() {

    for (int i=0; i < n_tolls; i ++) {
        p_past[i] = p[i];
        p[i] = p[i] + v[i];
        if (p[i] >= upper_bounds[i]) {
            p[i]= upper_bounds[i];
            v[i] = -get_rand(0,0.4)*v[i];
            }
        if (p[i] <= 0) {
            p[i]= 0.;
            v[i] = -get_rand(0,0.4)*v[i];
            }
    }
}

void Particle::update_vel(double* g, double* u_bounds) {
    //std::cout<<"velocity = [";
    for(int i=0; i<n_tolls; i++) {
        double extra_param = (fitness_memb[1]+sigma_memb[0])*0.05;
        double r = get_rand(-0.*u_bounds[i],0.5*u_bounds[i]);
        //if (p[i]==u_bounds[i]) r = get_rand(-0.3*u_bounds[i],0);
        v[i] = w*v[i] + c_soc*((g[i] - p[i])) + c_cog*((personal_best[i] - p[i])) + extra_param*r;
        //std::cout<<v[i]<<"("<<U*u_bounds[i]<<") --> ";
        if (std::abs(v[i])>U*u_bounds[i])
            v[i] = U*u_bounds[i]*v[i]/std::abs(v[i]);
        if (std::abs(v[i])<L*u_bounds[i]){
            if (v[i]==0)
                v[i]=L*u_bounds[i];
            else
                v[i]=L*u_bounds[i]*v[i]/std::abs(v[i]);
        }
        //std::cout<<v[i]<<", ";
    }
    //std::cout<<"]"<<std::endl;
}

double Particle::compute_obj_and_update_best(){
    run_cost_past = run_cost;
    run_cost=0;
    int i,j,cheapest_path_idx;
    for(i=0; i<n_commodities; i++) {

        commodity_cost=init_commodity_cost;
        bool found = false;
        //std::cout<<i<<"   "<<commodities_tax_free[i]<<std::endl;
        for(j=0; j< n_tolls; j++) {
            
            toll_cost = p[j] + transfer_costs[i][j];
            if(toll_cost <= commodity_cost) {
                if (toll_cost < commodity_cost) {
                    commodity_cost = toll_cost;
                    cheapest_path_idx = j;
                }
                else {
                    if ( p[j] > p[cheapest_path_idx]) {
                        commodity_cost = toll_cost;
                        cheapest_path_idx = j;
                    }
                }
            }
        }
        if(commodities_tax_free[i] >= commodity_cost) {
            found = true;
            run_cost += p[cheapest_path_idx]*n_users[i];
        }
        /*
        std::string not_free;
        if(found) not_free = "  True ";
        else  not_free = " False ";
        std::cout.precision(17);
        std::cout<<"comm " <<i<<"   p "<<cheapest_path_idx<<"   not free "<<not_free<<"   n users "<<n_users[i] <<"   transf "<< transfer_costs[i][cheapest_path_idx]<<
        "   p "<<p[cheapest_path_idx]<<"   cost "
           <<p[cheapest_path_idx] + transfer_costs[i][cheapest_path_idx] << "   free "<< commodities_tax_free[i]<<"   gain "
           <<p[cheapest_path_idx]*n_users[i]<<std::endl; */

    }


    if(run_cost> personal_best_val){
        for(int i=0; i<n_tolls; i++) personal_best[i] = p[i];
        personal_best_val = run_cost;
    }
    //std::cout<<"("<<idx<<")"<<run_cost<<std::endl;
    return run_cost;
}

void Particle::print() {
    std::cout<<"Transfer"<<std::endl;
    for(int i=0; i<n_commodities; i++) {
        for(int j=0; j<n_tolls; j++) std::cout<<transfer_costs[i][j]<<' ';
        std::cout<<std::endl;
    }
    std::cout<<std::endl<<"upper bounds"<<std::endl<<upper_bounds[0]<<std::endl;
    std::cout<<std::endl<<"n users"<<std::endl<<n_users[0]<<std::endl;
    std::cout<<std::endl<<"p"<<std::endl<<p[0]<<std::endl;
    
}

std::ostream& operator<<( std::ostream &os, Particle& v ) {
    std::cout<<"pos -> "<<v.p[0]<<" vel -> "<<v.v[0];
    return os;
}


