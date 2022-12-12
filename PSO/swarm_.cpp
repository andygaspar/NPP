#include <iostream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include "particle_.cpp"
#include <omp.h>
//#include <Python.h>

/*
Swarm class is basically an ensemble of Particle objects, in particular characterized by:
- particles: Vector object with elements of type Particle
- n_particles: number of particles in the ensemble
- n_dim: dimensionality of the space
- lim_l, lim_h: low and high limit of the space
- w, c_soc, c_cog: PSO parameters
- p_best: coordinates associated with the best position found so far
          in the ensemble's history according to a given objective function
- fp: function pointer to the objective function considered

A Swarm object can be initialized randomly placing the particles in the space
and the methods implemented allow to:
- linearly decrease the w PSO parameter
- update each particle in the swarm a certain number of times iteratively
- output stream operator
*/
class Swarm {
    private:
    std::vector<Particle> particles;
    short n_particles;
    int n_iterations;
    short n_commodities;
    short n_tolls;
    double lim_l=0.0;
    double* lim_h;
    //double w=0.9;
    //double c_soc=1.49445;
    //double c_cog=1.49445;
    double best_val;
    short best_particle_idx;
    double* p_best;
    std::vector<double> p_optimum {8.219140435558018, 6.567166662716659, 6.897422299785148, 7.7799910260801255, 8.478832704876282, 7.64673214911457, 7.89582185652813, 8.300583301169524, 8.656954373882115, 6.177564309048648, 6.2842481618841965, 7.843417304581834, 5.500861990011849, 5.119255635800755, 8.661871678969021, 8.009236008443708, 7.212185572422072, 8.289284009305305, 7.2247392288701695, 5.284718513750903, 5.794744203815469, 8.604893621271145, 7.184956238105855, 8.577689733761673, 6.328490546928464, 7.060834626270889, 6.411353015064239, 7.9156289331762935, 6.467858434512067, 6.331419801957594, 6.067830009826707, 6.950398736121713, 6.789225439719311, 6.817139859156157, 7.066229566569717, 8.06483644041046, 7.827362083923671, 4.555527248105079, 7.3084152387370125, 7.607670443822741, 5.060104609505714, 4.734343873371017, 7.832279389010637, 7.179643718485295, 6.976438711663036, 6.599676744148325, 7.849007343309665, 4.4551262237924565, 4.965151913857021, 8.368464727255933, 6.089194402148229, 8.341942873002584, 5.603268781292286, 6.231242336312446, 6.441775700304419, 7.086036643217843, 6.232111573752952, 5.153134316455862, 5.941804293590593, 6.591112071275887, 5.980953972449242, 6.008101222927706, 4.802075839082619, 6.769233740281692, 5.865224084775718, 6.250286895094973, 4.972727091315306, 6.369801446176325, 4.46073225926213, 6.774151045368626, 6.121515374843284, 4.590127628730301, 6.401563375704882, 5.416106154672764, 4.860549306240642, 4.895347069343842, 6.71717298767075, 3.686448776019006, 5.079182271674796, 4.440769913328069, 5.692133990642276, 6.184046086389742, 6.02790829957587, 4.5279718782834095, 4.8631764352604705, 6.32605527304305, 5.496398215658132, 4.979007265708503, 6.743118304637591, 6.426341708283388, 5.069060405712179, 5.28577071406329, 6.285952308049872, 5.1581568867866, 3.412625737598148, 6.446292538676005, 6.088602328482523, 6.075909652070152, 5.909859379411017, 6.527289207536775, 5.194313336649657, 4.608796749539437, 7.0467465914831, 5.627491241573949, 7.020224737229739, 5.646054126490611, 6.165080858269903, 4.972401526999988, 4.99881434235661, 6.013753901623261, 6.165140797398635, 5.441800653720293, 5.174222846358077, 7.625687030932568, 7.308910434578342, 5.857730382846938, 5.2010177311705945, 7.168521034344877, 4.467107386670904, 4.295194463893125, 7.328861264970982, 5.927587852838059, 6.537289302185172, 5.932413130507451, 7.409857933831759, 5.982983313784366, 5.397466726674153, 7.069300342579498, 6.510059967868926, 7.902793463524699, 5.485039650846204, 6.004066382625414, 4.334378192738029, 4.97863913565152, 5.8527394259787755, 5.252868417367324, 5.873064525154291, 8.324528709728753, 8.007752113374512, 5.226825991493513, 5.749104390144112, 7.867362713141034, 4.885340337581592, 4.9940361426893105, 8.02770294376728, 6.272955583078982, 7.236130980981272, 7.491269784502208, 7.248684637429377, 5.352078922430991, 4.24850429426483, 8.628156996574218, 7.208901646665254, 8.601635142320902, 5.402524328199775, 5.9214510599790255, 5.937510768729849, 5.164755280022689, 6.491803843071267, 5.0409639693925214, 7.4924281539669835, 7.175651557612785, 5.896880061705559, 4.917003834382399, 7.0352621573793215, 5.869797806306451, 4.161935586927569, 7.195602388005426, 5.443298525694075, 6.404030425219531, 6.659169228740467, 7.276599056866203, 6.022132992643492, 5.436616405532789, 7.796056440812492, 6.376801090903339, 7.769534586559143, 4.061000910626376, 5.192130350772004, 5.684042446519442, 3.877905681858067, 5.659703287309512, 7.741517861380572, 7.4247412650263485, 5.334774621923913, 5.532239348973349, 7.284351864792882, 3.9441516257479066, 4.4110252943411865, 7.444692095419015, 5.403467828721659, 6.653120132633148, 6.908258936154027, 7.525688764279767, 5.460027552861359, 4.874510965751146, 8.045146148226056, 6.625890798316902, 8.018624293972707, 3.7670286768553183, 4.455066446548813, 4.665599810540812, 5.309860753454245, 5.9087929947230755, 8.502650378734558, 5.207465248259439, 7.98370353354783, 6.706143729768172, 5.712042609660074, 4.964951640653197, 8.507567683821463, 7.854932013296121, 4.896490840988719, 8.134980014157748, 6.6194029694931285, 4.164503311545564, 4.462071186593408, 8.450589626123588, 5.407083319618096, 3.850015395818181, 6.1741865517809345, 6.906530631123331, 7.117063995115274, 7.761324938028707, 5.4151360130939565, 4.316527217261756, 7.666926937193631, 8.045484382146839, 4.4256493412445135, 5.172157811695087, 8.205824612772943, 6.678140441743358, 7.4142526499870645, 7.818203417803545, 8.286821281633737, 5.623059828636617, 5.32366361231367, 8.806278665580027, 7.387023315671058, 8.779756811326678, 5.8574099554267605, 6.589754034769097, 6.8002873987611, 7.444548341674494, 6.669925512077045, 4.8426734411726216, 5.7752527800300015, 6.28572753543267, 4.376658348518447, 3.997535514057227, 5.786119568805276, 4.5103844831683375, 4.484962819592056, 6.01816017115209, 5.906403104969513, 5.320886517859293, 5.511093535104948, 4.863197074645967, 5.484571680851616, 3.945271022952795, 5.608060079898621, 6.099972175646059, 4.111307743463044, 3.723805058345336, 7.526537536960149, 4.59104767262815, 4.653210966508482, 7.686877767586282, 5.279507693711508, 6.895305804800444, 7.1504446083213224, 7.767874436447048, 4.967926372110059, 4.382309784999845, 8.287331820393337, 6.8680764704842545, 8.260809966139988, 4.496662030840852, 5.0156887626201385, 4.897226209750045, 5.185900618444123, 6.1509786668903565, 6.279830141430637, 4.507785644065507, 8.050401687233801, 7.397766016708431, 5.61774600102072, 7.677814017570057, 8.015996339716622, 4.7705780014949255, 5.183274212080169, 7.993423629535869, 5.590516666704492, 6.983250162360321, 5.717020555193244, 6.449364634535613, 6.659897998527555, 7.304158941441017, 5.240193255491789, 3.1763981305860796, 3.7733707437537, 4.64768739302105, 5.0149618445689725, 4.878783684433557, 5.496213512559308, 6.410980466370148, 5.252023089089802, 4.417026042614969, 5.304260454686741, 4.347302942307435, 4.44984838435343, 4.724165922808426, 4.709349179470053, 4.266753155585178, 4.572938966161786, 5.177075116782049, 4.524439446256707, 3.885727482693966, 4.804487447118305, 5.142669769264888, 4.501911279455925, 3.9163946923457047, 5.268579927498706, 3.8493245775895275, 5.242058073245374, 3.4558719571143968, 3.974898688893717, 3.786571428075831, 4.430832370989293, 3.8235717322470464, 6.698091272135984, 7.4191699550740395, 7.83815424819619, 8.291738586720697, 4.833588752642072, 5.343614442706295, 7.9511809954684205, 7.391940620758078, 8.784674116413953, 5.8773607858194055, 6.609704865161916, 6.820238229153688, 7.464499172067235, 6.67484281716429, 6.7665342845486975, 7.021673088069605, 7.639102916195327, 5.911372499742697, 5.325855912632484, 7.298545324943066, 6.739304950232508, 8.132038445888266, 5.592954436526611, 6.111881168305814, 5.700057824169335, 4.944580667769628, 6.022207146638635, 7.046582285410295, 7.384764607556868, 4.863002728996321, 3.998258673291918, 7.362191897376192, 4.318685590870643, 5.173597273580839, 5.085788823033482, 5.81813290237584, 6.02866626636785, 6.672927209281255, 3.466723309147954, 7.9191509170569425, 4.336576442863077, 4.918970573892437, 8.438608301003256, 7.019352951094078, 8.412086446749896, 5.3409276265543895, 6.073271705896786, 6.283805069888729, 6.928066012802191, 6.30225514750029, 5.457285174866549, 4.984455724809223, 7.374745553824236, 7.357535273240675, 6.896509402085286, 5.958357454680197, 6.690701534022512, 6.901234898014479, 7.545495840927909, 6.640437469646848, 4.936526089537637, 5.434042805448813, 4.014787455539675, 3.5537615843842705, 4.43314720341678, 5.733313010836127, 6.225225106583537, 3.9153526989577756, 4.80084697854943, 5.944068495513378, 4.524813145604224, 5.91754664126006, 3.5685031477123914, 5.1477964237258504, 5.639708519473345, 4.545315497763397, 3.936202922845041, 7.334962563060117, 8.727696058715727, 6.477814838626411, 7.210158917968803, 7.420692281960783, 8.0649532248742, 6.617864759466094, 5.684189752210244, 5.058559488717293, 5.790903568059889, 6.001436932051661, 6.645697874965066, 4.299508950030344, 6.451292984373083, 7.183637063715453, 7.394170427707422, 8.038431370620884, 4.832227470487567, 3.7721809288193526, 5.2575096221774515, 3.8164339976215444, 4.341461685123448, 5.776536353956715, 4.335460729400893, 5.0738057644658205, 4.080997795798567, 5.625209397310073, 5.92860007137125};
    short num_threads;
    double run_cost_best;

    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    public:
    Swarm(double* comm_tax_free, int* n_usr, double* transf_costs, double* u_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th);
    //Swarm() {n_particles=1; n_dim=2; particles=Vector<Particle>{n_particles}; p_best=particles[0].pos(); num_threads=2;}
    void set_init_sols(double* solutions, int n_solutions);
    double get_best_val() {return best_val;}
    double * get_best() {return p_best;}
    void run();
    double compute_obj() {return particles[0].compute_obj_and_update_best();}
    double compute_distance() {return particles[0].compute_sigma(p_optimum);}

    //void update_w() {w = w - 0.5/(n_iterations);}
    int size() {return n_particles;}
    void print();
    void print_particles();
    void save_particles();
};

void Swarm::save_particles() {
    std::string file_name = "data.csv";
    //std::remove(file_name);
    std::ofstream outfile;
    outfile.open(file_name);
    for (int i=0;i<n_particles;++i) {
        for (int j=0;j<2;++j) {
            outfile<<particles[i].p[j];
            if (i<2-1)
                outfile<<",";
            else
                outfile<<std::endl;
        }
    }
    outfile.close();
}

Swarm::Swarm(double* comm_tax_free, int* n_usr,double* transf_costs, double* u_bounds, short n_comm, short n_to, short n_parts, int n_iter, short num_th) {
    n_iterations = n_iter;
    n_particles=n_parts; 
    n_tolls=n_to;
    n_commodities=n_comm;
    lim_h = u_bounds;
    //w = 0.9;
    num_threads=num_th;
    best_val = 0;

    p_best = new double[n_tolls];
    particles=std::vector<Particle>(n_particles);

    double d_M = 0;
    for (int i=0;i<n_tolls;++i)
        d_M += std::pow(u_bounds[i],2);
    d_M = std::sqrt(d_M);

    for(int i=0;i<n_particles;++i){
        particles[i] = Particle{comm_tax_free, n_usr ,transf_costs, 
        u_bounds, n_commodities, n_tolls, i, d_M};
    }

    double tmp = 0; 
    run_cost_best = 0;
    for(int i=0;i<n_particles;++i) {
        tmp = particles[i].compute_obj_and_update_best();
        if (tmp>run_cost_best)
            run_cost_best = tmp;
            best_particle_idx = i;
    }

    for(int i=0; i< n_tolls; i++) p_best[i]=particles[best_particle_idx].p[i];

}



void Swarm::run(){
    std::vector<double> run_results(n_particles);
    int i;
    bool new_best = false;

    for(int iter=0; iter< n_iterations; iter++) {

        if(iter%100000==0) {

            //this->save_particles();

            std::cout<<"("<<best_particle_idx<<") "<<best_val<<
            //" pos= "<<particles[best_particle_idx].personal_best[0]<<
            "("<<particles[best_particle_idx].run_cost<<
            //" pos= "<< particles[best_particle_idx].p[0]<<
            ")"<<"  iter "<< iter<< 
            " w: "<<particles[best_particle_idx].w<< " c_soc: "<<
             particles[best_particle_idx].c_soc<< " c_cog: "
             <<particles[best_particle_idx].c_cog<<
             " fitness: "<<particles[best_particle_idx].fitness<<"[ "<<
             particles[best_particle_idx].fitness_memb[0]<<", "<<
             particles[best_particle_idx].fitness_memb[1]<<", "<<
             particles[best_particle_idx].fitness_memb[2]<<" ]"<< 
             " sigma: "<<particles[best_particle_idx].sigma<<"[ "<<
             particles[best_particle_idx].sigma_memb[0]<<", "<<
             particles[best_particle_idx].sigma_memb[1]<<", "<<
             particles[best_particle_idx].sigma_memb[2]<<" ]"<<
             " U: "<<"("<<particles[best_particle_idx].U<<")"<<
             particles[best_particle_idx].U*lim_h[0]<<
             " v_last: "<<particles[best_particle_idx].v[0]<<
             " L: "<<"("<<particles[best_particle_idx].L<<")"<<
             particles[best_particle_idx].L*lim_h[0]<<
             " real distance: "<<particles[best_particle_idx].personal_best[0]-p_optimum[0]<<std::endl;

            /*char filename[] = "plot.py";
            FILE* fp;

            Py_Initialize();

            fp = _Py_fopen(filename, "r");
            PyRun_SimpleFile(fp, filename);

            Py_Finalize();*/
        }


        #pragma omp parallel for num_threads(this->num_threads) shared(run_results, particles) //reduction(max : run_result)//implicit(none) private(i) shared(run_results, n_particles, particles)
        for(i=0;i<n_particles;++i) {
            //std::cout<<"["<<best_particle_idx<<"]";
            particles[i].update_params(p_best, run_cost_best);
            particles[i].update_vel(p_best, lim_h);
            particles[i].update_pos();
            run_results[i] = particles[i].compute_obj_and_update_best();
            //std::cout<<"("<<i<<")"<<run_results[i]<<std::endl;
        }

        for(i=0;i<n_particles;++i){
            if(run_results[i] > best_val) {
                new_best = true;
                best_val = run_results[i];
                best_particle_idx = i;
            }
        }
        if(new_best) {
            for(int i=0; i< n_tolls; i++) p_best[i]=particles[best_particle_idx].p[i];
            new_best = false;
        }
        /*#pragma omp parallel for num_threads(this->num_threads) shared(p_best, run_cost_worst)
        for(i=0;i<n_particles;++i) {
            std::cout<<"["<<best_particle_idx<<"]";
            particles[i].update_params(p_best, run_cost_worst);
            particles[i].update_pos();
            particles[i].update_vel(p_best, lim_h);
        }*/
        //this->update_w();

        
    }
}


void Swarm::set_init_sols(double* solutions, int n_solutions) {
    std::cout<<"   init solutions "<< n_solutions<<"  tolls"<<n_tolls<< std::endl;
    int n_sols = n_solutions;
    if(n_sols > n_particles) n_sols = n_particles;

    for(int i=0; i< n_sols; i ++) {
        for(int j=0; j< n_tolls; j++) particles[i].p[j] = solutions[i*n_tolls + j];
    }

    for(int j=0; j< n_tolls; j++) std::cout<<particles[0].p[j]<< " ";
    std::cout<< std::endl; 
    std::cout<<particles[0].compute_obj_and_update_best()<<std::endl;
    std::cout<< "  ++++++  " <<particles[0].personal_best_val<< std::endl;
    for(int i=0; i< n_tolls; i++) p_best[i]=particles[0].p[i];
}





std::ostream& operator<<( std::ostream &os, Swarm& s ) {
    std::cout<<"best pos -> "<<s.p_best<< " best obj -> "<<std::endl;
    for(int i=0; i<s.n_particles; ++i) {
        std::cout<<s.particles[i]<<"  obj -> "<< " best obj -> ";
        if (i<(s.n_particles-1))
            std::cout<<std::endl;
    }
    return os;
}

void Swarm::print() {
    std::cout<<*this<<std::endl;
}

void Swarm::print_particles(){
    for(int i=0; i< n_particles; i++ ) particles[i].print();
}
