#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "utils.cpp"

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
    private:
    int ndim;
    Vector<double> p;
    Vector<double> v;
    Vector<double> personal_best;
    double personal_best_val;
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    public:
    Particle(double* array, int n) {
        std::vector<double> array_(array, array + n*sizeof(array)/sizeof(array[0]));
        p = Vector<double> {array_};
        v= Vector<double> {n};
        personal_best = p;
        ndim = n;
        personal_best_val = 0;
        }
    Particle(int n, Vector<double> pos, Vector<double> ve)
    {ndim = n; p=pos; v=ve; personal_best=pos; personal_best_val = 0;}
    Particle(int n,double ll, double lh)
    {ndim=n; p =Vector<double>{ndim,ll,lh}; v=Vector<double>{ndim}; personal_best=p;}
    Particle(int n) {ndim=n; p =Vector<double>{ndim}; v=Vector<double>{ndim}; personal_best=p;}
    Particle() {ndim=2; p =Vector<double>{ndim}; v=Vector<double>{ndim}; personal_best=p;}
    ~Particle() {}
    double get_personal_best_val() {return personal_best_val;}
    void update_pos();
    void reflection();
    void update_vel(double w, double c_soc, double c_cog, Vector<double> g);
    void update_best(double new_personal_best_val) {personal_best_val = new_personal_best_val; personal_best=p;}
    Vector<double> p_b() {return personal_best;}
    Vector<double> pos() {return p;}
};

void Particle::update_pos() {
    //    p = p + v;
    for (int i=0; i < p.size(); i ++) p[i] = p[i] + v[i];
    if (p >= 1. or p <= 0.){
        this->reflection();
    }

}

void Particle::reflection() {
    for(int i=0;i<ndim;++i) {
        if(p[i]<=0) {
            p[i]=0;
            v[i]=-v[i];
            }
        if(p[i]>=1) {
            p[i]=1;
            v[i]=-v[i];
        }
    }
}

void Particle::update_vel(double w, double c_soc, double c_cog, Vector<double> g) {
    Vector<double> r1{p.size(), 0.0, 1.0};
    Vector<double> r2{p.size(), 0.0, 1.0};
    v = w*v + c_soc*(r1*(g - p)) + c_cog*(r2*(personal_best - p));
}

std::ostream& operator<<( std::ostream &os, Particle& v ) {
    std::cout<<"pos -> "<<v.p<<" vel -> "<<v.v;
    return os;
}


