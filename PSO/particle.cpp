#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "utils.cpp"


class Particle {
    private:
    int ndim;
    Vector<float> p;
    Vector<float> v;
    Vector<float> p_best;
    friend std::ostream& operator<<( std::ostream &os, Particle& v );

    public:
    Particle(int n, Vector<float> pos, Vector<float> ve) {ndim = n; p=pos; v=ve; p_best=pos;}
    Particle(int n, Vector<float> pos, Vector<float> ve, float ll, float lh) {ndim = n; p=pos; v=ve; p_best=pos;}
    Particle(int n,float ll, float lh) 
    {ndim=n; p =Vector<float>{ndim,ll,lh}; v=std::vector<float>(ndim); p_best=p;}
    Particle(int n) {ndim=n; p =Vector<float>{ndim}; v=std::vector<float>(ndim); p_best=p;}
    Particle() {ndim=2; p =Vector<float>{ndim}; v=std::vector<float>(ndim); p_best=p;}
    void update_pos() {p = p + v;}
    void update_vel(float w, float c_soc, float c_cog, Vector<float> g);
    void update_best() {p_best=p;}
    Vector<float> p_b() {return p_best;}
    Vector<float> pos() {return p;}
};

void Particle::update_vel(float w, float c_soc, float c_cog, Vector<float> g) {
    Vector<float> r1{p.size(), 0.0, 1.0};
    Vector<float> r2{p.size(), 0.0, 1.0};
    v = w*v + c_soc*(r1*(g - p)) + c_cog*(r2*(p_best - p));
}

std::ostream& operator<<( std::ostream &os, Particle& v ) {
    std::cout<<"pos -> "<<v.p<<" vel -> "<<v.v;
    return os;
}


