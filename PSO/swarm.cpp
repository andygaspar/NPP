#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "particle.cpp"
#include <omp.h>


//forse c'è da specificare il numero di cpu? omp.get_cpu_count()

class Swarm {
    private:
    int n_particles;
    int n_dim;
    float lim_l=0.0;
    float lim_h=100.0;
    float w=0.9;
    float c_soc=1.49445;
    float c_cog=1.49445;
    float (*fp)(Vector<float>);
    Vector<Particle> particles;
    Vector<float> p_best;
    friend std::ostream& operator<<( std::ostream &os, Swarm& s );

    public:
    Swarm(int n, int n_, float (*f)(Vector<float>), float ll, float lh);
    Swarm(int n, int n_, float (*f)(Vector<float>));
    Swarm() {n_particles=1; n_dim=2; particles=Vector<Particle>{n_particles}; p_best=particles[0].pos();}
    ~Swarm() {}
    int size() {return n_particles;}
    Vector<float> position(int i) {return particles[i].pos();}
    void set_ndim(int n) {n_dim=n; *this=Swarm{n_particles,n,fp};}
    void set_nparticles(int n) {n_particles=n; *this=Swarm{n,n_dim,fp};}
    void set_lim(float ll, float lh) {lim_l=ll; lim_h=lh; *this=Swarm{n_particles,n_dim,fp,ll,lh};}
    void set_w(float val) {w=val;}
    void set_objective_function(float (*f)(Vector<float>)) {fp = f;}
    void test(Particle p);
    void update_best(Vector<float> pb) {p_best=pb;}
    void update_w(int n_iter) {w = w - 0.5/n_iter;}
    void update_swarm();
    void update(int num);
    void print();
};

Swarm::Swarm(int n, int n_, float (*f)(Vector<float>), float ll, float lh) {
    n_particles=n; 
    n_dim=n_; 
    particles=Vector<Particle>{n_particles}; 
    for(int i=0;i<n_particles;++i)
        particles[i] = Particle{n_dim,ll,lh};
    p_best=particles[0].pos(); 
    fp = f;
}

Swarm::Swarm(int n, int n_, float (*f)(Vector<float>)) {
    n_particles=n; 
    n_dim=n_; 
    particles=Vector<Particle>{n_particles}; 
    for(int i=0;i<n_particles;++i)
        particles[i] = Particle{n_dim,lim_l,lim_h};
    p_best=particles[0].pos(); 
    fp = f;
}

void Swarm::update_swarm() {
    #pragma omp parallel for  
    for(int i=0;i<n_particles;++i) {
        particles[i].update_pos();
        particles[i].update_vel(w,c_soc,c_cog,p_best);
        if (fp(particles[i].pos()) > fp(particles[i].p_b()))
            particles[i].update_best();
        if (fp(particles[i].pos()) > fp(p_best))
            this->update_best(particles[i].pos());
    }
}

void Swarm::update(int num) {
    for(int i=0;i<num;++i) {
        //std::cout<<omp_get_thread_num()<<"/"<<omp_get_num_threads()<<std::endl;
        this->update_swarm();
        this->update_w(num);
        //std::cout<<*this<<std::endl<<std::endl;
    }
}

std::ostream& operator<<( std::ostream &os, Swarm& s ) {
    std::cout<<"best pos -> "<<s.p_best<< " best obj -> "<<s.fp(s.p_best)<<std::endl;
    for(int i=0; i<s.n_particles; ++i) {
        std::cout<<s.particles[i]<<"  obj -> "<<s.fp(s.particles[i].pos())<< " best obj -> "<<s.fp(s.particles[i].p_b());
        if (i<(s.n_particles-1))
            std::cout<<std::endl;
    }
    return os;
}

void Swarm::print() {
    std::cout<<*this<<std::endl;
}
