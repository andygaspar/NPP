#include "Swarm.cpp"
#include <chrono>

int main() {
    clock_t start, end;
    Swarm s{10,2, &obj};
    start=clock();
    s.update(100);
    end=clock();
    std::cout<<s<<std::endl<<std::endl;
    std::cout<<"time: "<<((float) end - start)/CLOCKS_PER_SEC<<" s"<<std::endl;
}