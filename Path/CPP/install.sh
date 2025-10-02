#g++ -c -Ofast -fopenmp -fPIC Path/CPP/GAPSO/gapso_bridge.cc -o Path/CPP/GAPSO/gapso_bridge.o -ljsoncpp
#g++ -shared -fopenmp -Wl,-soname,gapso_bridge.so -o Path/CPP/libs/gapso_bridge.so Path/CPP/GAPSO/gapso_bridge.o -ljsoncpp
#rm -f Path/CPP/GAPSO/gapso_bridge.o

g++ -c -Ofast -fopenmp -fPIC Path/CPP/GA/ga_bridge.cc -o Path/CPP/GA/ga_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,ga_bridge.so -o Path/CPP/libs/ga_bridge.so Path/CPP/GA/ga_bridge.o -ljsoncpp
rm -f Path/CPP/GA/ga_bridge.o

g++ -c -Ofast -fopenmp -fPIC Path/CPP/GAH/ga_h_bridge.cc -o Path/CPP/GAH/ga_h_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,ga_h_bridge.so -o Path/CPP/libs/ga_h_bridge.so Path/CPP/GAH/ga_h_bridge.o -ljsoncpp
rm -f Path/CPP/GAH/ga_h_bridge.o

#g++ -c -Ofast -fopenmp -fPIC Path/CPP/PSO/pso_bridge.cc -o Path/CPP/PSO/pso_bridge.o -ljsoncpp
#g++ -shared -fopenmp -Wl,-soname,pso_bridge.so -o Path/CPP/libs/pso_bridge.so Path/CPP/PSO/pso_bridge.o -ljsoncpp
#rm -f Path/CPP/PSO/pso_bridge.o
