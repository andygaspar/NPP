g++ -c -Ofast -fopenmp -fPIC PSO/bridge.cc -o PSO/bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,bridge.so -o PSO/bridge.so PSO/bridge.o -ljsoncpp
rm -f PSO/bridge.o
