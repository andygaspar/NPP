g++ -c -fopenmp -fPIC PSO_/bridge.cc -o PSO_/bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,bridge.so -o PSO_/bridge.so PSO_/bridge.o -ljsoncpp
rm -f PSO_/bridge.o
