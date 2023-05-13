g++  -c -fopenmp -fPIC PSO_/bridge.cc -o PSO_/bridge.o
g++ -shared -fopenmp -Wl,-soname,PSO_/bridge.so -o PSO_/bridge.so PSO_/bridge.o
rm -f PSO_/bridge.o
