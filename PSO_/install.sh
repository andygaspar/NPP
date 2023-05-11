g++  -c -g -fopenmp -fPIC PSO/bridge_.cpp -o PSO/bridge.o
g++ -shared -fopenmp -Wl,-soname,PSO/bridge.so -o PSO/bridge.so PSO/bridge.o
rm -f PSO/bridge.o
