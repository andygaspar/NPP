g++  -c -fopenmp -fPIC bridge_.cpp -o bridge.o
g++ -shared -fopenmp -Wl,-soname,bridge.so -o bridge.so bridge.o
rm -f PSO/bridge.o
