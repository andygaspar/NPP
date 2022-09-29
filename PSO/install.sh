g++  -c -fopenmp -fPIC bridge.cpp -o bridge.o
g++ -shared -fopenmp -Wl,-soname,bridge.so -o bridge.so bridge.o