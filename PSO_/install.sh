g++  -c -W -fopenmp -fPIC bridge.cc
g++ -shared -fopenmp -Wl,-soname,bridge.so -o bridge.so bridge.o
rm -f bridge.o
