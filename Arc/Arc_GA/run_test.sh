g++  -Ofast -fopenmp -W -fPIC -fPIC -I$GUROBI_HOME/include -o test_file  test_genetic.cc  -ljsoncpp -L$GUROBI_HOME/lib -lgurobi_c++ -lgurobi120 -ljsoncpp -lm
# g++  -Ofast -mtune=native -march=native -fopenmp -W -fPIC -o test_file  test_genetic.cc  -ljsoncpp
./test_file

# rm *.o -march=native -Ofast

#g++ -c -fopenmp -fPIC -g arc_ga_bridge.cc -o arc_ga_bridge.o -ljsoncpp
# g++ -g -shared -fopenmp -fPIC -Wl,-soname,arc_ga_bridge.so -o arc_ga_bridge.so arc_ga_bridge.cc -ljsoncpp
# rm -f arc_ga_bridge.o

#valgrind --leak-check=full --leak-check=full --show-leak-kinds=all ./CPP/test_file
