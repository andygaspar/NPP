#g ++ -g -fopenmp -W -fPIC -o test_file  test_genetic.cc  -ljsoncpp
# ./test_file

#g++ -c -fopenmp -fPIC -g arc_ga_bridge.cc -o arc_ga_bridge.o -ljsoncpp
g++ -g -shared -fopenmp -fPIC -Wl,-soname,arc_ga_bridge.so -o arc_ga_bridge.so arc_ga_bridge.cc -ljsoncpp
rm -f arc_ga_bridge.o
#valgrind --leak-check=full --leak-check=full --show-leak-kinds=all ./CPP/test_file
