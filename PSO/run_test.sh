g++ -g -fopenmp -fPIC -o PSO/test_file  PSO/test_genetic.cc  -ljsoncpp
#./PSO/test_file
#valgrind --leak-check=full --leak-check=full --show-leak-kinds=all ./PSO/test_file
