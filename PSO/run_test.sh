g++ -g -fopenmp -fPIC -o PSO/test_file  PSO/test_from_file.cc  -ljsoncpp

valgrind --leak-check=full --leak-check=full --show-leak-kinds=all ./PSO/test_file
