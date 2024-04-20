g++ -g -fopenmp -fPIC -o test_file  test_genetic.cc  -ljsoncpp
./test_file
#valgrind --leak-check=full --leak-check=full --show-leak-kinds=all ./CPP/test_file
