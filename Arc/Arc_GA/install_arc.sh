# g++ -c -Ofast -fopenmp -fPIC CPP/GAPSO/gapso_bridge.cc -o CPP/GAPSO/gapso_bridge.o -ljsoncpp
# g++ -shared -fopenmp -Wl,-soname,gapso_bridge.so -o CPP/libs/gapso_bridge.so CPP/GAPSO/gapso_bridge.o -ljsoncpp
# rm -f CPP/GAPSO/gapso_bridge.o

# g++ -c -Ofast -fopenmp -fPIC CPP/GA/ga_bridge.cc -o CPP/GA/ga_bridge.o -ljsoncpp
# g++ -shared -fopenmp -Wl,-soname,ga_bridge.so -o CPP/libs/ga_bridge.so CPP/GA/ga_bridge.o -ljsoncpp
# rm -f CPP/GA/ga_bridge.o

# g++ -c -Ofast -fopenmp -fPIC CPP/GA/ga_operators_bridge.cc -o CPP/GA/ga_operators_bridge.o -ljsoncpp
# g++ -shared -fopenmp -Wl,-soname,ga_operators_bridge.so -o CPP/libs/ga_operators_bridge.so CPP/GA/ga_operators_bridge.o -ljsoncpp
# rm -f CPP/GA/ga_operators_bridge.o

# g++ -c -Ofast -fopenmp -fPIC -I$GUROBI_HOME/include Arc/Arc_GA/arc_ga_bridge.cc -o Arc/Arc_GA/arc_ga_bridge.o -ljsoncpp
# g++ -shared -fopenmp -Wl,-soname,Arc/Arc_GA/arc_ga_bridge.so -o Arc/Arc_GA/arc_ga_bridge.so Arc/Arc_GA/arc_ga_bridge.o -ljsoncpp -L$GUROBI_HOME/lib -lgurobi_c++ -lgurobi120 -ljsoncpp -lm
# rm -f Arc/Arc_GA/arc_ga_bridge.o

# g++ -c -Ofast -fopenmp -fPIC -I$GUROBI_HOME/include Arc/Arc_GA/arc_ga_bridge.cc -o Arc/Arc_GA/arc_ga_bridge.o
# g++ -shared -fopenmp -Wl,-soname,Arc/Arc_GA/arc_ga_bridge.so -o Arc/Arc_GA/arc_ga_bridge.so Arc/Arc_GA/arc_ga_bridge.o -L$GUROBI_HOME/lib -lgurobi_c++ -lgurobi120 -ljsoncpp -lm

g++ -c -Ofast -fopenmp -fPIC -I$GUROBI_HOME/include -g Arc/Arc_GA/arc_ga_bridge.cc -o Arc/Arc_GA/arc_ga_bridge.o
g++ -shared -fopenmp -Wl,-rpath,/opt/gurobi1203/linux64/lib -Wl,-soname,Arc/Arc_GA/arc_ga_bridge.so -o Arc/Arc_GA/arc_ga_bridge.so Arc/Arc_GA/arc_ga_bridge.o -L$GUROBI_HOME/lib -lgurobi_c++ -lgurobi120 -ljsoncpp -lm


rm -f Arc/Arc_GA/arc_ga_bridge.o
# g++ -c -Ofast -fopenmp -fPIC CPP/PSO/pso_bridge.cc -o CPP/PSO/pso_bridge.o -ljsoncpp
# g++ -shared -fopenmp -Wl,-soname,pso_bridge.so -o CPP/libs/pso_bridge.so CPP/PSO/pso_bridge.o -ljsoncpp
# rm -f CPP/PSO/pso_bridge.o
