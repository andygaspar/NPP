#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>


using string = std::string;

struct dimensions{
    int rows;
    int cols;
};

class FileReader{
    public:
    double* commodities_tax_free;
    short n_commodities;
    short n_tolls;

    dimensions comm_t_free_dims;
    double* upper_bounds;
    dimensions ub_dims;
    double* transfer_costs;
    dimensions tra__dims;
    int* n_users;
    dimensions n_users_dims;

    FileReader(string folder_name){
        comm_t_free_dims = get_dims(folder_name + "/commodities_tax_free.csv");
        std::cout<<comm_t_free_dims.rows<<' '<<comm_t_free_dims.cols<<std::endl;
        commodities_tax_free = fill_array(folder_name + "/commodities_tax_free.csv", comm_t_free_dims);
        print_array(commodities_tax_free, comm_t_free_dims);

        n_commodities=comm_t_free_dims.rows;

        ub_dims = get_dims(folder_name + "/upper_bounds.csv");
        std::cout<<ub_dims.rows<<' '<<ub_dims.cols<<std::endl;
        upper_bounds = fill_array(folder_name + "/upper_bounds.csv", ub_dims);
        print_array(upper_bounds, ub_dims);
        n_tolls = ub_dims.rows;

        n_users_dims = get_dims(folder_name + "/n_users.csv");
        std::cout<<n_users_dims.rows<<' '<<n_users_dims.cols<<std::endl;
        n_users = fill_array_int(folder_name + "/n_users.csv", n_users_dims);
        print_array(n_users, n_users_dims);

        tra__dims = get_dims(folder_name + "/transfer_costs.csv");
        std::cout<<tra__dims.rows<<' '<<tra__dims.cols<<std::endl;
        transfer_costs = fill_mat(folder_name + "/transfer_costs.csv", tra__dims);
        print_mat(transfer_costs, tra__dims);
    }

    ~FileReader(){
        delete [] commodities_tax_free;
        delete [] upper_bounds;
        delete [] transfer_costs;
        delete [] n_users;
    }
            
    dimensions get_dims(string file_name){
        std::fstream fin;

        int rows=0;
        int cols=0;
        std::fstream file(file_name);
        std::string line;
        while (getline(file, line)){
            std::stringstream ss( line );                     
            std::string data;
            cols=0;
            while (getline( ss, data, ',' ) ) cols++; 
            rows++;
        }
    return dimensions{rows, cols};
    }

    double* fill_array(string file_name, dimensions dims) {
        double* array = new double[dims.rows];
        std::fstream file(file_name);
        std::string line;
        int row=0;
        while (getline(file, line)){
            std::stringstream ss( line );                     
            std::string data;
            while (getline( ss, data, ',' ) )           
            {
                array[row]=stod(data);
            }
            row++;
        }
        return array;
    }

    int* fill_array_int(string file_name, dimensions dims) {
        int* array = new int[dims.rows];
        std::fstream file(file_name);
        std::string line;
        int row=0;
        while (getline(file, line)){
            std::stringstream ss( line );                     
            std::string data;
            while (getline( ss, data, ',' ) )           
            {
                array[row]=stoi(data);
            }
            row++;
        }
        return array;
    }

    double* fill_mat(string file_name, dimensions dims) {
        double* array = new double[dims.rows* dims.cols];
        std::fstream file(file_name);
        std::string line;
        int row=0;
        while (getline(file, line)){
            std::stringstream ss( line );                     
            std::string data;
            int col = 0;
            while (getline( ss, data, ',' ) )           
            {
                array[row*dims.cols + col]=stod(data);
                col++;
            }
            row++;
        }
        return array;
    }

    
    template<typename T>
    void print_array(T* array, dimensions dims) {
        for(int i=0; i < dims.rows; i++) std::cout<<array[i]<<' ';
        std::cout<<std::endl;
    }

    void print_mat(double* mat, dimensions dims) {
    for(int i=0; i < dims.rows; i++) {
        for(int j=0; j< dims.cols; j++) std::cout<<mat[i*dims.cols + j]<<' ';
    std::cout<<std::endl;
    }
    }


};