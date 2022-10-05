#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>

template <typename T> class Vector;
template <typename T> Vector<T> operator+( Vector<T> lhs, Vector<T> rhs);
template <typename T> Vector<T> operator-(Vector<T> lhs, Vector<T> rhs);
template <typename T> Vector<T> operator*( Vector<T> lhs, Vector<T> rhs);
template <typename T> Vector<T> operator*( Vector<T> lhs, T rhs);
template <typename T> Vector<T> operator*( T rhs, Vector<T> lhs);
template <typename T> std::ostream& operator<<( std::ostream &os, Vector<T>& v );

/*
Vector is a templated class implementing a vector object and allowing:
- binary operators of sum, difference and Hadamard product
- product with a scalar
- push_back and pop_back methods
- random values vector generation
- output stream operator
*/
template<typename T>
class Vector {
    private:
    std::vector<T> v;
    friend Vector<T> operator +<T>( Vector<T> lhs, Vector<T> rhs);
    friend Vector<T> operator -<T>(Vector<T> lhs, Vector<T> rhs); 
    friend Vector<T> operator *<T>( Vector<T> lhs, Vector<T> rhs);
    friend Vector<T> operator *<T>( Vector<T> lhs, T rhs);
    friend Vector<T> operator *<T>( T rhs, Vector<T> lhs);
    friend std::ostream& operator<<<T>( std::ostream &os, Vector<T>& v );

    public:
    Vector() {v={};}
    Vector(std::vector<T> vec) {v = vec;}
    Vector(int ndim) {v = std::vector<T>(ndim);}
    Vector(int ndim, T i, T f);
    Vector(const Vector<T>& vec) {v = vec.v;}
    Vector<T>& operator = (const Vector<T>& vec) {v = vec.v; return *this;}
    Vector<T>& operator = (const std::vector<T>& vec) {v = vec; return *this;}
    int size() const {return v.size();}
    T& operator [] (int n) {return v[n];}
    void push_back(T val) {v.push_back(val);}
    void pop_back() {v.pop_back();}
    std::vector<float>::iterator begin() {return v.begin();}
    std::vector<float>::iterator end() {return v.end();}
    void random(int ndim, T i, T f);
};

//random vector initialization
template<typename T>
Vector<T>::Vector(int ndim, T i, T f) {
    v=std::vector<T>(ndim);
    std::default_random_engine generator(std::rand());
    std::uniform_real_distribution<T> distribution(i,f);
    for(int j=0;j<ndim;++j)
        v[j] = distribution(generator);
}

template<typename T>
Vector<T> operator + ( Vector<T> lhs, Vector<T> rhs) {
    Vector<T> c = lhs;
    for (int i=0; i<lhs.size(); ++i) 
        c.v[i] =lhs.v[i] + rhs.v[i];
    return c;
}

template<typename T>
Vector<T> operator - ( Vector<T> lhs, Vector<T> rhs) {
    Vector<T> c = lhs;
    for (int i=0; i<lhs.size(); ++i) 
        c.v[i] =lhs.v[i] - rhs.v[i];
    return c;
}

template<typename T>
Vector<T> operator * ( Vector<T> lhs, Vector<T> rhs) {
    Vector<T> c = lhs;
    for (int i=0; i<lhs.size(); ++i) 
        c.v[i] =lhs.v[i] * rhs.v[i];
    return c;
}

template<typename T>
Vector<T> operator * ( Vector<T> lhs, T rhs) {
    Vector<T> c = lhs;
    for (int i=0; i<lhs.size(); ++i) 
        c.v[i] =lhs.v[i] * rhs;
    return c;
}

template<typename T>
Vector<T> operator * ( T rhs, Vector<T> lhs) {
    Vector<T> c = lhs;
    for (int i=0; i<lhs.size(); ++i) 
        c.v[i] =lhs.v[i] * rhs;
    return c;
}

template<typename T>
std::ostream& operator<<( std::ostream &os, Vector<T>& v ) {
    std::cout<<"["<<v.size()<<"] ";
    std::cout<<"<";
    for (int i=0; i<v.size(); ++i) {
        std::cout<<v[i];
        if (i<(v.size()-1))
            std::cout<<", ";
    }
    std::cout<<">";
    return os;
}

template<typename T>
void Vector<T>::random(int ndim, T i, T f) {
    *this = Vector<T>{ndim};
    std::default_random_engine generator(std::rand());
    std::uniform_real_distribution<T> distribution(i,f);
    for(int j=0;j<ndim;++j)
        this->v[j] = distribution(generator);
}

/*
example of objective function
*/
float obj(Vector<float> v) {
    return -(v[0]*v[0])-(v[1]*v[1]);
}
