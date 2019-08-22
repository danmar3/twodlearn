//************************************************************************
//***********************************************************************

#ifndef EIGEN_CUDA_H_
#define EIGEN_CUDA_H_

#include <stdlib.h>
#include <cuda.h>
#include <functional>
#include "cuda_error.h"
#include "Eigen/Dense"
#include "twodlearn/common/cuda/twin_object.h"



template<typename T, int eig_opt = Eigen::ColMajor>
  class TwinMat : public TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>, T>{
 public:

 /* ---------- Attributes ----------- */

 // mat: eigen matrix
 Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt> local_mat;
 Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>> mat;

 // get_data_eiten: function pointer that returns the pointer where mat data is stored
 //double* (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::*get_data_eigen)(void); // origin
 double* (Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>>::*get_data_eigen)(void); // origin

 /* ---------- Constructors ----------- */
 TwinMat():
 TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>, T>(),
 mat(NULL, Eigen::Dynamic, Eigen::Dynamic)
 {};

 TwinMat(int n_rows, int n_cols):
 TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>, T>(),
 mat(NULL, Eigen::Dynamic, Eigen::Dynamic)
 {
   // create the matrix object
   local_mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::Random(n_rows, n_cols);

   new (&mat) Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt> >(&local_mat(0,0), local_mat.rows(), local_mat.cols());
   //mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::Zero(n_rows, n_cols);
   //mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::Ones(n_rows, n_cols);

   // set the function to get the pointer to the data on the matrix class
   //get_data_eigen = &Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::data; //original
   get_data_eigen = &Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>>::data;

   // call TwinObject initialization
   this->initialize(&local_mat, mat.size()*sizeof(T), std::bind(get_data_eigen, &mat ));
 }

 /* ------------ Operators ------------ */
 void operator=( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>& mat_in){
   // free previous cuda memory
   this->free_device();

   // update mat
   //mat = mat_in;
   new (&mat) Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt> >(&mat_in(0,0), mat_in.rows(), mat_in.cols());

   // set the function to get the pointer to the data on the matrix class
   //get_data_eigen = &Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::data;
   get_data_eigen = &Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>>::data;

   // update parent class attributes
   this->initialize(&mat_in, mat.size()*sizeof(T), std::bind(get_data_eigen, &mat ));

 }




};


/*
template<typename T >
class TwinMat : public TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, T>{
 public:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;

  double* (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::*get_data_eigen)(void);

 TwinMat(int n_rows, int n_cols):
  TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, T>(){
    // create the matrix object
    mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(n_rows, n_cols);
    //mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_rows, n_cols);

    // set the function to get the pointer to the data on the matrix class
    get_data_eigen = &Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::data;

    // call TwinObject initialization
    this->initialize(&mat, mat.size()*sizeof(T), std::bind(get_data_eigen, &mat ));
  }
};
*/

#endif //
