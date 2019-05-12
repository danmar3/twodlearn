//  ***********************************************************************
//  Cuda support for Eigen matrices
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************


#ifndef EIGEN_CUDA_H_
#define EIGEN_CUDA_H_

#include <stdlib.h>
#include <cuda.h>
#include <functional>
#include "Eigen/Dense"
#include "twodlearn/core/cuda/cuda_error.cu.h"
#include "twodlearn/core/cuda/twin_object.cu.h"



template<typename T, int eig_opt = Eigen::ColMajor>
class TwinMat :
  public TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>,
                    T>{
public:

  /* ---------- Attributes ----------- */

  // mat: eigen matrix
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt> mat;

  // get_data_eiten: function pointer that returns the pointer
  //  where mat data is stored
  double* (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>
                ::*get_data_eigen)(void);

  /* ---------- Constructors ----------- */
  TwinMat():
  TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>, T>()
  {};

  TwinMat(int n_rows, int n_cols):
  TwinObject<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>, T>()
  {
    // create the matrix object
    mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>
               ::Random(n_rows, n_cols);
    //mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::Zero(n_rows, n_cols);
    //mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>::Ones(n_rows, n_cols);

    // set the function to get the pointer to the data on the matrix class
    get_data_eigen = &Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>
                           ::data;

    // call TwinObject initialization
    this->initialize(&mat, mat.size()*sizeof(T),
                     std::bind(get_data_eigen, &mat ));
  }

  /* ------------ Operators ------------ */
  void operator=( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>& mat_in){
    // free previous cuda memory
    this->free_device();
    // update mat
    mat = mat_in;
    // set the function to get the pointer to the data on the matrix class
    get_data_eigen = &Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, eig_opt>
                           ::data;
    // update parent class attributes
    this->initialize(&mat, mat.size()*sizeof(T),
                     std::bind(get_data_eigen, &mat ));

  }
};

#endif //
