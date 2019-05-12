//  ************************************************************************
//  This file defines common Cuda error functions
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************

#ifndef CUDA_ERROR_H_
#define CUDA_ERROR_H_

#include <cuda.h>

#define CUDA_CHECK(value) cuda_check(__FILE__,__LINE__, #value, value)

inline void cuda_check(const char *file, unsigned line, const char *statement, cudaError_t err){

  if (err != cudaSuccess) {
    std::cerr << "CUDA operation returned error: " << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;

    exit(err);
  }
}


#endif
