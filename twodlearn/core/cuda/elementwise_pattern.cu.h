//  ***********************************************************************
//  GPU implementation of element-wise pattern operations
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************
#ifndef ELEMENTWISE_PATTERN_CU_H_
#define ELEMENTWISE_PATTERN_CU_H_

#include <cuda.h>

/* device functors definition*/
template<typename T>
struct PowFunc{
  const T a;

  PowFunc(T _a) : a(_a){}

  __host__ __device__
  float operator() (float x){
    return powf(x, a);
  }

  __host__ __device__
  double operator() (double x){
    return pow(x, a);
  }

};

template<typename T>
struct EWMatmulFunc{
  const T a;

  EWMatmulFunc(T _a) : a(_a){}

  __host__ __device__
  T operator() (T x1, T x2){
    return a * x1 * x2;
  }

};


/* kernels definition */
template <typename FunctorOp, typename T, int BLOCK_SIZE>
  __global__ void elementwise_pattern_cuda(T* out_tensor, int length, FunctorOp elementwise_op);


template <typename FunctorOp, typename T, int BLOCK_SIZE>
  __global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor, int length, FunctorOp elementwise_op);


template <typename FunctorOp, typename T, int BLOCK_SIZE>
  __global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor1, T* in_tensor2, int length, FunctorOp elementwise_op);




/* ----------------------------- kernel implementation -------------------------------- */

template <typename FunctorOp, typename T, int BLOCK_SIZE>
__global__ void elementwise_pattern_cuda(T* in_tensor, int length, FunctorOp elementwise_op){
  // Block index
  int bx = blockIdx.x;

  // Thread index
  int tx = threadIdx.x;

  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    in_tensor[a_idx] = elementwise_op( in_tensor[a_idx] );
}



template <typename FunctorOp, typename T, int BLOCK_SIZE>
__global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor, int length, FunctorOp elementwise_op){

  // Block index
  int bx = blockIdx.x;

  // Thread index
  int tx = threadIdx.x;

  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    out_tensor[a_idx] = elementwise_op( in_tensor[a_idx] );
}


template <typename FunctorOp, typename T, int BLOCK_SIZE>
__global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor1, T* in_tensor2, int length, FunctorOp elementwise_op){
  // Block index
  int bx = blockIdx.x;

  // Thread index
  int tx = threadIdx.x;

  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    out_tensor[a_idx] = elementwise_op( in_tensor1[a_idx], in_tensor2[a_idx] );
}





#endif
