//  ***********************************************************************
//  Description:   GPU implementation of element-wise pattern operations
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************
#include "elementwise_pattern_cu.h"

// look at http://docs.nvidia.com/cuda/thrust/#axzz4KT0tW3IU to see examples of how to use thurst


template <typename FunctorOp, typename T, int BLOCK_SIZE>
__global__ void elementwise_pattern_cuda(T* in_tensor, int length, FunctorOp elementwise_op){
  // Block index
  int bx = blockIdx.x;

  // Thread index
  int tx = threadIdx.x;

  // create functor
  //FunctorOp elementwise_op;

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

  // create functor
  //FunctorOp elementwise_op;

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

  // create functor
  //FunctorOp elementwise_op;

  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    out_tensor[a_idx] = elementwise_op( in_tensor1[a_idx], in_tensor2[a_idx] );
}
