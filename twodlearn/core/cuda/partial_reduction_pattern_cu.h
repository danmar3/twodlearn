// ************************************************************************
//   Written by: Daniel L. Marino (marinodl@vcu.edu) (2016)
// ***********************************************************************
//
//   Description:   GPU implementation of partial reductions
//
// ***********************************************************************

#ifndef ELEMENTWISE_PATTERN_CU_H_
#define ELEMENTWISE_PATTERN_CU_H_

#include <cuda.h>

template <typename FunctorOp, typename T, int BLOCK_SIZE>
  __global__ void elementwise_pattern_cuda(T* out_tensor, int length);



template <typename FunctorOp, typename T, int BLOCK_SIZE>
  __global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor, int length);



#endif
