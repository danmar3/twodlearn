//  ***********************************************************************
//  Description:   Basic matrix multiplication
//  Written by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************


#ifndef TENSORFLOW_KERNELS_MY_MATMUL_OP_H_
#define TENSORFLOW_KERNELS_MY_MATMUL_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "twodlearn/core/cuda/matmul_pattern.cu.h"
#include <iostream>
//#include "tensorflow/core/framework/tensor_types.h"
//#include "tensorflow/core/framework/register_types.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow{
namespace functor {

  // definition of the functor
  template <typename Device, typename T>
    struct MyMatmulFunctor {
      // Computes on device "d": out = in0 * in1, where * is matrix
      // multiplication.
      void operator()(
		      //const Device& d,
		      const OpKernelContext* context,
		      Tensor& out,
		      const Tensor& in0,
		      const Tensor& in1
		      );
    };


} // end namespace functor
} // end namespace tensorflow

#endif // TENSORFLOW_KERNELS_MY_MATMUL_OP_H_
