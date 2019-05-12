//  ***********************************************************************
//  Implementation of a Gaussian Mixture Model
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************


#ifndef TENSORFLOW_KERNELS_GMM_MODEL_OP_H_
#define TENSORFLOW_KERNELS_GMM_MODEL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


#include "cuda.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
//#include "tensorflow/core/util/stream_executor_util.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"
#include "unsupported/Eigen/CXX11/ThreadPool"
//#include "tensorflow/core/platform/stream_executor.h"


#include "twodlearn/core/cuda/matmul_pattern.cu.h"
#include <iostream>
#include <cmath>

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define PI 3.14159265358979323846

namespace tensorflow{
namespace functor {

  // definition of the functor
  template <typename Device, typename T>
    struct GmmModelFunctor {
      // Computes on device "d": p_x= sum_k (w_k * Gaussian(x_i, mu_k, sigma_k)).
      void operator()(
		      const OpKernelContext* context,
		      Tensor& p_x_tf,
		      Tensor& gaussians_tf,
		      Tensor& sigma_inv_x_mu_tf,
		      const Tensor& x_tf,
		      const Tensor& w_tf,
		      const Tensor& mu_tf,
		      const Tensor& sigma_tf
		      );
    };


} // end namespace functor
} // end namespace tensorflow

#endif // TENSORFLOW_KERNELS_GMM_MODEL_OP_H_
