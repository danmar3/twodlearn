//  ***********************************************************************
//  Test of matmul pattern implementation
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************
/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <functional>

/* Includes, cuda */
#include "twodlearn/core/cuda/eigen_cuda.cu.h"
#include "twodlearn/core/cuda/matmul_pattern.cu.h"

#define BLOCK_SIZE 32

/* Includes, eigen */
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

/* Main */
int main(int argc, char **argv){

  unsigned m= 1000;
  unsigned k= 800;
  unsigned n= 1101;

  struct timespec start_cpu, end_cpu;

  // Allocate and fill h_A and h_B with data:
  TwinMat<double, RowMajor> a(m, k);
  a.transfer_h2d();

  TwinMat<double, RowMajor> b(k, n);
  b.transfer_h2d();

  TwinMat<double, RowMajor> c(m, n);
  //c.transfer_h2d();

  // For performance measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // 1. --------------------------- matmul test ---------------------------
  // 1.1. matmul on cpu
  cout << "Running matmul on cpu" << endl;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);
  MatrixXd c_eigen = a.mat * b.mat;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
  uint64_t cpu_time_ms = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) +
  end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;

  // 1.2. matmul on gpu
  cout << "Running matmul on GPU" << endl;
  MulFunc<double> mul_cu;
  SumFunc<double> sum_cu;
  dim3 dim_grid( 1 + ((c.mat.cols() -1)/BLOCK_SIZE),
		 1 + ((c.mat.rows() -1)/BLOCK_SIZE),
		 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
  cout << dim_grid.x << " " << dim_grid.y << " " << BLOCK_SIZE << endl;

  cudaEventRecord(start);

  matmul_pattern_cuda<MulFunc<double>, SumFunc<double>, double, BLOCK_SIZE>
    <<<dim_grid, dim_block>>>
    (c.device, a.device, b.device, a.mat.rows(), a.mat.cols(),
     b.mat.cols(), mul_cu, sum_cu);
  cudaDeviceSynchronize();
  c.transfer_d2h();

  cudaEventRecord(stop);

  // 1.3. print performance
  // error
  MatrixXd diff= c_eigen - c.mat;
  diff = diff.array().pow(2);
  cout << "difference: " << diff.sum() << "\n";
  // time
  cudaEventSynchronize(stop);
  float gpu_time_ms;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  cout << "time on cpu: " << cpu_time_ms << "[ms] \n";
  cout << "time on gpu: " << gpu_time_ms << "[ms] \n";
  if ((a.mat.rows()) < 5 &&
      (a.mat.cols()) < 5 &&
      (b.mat.cols()) < 5){
    cout << "A:" << endl<< a.mat << endl;
    cout << "B:" << endl<< b.mat << endl;
    cout << "C(CPU):" << endl<< c_eigen << endl;
    cout << "C(GPU):" << endl<< c.mat << endl;
  }
}
