//  ***********************************************************************
//  Class that handles the memory storage of a given object in host
//    and device memory
//  Wrote by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//  ***********************************************************************
#ifndef TWIN_OBJECT_H_
#define TWIN_OBJECT_H_

#include <stdlib.h>
#include <cuda.h>
#include <functional>
#include "twodlearn/core/cuda/cuda_error.cu.h"

template <class Obj, typename T>
class TwinObject{
public:
  Obj* obj_ptr; // pointer to the host object

  T* host;   // pointer to host memory
  T* device;   // pointer to device memory

  std::size_t obj_size; // size of the object

  std::function<T*(void)> get_h;

  /* Methods */
  // constructors
  TwinObject( ) : obj_ptr(nullptr),
  obj_size(0),
  host(nullptr),
  device(nullptr) {}

  TwinObject(Obj* in_obj, std::size_t in_size, T* h_in ):
  obj_ptr(in_obj),
  obj_size(in_size),
  host(h_in)
  {
    // allocate memory on device
    CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }

  TwinObject(Obj* in_obj, std::size_t in_size,
             std::function<T*(void)> get_h_in):
  obj_ptr(in_obj),
  obj_size(in_size),
  get_h(get_h_in){
    // allocate memory on device
    CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }

  // destructor
  ~TwinObject(){
    // free cuda memory
    cudaFree(device);
  }

  void initialize(Obj* in_obj, std::size_t in_size,
                  std::function<T*(void)> get_h_in) {
    obj_ptr = in_obj;
    obj_size = in_size;
    get_h = get_h_in;

    // allocate memory on device
    CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }


  void set_get_h(std::function<T*(void)> get_h_in){
    get_h = get_h_in;
  }


  void transfer_d2h(){
    if (get_h != nullptr)
    host = get_h();

    CUDA_CHECK( cudaMemcpy(host, device, obj_size, cudaMemcpyDeviceToHost) );
  }

  void transfer_h2d(){
    if (get_h != nullptr){
      host = get_h();
    }

    CUDA_CHECK( cudaMemcpy(device, host, obj_size, cudaMemcpyHostToDevice) );
  }

  void free_device(){
    cudaFree(device);
  }

  void realocate_device(){
    CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }
};


#endif //
