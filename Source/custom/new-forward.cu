#include <cmath>
#include <iostream>

#include "gpu-new-forward.h"

#define TENSOR

#if defined(BASELINE)
#include "optimization/baseline.cuh"
#elif defined(SEPARATE)
#include "optimization/separate.cuh"
#elif defined(UNROLL)
#include "optimization/unroll.cuh"
#elif defined(IMPLICIT)
#include "optimization/implicit.cuh"
#elif defined(REGISTER)
#include "optimization/register.cuh"
#elif defined(FP16)
#include "optimization/fp16.cuh"
#elif defined(TENSOR)
#include "optimization/tensor.cuh"
#elif defined(STREAMS)
#include "optimization/streams.cuh"
#endif

void GPUInterface::get_device_properties()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
		std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
		std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
		std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
		std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
		std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
		std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
			<< deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
		std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
			<< deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
		std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
	}
}


