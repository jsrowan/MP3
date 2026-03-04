#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <device_launch_parameters.h>
#include <cassert>

#ifdef __INTELLISENSE__
#define __syncthreads()
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

// Parameters for different convolutional layers
constexpr int MAP_1 = 4;
constexpr int CHANNEL_1 = 1;
constexpr int HEIGHT_1 = 86;
constexpr int WIDTH_1 = 86;
constexpr int K_1 = 7;
constexpr int HEIGHT_OUT_1 = HEIGHT_1 - K_1 + 1;
constexpr int WIDTH_OUT_1 = WIDTH_1 - K_1 + 1;

constexpr int MAP_2 = 16;
constexpr int CHANNEL_2 = 4;
constexpr int HEIGHT_2 = 40;
constexpr int WIDTH_2 = 40;
constexpr int K_2 = 7;
constexpr int HEIGHT_OUT_2 = HEIGHT_2 - K_2 + 1;
constexpr int WIDTH_OUT_2 = WIDTH_2 - K_2 + 1;

CUDA_CALLABLE constexpr inline int CeilDiv(int x, int y)
{
	return (x + y - 1) / y;
}

CUDA_CALLABLE constexpr inline int NearestMul(int x, int y)
{
	return y * CeilDiv(x, y);
}

inline void CudaCheck(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void InitTensor(int dimW, int dimZ, int dimY, int dimX, float** ppDeviceData, const float* pHostData = nullptr)
{
	size_t size = dimW * dimZ * dimY * dimX * sizeof(float);
	CudaCheck(cudaMalloc(ppDeviceData, size));
	if (pHostData)
	{
		CudaCheck(cudaMemcpy(*ppDeviceData, pHostData, size, cudaMemcpyHostToDevice));
	}
}

inline void CopyTensor(int dimW, int dimZ, int dimY, int dimX, const float* pDeviceData, float* pHostData)
{
	size_t size = dimW * dimZ * dimY * dimX * sizeof(float);
	CudaCheck(cudaMemcpy(pHostData, pDeviceData, size, cudaMemcpyDeviceToHost));
}

inline void DestroyTensor(float* pDeviceData)
{
	CudaCheck(cudaFree(pDeviceData));
}

template<int SizeW = 0, int SizeZ = 0, int SizeY = 0, int SizeX = 0>
class ConstTensor4D
{
public:
	ConstTensor4D(const float* pDeviceData) :
		m_pDeviceData(pDeviceData)
	{
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	static constexpr int sizeW = SizeW;
	static constexpr int sizeZ = SizeZ;
	static constexpr int sizeY = SizeY;
	static constexpr int sizeX = SizeX;

private:
	const float* m_pDeviceData;
};

template<int SizeW = 0, int SizeZ = 0, int SizeY = 0, int SizeX = 0>
class Tensor4D
{
public:
	Tensor4D(float* pDeviceData) :
		m_pDeviceData(pDeviceData)
	{
	}

	operator ConstTensor4D<SizeW, SizeZ, SizeY, SizeX>()
	{
		return ConstTensor4D<SizeW, SizeZ, SizeY, SizeX>(m_pDeviceData);
	}

	CUDA_CALLABLE float& operator()(int w, int z, int y, int x)
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	CUDA_CALLABLE float& operator[](int idx)
	{
		return m_pDeviceData[idx];
	}

	static constexpr int sizeW = SizeW;
	static constexpr int sizeZ = SizeZ;
	static constexpr int sizeY = SizeY;
	static constexpr int sizeX = SizeX;

private:
	float* m_pDeviceData;
};

template<int SizeZ, int SizeY, int SizeX>
class ConstTensor4D<0, SizeZ, SizeY, SizeX>
{
public:
	ConstTensor4D(int sizeW, const float* pDeviceData) :
		sizeW(sizeW), m_pDeviceData(pDeviceData)
	{
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	int sizeW;
	static constexpr int sizeZ = SizeZ;
	static constexpr int sizeY = SizeY;
	static constexpr int sizeX = SizeX;

private:
	const float* m_pDeviceData;
};

template<int SizeZ, int SizeY, int SizeX>
class Tensor4D<0, SizeZ, SizeY, SizeX>
{
public:
	Tensor4D(int sizeW, float* pDeviceData) :
		sizeW(sizeW), m_pDeviceData(pDeviceData)
	{
	}

	operator ConstTensor4D<0, SizeZ, SizeY, SizeX>()
	{
		return ConstTensor4D<0, SizeZ, SizeY, SizeX>(sizeW, m_pDeviceData);
	}

	CUDA_CALLABLE float& operator()(int w, int z, int y, int x)
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	CUDA_CALLABLE float& operator[](int idx)
	{
		return m_pDeviceData[idx];
	}

	int sizeW;
	static constexpr int sizeZ = SizeZ;
	static constexpr int sizeY = SizeY;
	static constexpr int sizeX = SizeX;

private:
	float* m_pDeviceData;
};

template<int SizeW, int SizeY, int SizeX>
class ConstTensor4D<SizeW, 0, SizeY, SizeX>
{
public:
	ConstTensor4D(int sizeZ, const float* pDeviceData) :
		sizeZ(sizeZ), m_pDeviceData(pDeviceData)
	{
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	static constexpr int sizeW = SizeW;
	int sizeZ;
	static constexpr int sizeY = SizeY;
	static constexpr int sizeX = SizeX;

private:
	const float* m_pDeviceData;
};

template<int SizeW, int SizeY, int SizeX>
class Tensor4D<SizeW, 0, SizeY, SizeX>
{
public:
	Tensor4D(int sizeZ, float* pDeviceData) :
		sizeZ(sizeZ), m_pDeviceData(pDeviceData)
	{
	}

	operator ConstTensor4D<SizeW, 0, SizeY, SizeX>()
	{
		return ConstTensor4D<SizeW, 0, SizeY, SizeX>(sizeZ, m_pDeviceData);
	}

	CUDA_CALLABLE float& operator()(int w, int z, int y, int x)
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	CUDA_CALLABLE float& operator[](int idx)
	{
		return m_pDeviceData[idx];
	}

	static constexpr int sizeW = SizeW;
	int sizeZ;
	static constexpr int sizeY = SizeY;
	static constexpr int sizeX = SizeX;

private:
	float* m_pDeviceData;
};

template<>
class ConstTensor4D<0, 0, 0, 0>
{
public:
	ConstTensor4D(int sizeW, int sizeZ, int sizeY, int sizeX, const float* pDeviceData) :
		sizeW(sizeW),sizeZ(sizeZ), sizeY(sizeY), sizeX(sizeX), m_pDeviceData(pDeviceData)
	{
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	int sizeW;
	int sizeZ;
	int sizeY;
	int sizeX;

private:
	const float* m_pDeviceData;
};

template<>
class Tensor4D<0, 0, 0, 0>
{
public:
	Tensor4D(int sizeW, int sizeZ, int sizeY, int sizeX, float* pDeviceData) :
		sizeW(sizeW), sizeZ(sizeZ), sizeY(sizeY), sizeX(sizeX), m_pDeviceData(pDeviceData)
	{
	}

	operator ConstTensor4D<0, 0, 0, 0>()
	{
		return ConstTensor4D<0, 0, 0, 0>(sizeW, sizeZ, sizeY, sizeX, m_pDeviceData);
	}

	CUDA_CALLABLE float& operator()(int w, int z, int y, int x)
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator()(int w, int z, int y, int x) const
	{
		return m_pDeviceData[sizeZ * sizeY * sizeX * w + sizeY * sizeX * z + sizeX * y + x];
	}

	CUDA_CALLABLE const float& operator[](int idx) const
	{
		return m_pDeviceData[idx];
	}

	CUDA_CALLABLE float& operator[](int idx)
	{
		return m_pDeviceData[idx];
	}

	int sizeW;
	int sizeZ;
	int sizeY;
	int sizeX;

private:
	float* m_pDeviceData;
};

struct GPUInterface
{
	void get_device_properties();
	void conv_forward_gpu_prolog(float* pHostOutput, float* pHostInput, float* pHostMask,
		float** ppDeviceOutput, float** ppDeviceInput, float** ppDeviceMask,
		int batch, int map, int channel, int height, int width, int k);
	void conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput, const float* pDeviceMask,
		int batch, int map, int channel, int height, int width, int k);
	void conv_forward_gpu_epilog(float* pHostOutput, float* pDeviceOutput, float* pDeviceInput, float* pDeviceMask,
		int batch, int map, int channel, int height, int width, const int k);
};

