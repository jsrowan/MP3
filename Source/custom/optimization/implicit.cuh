#include "../gpu-new-forward.h"

constexpr int TILE_SIZE = 16;

// Output: B x M x H x W
// Input:  B x C x H x W
// Mask:   M x C x K x K
template<typename OutputType, typename InputType, typename MaskType>
__global__ void conv(OutputType output, InputType input, MaskType mask)
{
	__shared__ float sharedA[TILE_SIZE][TILE_SIZE];
	__shared__ float sharedB[TILE_SIZE][TILE_SIZE];

	constexpr int gemm_K = mask.sizeZ * mask.sizeY * mask.sizeX;
	constexpr int gemm_M = output.sizeZ;
	const int gemm_N = output.sizeW * output.sizeY * output.sizeX;
	
	const int gridWidth = CeilDiv(output.sizeW * output.sizeY * output.sizeX, TILE_SIZE);
	const int idx = blockIdx.x;
	const int gemm_m = blockDim.y * (idx / gridWidth) + threadIdx.y;
	const int gemm_n = blockDim.x * (idx % gridWidth) + threadIdx.x;

	const int b = gemm_n / (output.sizeY * output.sizeX);
	const int br = gemm_n % (output.sizeY * output.sizeX);

	const int ho = br / output.sizeX;
	const int wo = br % output.sizeX;

	float value = 0.0f;
	for (int i = 0; i < gemm_K; i += TILE_SIZE)
	{
		const int c1 = (i + threadIdx.x) / (mask.sizeX * mask.sizeY);
		const int cr1 = (i + threadIdx.x) % (mask.sizeX * mask.sizeY);
		const int c2 = (i + threadIdx.y) / (mask.sizeX * mask.sizeY);
		const int cr2 = (i + threadIdx.y) % (mask.sizeX * mask.sizeY);
		
		const int r1 = cr1 / mask.sizeX;
		const int s1 = cr1 % mask.sizeX;
		const int r2 = cr2 / mask.sizeX;
		const int s2 = cr2 % mask.sizeX;

		const int hi = ho + r2;
		const int wi = wo + s2;

		// Load elements into shared memory
		const bool inBoundsA = (gemm_m < gemm_M) && (i + threadIdx.x < gemm_K);
		sharedA[threadIdx.y][threadIdx.x] = inBoundsA ? mask(gemm_m, c1, r1, s1) : 0.0f;

		const bool inBoundsB = (i + threadIdx.y < gemm_K) && (gemm_n < gemm_N);
		sharedB[threadIdx.y][threadIdx.x] = inBoundsB ? input(b, c2, hi, wi) : 0.0f;
		__syncthreads();

		for (int j = 0; j < TILE_SIZE; j++)
		{
			value += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
		}
		__syncthreads();
	}

	if ((gemm_m < gemm_M) && (gemm_n < gemm_N))
	{
		output(b, gemm_m, ho, wo) = value;
	}
}

void GPUInterface::conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput,
	const float* pDeviceMask, int batch, int map, int channel, int height, int width, int k)
{
	int heightOut = height - k + 1;
	int widthOut = width - k + 1;

	int gridHeight = CeilDiv(map, TILE_SIZE);
	int gridWidth = CeilDiv(batch * heightOut * widthOut, TILE_SIZE);
	dim3 dimGrid(gridHeight * gridWidth, 1, 1);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

	// Pass 1
	if (map == MAP_1)
	{
		Tensor4D<0, MAP_1, HEIGHT_OUT_1, WIDTH_OUT_1> output(batch, pDeviceOutput);
		ConstTensor4D<0, CHANNEL_1, HEIGHT_1, WIDTH_1> input(batch, pDeviceInput);
		ConstTensor4D<MAP_1, CHANNEL_1, K_1, K_2> mask(pDeviceMask);

		conv<<<dimGrid, dimBlock>>>(output, input, mask);
	}
	// Pass 2
	else if (map == MAP_2)
	{
		Tensor4D<0, MAP_2, HEIGHT_OUT_2, WIDTH_OUT_2> output(batch, pDeviceOutput);
		ConstTensor4D<0, CHANNEL_2, HEIGHT_2, WIDTH_2> input(batch, pDeviceInput);
		ConstTensor4D<MAP_2, CHANNEL_2, K_2, K_2> mask(pDeviceMask);

		conv<<<dimGrid, dimBlock>>>(output, input, mask);
	}
}

void GPUInterface::conv_forward_gpu_prolog(float* pHostOutput, float* pHostInput,
	float* pHostMask, float** ppDeviceOutput, float** ppDeviceInput, float** ppDeviceMask,
	int batch, int map, int channel, int height, int width, int k)
{
	const int heightOut = height - k + 1;
	const int widthOut = width - k + 1;

	InitTensor(batch, channel, height, width, ppDeviceInput, pHostInput);
	InitTensor(map, channel, k, k, ppDeviceMask, pHostMask);
	InitTensor(batch, map, heightOut, widthOut, ppDeviceOutput);
}

void GPUInterface::conv_forward_gpu_epilog(float* pHostOutput, float* pDeviceOutput, float* pDeviceInput,
	float* pDeviceMask, int batch, int map, int channel, int height, int width, const int k)
{
	const int heightOut = height - k + 1;
	const int widthOut = width - k + 1;

	CopyTensor(batch, map, heightOut, widthOut, pDeviceOutput, pHostOutput);
	DestroyTensor(pDeviceOutput);
	DestroyTensor(pDeviceInput);
	DestroyTensor(pDeviceMask);
}