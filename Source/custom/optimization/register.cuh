#include "../gpu-new-forward.h"

constexpr int BK = 64;
constexpr int TK = 32;

// Output: B x M x H x W	(Matrix C)
// Input:  B x C x H x W	(Matrix B)
// Mask:   M x C x K x K	(Matrix A)
template<typename OutputType, typename InputType, typename MaskType>
__global__ void conv(OutputType output, InputType input, MaskType mask)
{
	constexpr int gemm_M = output.sizeZ;
	constexpr int gemm_K = input.sizeZ * mask.sizeX * mask.sizeY;
	const int gemm_N = output.sizeW * output.sizeY * output.sizeX;

	// Pad the shared memory to help avoid any bank conflict
	__shared__ float sharedA[gemm_M][BK + 1];
	float regB[TK];

	const int gemm_n = blockIdx.x * blockDim.x + threadIdx.x;
	if (gemm_n >= gemm_N)
	{
		return;
	}

	const int b = gemm_n / (output.sizeY * output.sizeX);
	const int br = gemm_n % (output.sizeY * output.sizeX);

	float result[gemm_M] = { 0.0f };
	#pragma unroll
	for (int blockStart_k = 0; blockStart_k < gemm_K; blockStart_k += BK)
	{
		// Load a tile from mask into shared memory
		for (int m = 0; m < gemm_M; m++)
		{
			sharedA[m][threadIdx.x] = mask[m * gemm_K + blockStart_k + threadIdx.x];
			if (blockStart_k + threadIdx.x < gemm_K)
			{
				sharedA[m][threadIdx.x] = mask[m * gemm_K + blockStart_k + threadIdx.x];
			}
			else
			{
				sharedA[m][threadIdx.x] = 0.0f;
			}
		}
		__syncthreads();

		#pragma unroll
		for (int threadStart_k = 0; threadStart_k < BK; threadStart_k += TK)
		{
			// Load a tile from input into the registers
			for (int k = 0; k < TK; k++)
			{
				const int gemm_k = blockStart_k + threadStart_k + k;

				if (gemm_k < gemm_K)
				{
					const int c = gemm_k / (mask.sizeY * mask.sizeX);
					const int cr = gemm_k % (mask.sizeY * mask.sizeX);

					const int hi = (br / output.sizeX) + (cr / mask.sizeX);
					const int wi = (br % output.sizeX) + (cr % mask.sizeX);

					regB[k] = input(b, c, hi, wi);
				}
				else
				{
					regB[k] = 0.0f;
				}
			}

			// Perform the inner product computation
			#pragma unroll
			for (int m = 0; m < gemm_M; m++)
			{
				#pragma unroll
				for (int k = 0; k < TK; k++)
				{
					result[m] += regB[k] * sharedA[m][threadStart_k + k];
				}
			}
		}
		__syncthreads();
	}

	// Store results to output
	#pragma unroll
	for (int m = 0; m < gemm_M; m++)
	{
		output(b, m, br / output.sizeX, br % output.sizeX) = result[m];
	}
}

void GPUInterface::conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput,
	const float* pDeviceMask, int batch, int map, int channel, int height, int width, int k)
{
	int heightOut = height - k + 1;
	int widthOut = width - k + 1;

	dim3 dimGrid(CeilDiv(batch * heightOut * widthOut, BK), 1, 1);
	dim3 dimBlock(BK, 1, 1);

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
	InitTensor(batch, map, heightOut, widthOut, ppDeviceOutput);
	InitTensor(map, channel, k, k, ppDeviceMask, pHostMask);
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