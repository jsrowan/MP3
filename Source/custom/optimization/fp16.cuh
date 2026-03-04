#include "../gpu-new-forward.h"

constexpr int TK = 32;

// Output: B x M x H x W	(Matrix C)
// Input:  B x C x H x W	(Matrix B)
// Mask:   M x C x K x K	(Matrix A)
template<typename OutputType, typename InputType, typename MaskType>
__global__ void conv(OutputType output, InputType input, MaskType mask)
{
	const int gemm_N = output.sizeW * output.sizeY * output.sizeX;
	constexpr int gemm_K = input.sizeZ * mask.sizeX * mask.sizeY;
	constexpr int gemm_M = output.sizeZ;
	static_assert(gemm_M % 2 == 0, "M dimension should be a multiple of 2 for half2 packing");

	const int gemm_n = blockIdx.x * blockDim.x + threadIdx.x;
	const int b = gemm_n / (output.sizeY * output.sizeX);
	const int br = gemm_n % (output.sizeY * output.sizeX);

	// Load the entire mask into shared memory
	constexpr int sharedSize_K = NearestMul(gemm_K, TK);
	__shared__ half2 sharedA[gemm_M / 2][sharedSize_K + 1];
	for (int m = 0; m < gemm_M / 2; m++)
	{
		if (threadIdx.x < gemm_K)
		{
			float load1 = mask[(2 * m + 0) * gemm_K + threadIdx.x];
			float load2 = mask[(2 * m + 1) * gemm_K + threadIdx.x];
			sharedA[m][threadIdx.x] = __floats2half2_rn(load1, load2);
		}		
		else
		{
			sharedA[m][threadIdx.x] = __floats2half2_rn(0.0f, 0.0f);
		}
	}
	__syncthreads();

	// Now that the mask has been cooperatively loaded, we can exit if the thread would write
	// out of bounds, since each thread is responsible for a single column of the output matrix,
	// i.e. there is no more cooperation between threads after this point.
	if (gemm_n >= gemm_N)
	{
		return;
	}

	half2 regB[TK / 2];
	half2 result[gemm_M / 2] = { __floats2half2_rn(0.0f, 0.0f) };
	#pragma unroll
	for (int start_k = 0; start_k < gemm_K; start_k += TK)
	{
		// Load a tile from input into the registers
		#pragma unroll
		for (int k = 0; k < TK / 2; k++)
		{
			int gemm_k = start_k + 2 * k;
			float load1;
			if (gemm_k < gemm_K)
			{
				const int c = gemm_k / (mask.sizeY * mask.sizeX);
				const int cr = gemm_k % (mask.sizeY * mask.sizeX);

				const int hi = (br / output.sizeX) + (cr / mask.sizeX);
				const int wi = (br % output.sizeX) + (cr % mask.sizeX);

				load1 = input(b, c, hi, wi);
			}
			else
			{
				load1 = 0.0f;
			}
			gemm_k++;
			float load2;
			if (gemm_k < gemm_K)
			{
				const int c = gemm_k / (mask.sizeY * mask.sizeX);
				const int cr = gemm_k % (mask.sizeY * mask.sizeX);

				const int hi = (br / output.sizeX) + (cr / mask.sizeX);
				const int wi = (br % output.sizeX) + (cr % mask.sizeX);

				load2 = input(b, c, hi, wi);
			}
			else
			{
				load2 = 0.0f;
			}
			regB[k] = __floats2half2_rn(load1, load2);
		}

		// Do the inner product computation
		#pragma unroll
		for (int m = 0; m < gemm_M / 2; m++)
		{
			#pragma unroll
			for (int k = 0; k < TK / 2; k++)
			{
				half2 lo = __low2half2(regB[k]);
				half2 hi = __high2half2(regB[k]);
				result[m] = __hfma2(sharedA[m][start_k + 2 * k + 0], lo, result[m]);
				result[m] = __hfma2(sharedA[m][start_k + 2 * k + 1], hi, result[m]);
				result[m] = __hfma2(sharedA[m][start_k + 2 * k + 0], lo, result[m]);
				result[m] = __hfma2(sharedA[m][start_k + 2 * k + 1], hi, result[m]);
			}
		}
	}

	// Write to output
	#pragma unroll
	for (int m = 0; m < gemm_M / 2; m++)
	{
		float2 out = __half22float2(result[m]);
		output(b, 2 * m + 0, br / output.sizeX, br % output.sizeX) = out.x;
		output(b, 2 * m + 1, br / output.sizeX, br % output.sizeX) = out.y;
	}
}

void GPUInterface::conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput,
	const float* pDeviceMask, int batch, int map, int channel, int height, int width, int k)
{
	int heightOut = height - k + 1;
	int widthOut = width - k + 1;

	int blockSize = NearestMul(channel * k * k, TK);
	dim3 dimGrid(CeilDiv(batch * heightOut * widthOut, blockSize), 1, 1);
	dim3 dimBlock(blockSize, 1, 1);

	// Pass 1
	if (map == MAP_1)
	{
		Tensor4D<0, MAP_1, HEIGHT_OUT_1, WIDTH_OUT_1> output(batch, pDeviceOutput);
		ConstTensor4D<0, CHANNEL_1, HEIGHT_1, WIDTH_1> input(batch, pDeviceInput);
		ConstTensor4D<MAP_1, CHANNEL_1, K_1, K_1> mask(pDeviceMask);

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