#include "../gpu-new-forward.h"

constexpr int TK = 32;

// Output: B x M x H x W	(Matrix C)
// Input:  B x C x H x W	(Matrix B)
// Mask:   M x C x K x K	(Matrix A)
template<typename OutputType, typename InputType, typename MaskType>
__global__ void conv1(OutputType output, InputType input, MaskType mask)
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

constexpr int WARP_SIZE = 32;
constexpr int WMMA_SIZE = 16;
constexpr int WARP_BLOCK_X = 32;
constexpr int WARP_BLOCK_Y = 16;
constexpr int THREAD_BLOCK_AX = 256;
constexpr int THREAD_BLOCK_BX = 512;
constexpr int THREAD_BLOCK_Y = 16;
constexpr int BLOCK_SIZE = 512;

// Output: B x M x H x W	(Matrix C)
// Input:  B x C x H x W	(Matrix B)
// Mask:   M x C x K x K	(Matrix A)
template<typename OutputType, typename InputType, typename MaskType>
__global__ void conv2(OutputType output, InputType input, MaskType mask)
{
	using namespace nvcuda::wmma;

	__shared__ half sharedA[THREAD_BLOCK_Y][THREAD_BLOCK_AX + 16];
	__shared__ half sharedB[THREAD_BLOCK_Y][THREAD_BLOCK_BX + 16];

	constexpr int gemm_M = output.sizeZ;
	constexpr int gemm_K = input.sizeZ * mask.sizeX * mask.sizeY;
	const int gemm_N = output.sizeW * output.sizeY * output.sizeX;

	// Load the tile of mask (A) into shared memory
	constexpr int loadStride = BLOCK_SIZE / THREAD_BLOCK_AX;
	const int load_y = threadIdx.x / THREAD_BLOCK_AX;
	const int load_x = threadIdx.x % THREAD_BLOCK_AX;
	#pragma unroll
	for (int m = 0; m < THREAD_BLOCK_Y; m += loadStride)
	{
		if (load_x < gemm_K)
		{
			sharedA[load_y + m][load_x] = __float2half(mask[(load_y + m) * gemm_K + load_x]);
		}
		else
		{
			sharedA[load_y + m][load_x] = 0.0f;
		}
	}
	__syncthreads();

	// Compute warp index
	const int wx = threadIdx.x / WARP_SIZE;

	// Calculate indices into input and output tensors
	const int gemm_n = blockIdx.x * THREAD_BLOCK_BX + threadIdx.x;
	const int b = gemm_n / (output.sizeY * output.sizeX);
	const int br = gemm_n % (output.sizeY * output.sizeX);
	const int ho = br / output.sizeX;
	const int wo = br % output.sizeX;

	fragment<matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, row_major> aFragment;
	fragment<matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, row_major> bFragment;

	fragment<accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half> cFragment1;
	fragment<accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half> cFragment2;

	fill_fragment(cFragment1, 0.0f);
	fill_fragment(cFragment2, 0.0f);
	#pragma unroll
	for (int start_k = 0; start_k < gemm_K; start_k += THREAD_BLOCK_Y)
	{
		// Load the mask warp tile from shared memory
		load_matrix_sync(aFragment, &sharedA[0][start_k], THREAD_BLOCK_AX + 16);

		// Load the tile of B into shared memory. Because each part of the tile is loaded and then used
		// by the same warp, block-level synchronization is not needed. 
		// wmma::load_matrix_sync will do warp-level synchronization.
		#pragma unroll
		for (int k = 0; k < THREAD_BLOCK_Y; k++)
		{
			const int gemm_k = start_k + k;
			if ((gemm_k < gemm_K) && (gemm_n < gemm_N))
			{
				const int c = gemm_k / (mask.sizeY * mask.sizeX);
				const int cr = gemm_k % (mask.sizeY * mask.sizeX);

				const int hi = ho + (cr / mask.sizeX);
				const int wi = wo + (cr % mask.sizeX);

				sharedB[k][threadIdx.x] = __float2half(input(b, c, hi, wi));
			}
			else
			{
				sharedB[k][threadIdx.x] = 0.0f;
			}
		}

		// Accumulate matrix products
		load_matrix_sync(bFragment, &sharedB[0][wx * WARP_BLOCK_X], THREAD_BLOCK_BX + 16);
		mma_sync(cFragment1, aFragment, bFragment, cFragment1);

		load_matrix_sync(bFragment, &sharedB[0][wx * WARP_BLOCK_X + WMMA_SIZE], THREAD_BLOCK_BX + 16);
		mma_sync(cFragment2, aFragment, bFragment, cFragment2);
	}

	// Store the output
	store_matrix_sync(&sharedB[0][wx * WARP_BLOCK_X], cFragment1, THREAD_BLOCK_BX + 16, mem_row_major);
	store_matrix_sync(&sharedB[0][wx * WARP_BLOCK_X + WMMA_SIZE], cFragment2, THREAD_BLOCK_BX + 16, mem_row_major);
	if (gemm_n < gemm_N)
	{
		#pragma unroll
		for (int m = 0; m < gemm_M; m++)
		{
			output(b, m, ho, wo) = __half2float(sharedB[m][threadIdx.x]);
		}
	}
}

void InvokeLayer1(float* pDeviceOutput, const float* pDeviceInput, const float* pDeviceMask, 
	int batch, cudaStream_t stream)
{
	int blockSize = NearestMul(CHANNEL_1 * K_1 * K_1, TK);
	dim3 dimGrid(CeilDiv(batch * HEIGHT_OUT_1 * WIDTH_OUT_1, blockSize), 1, 1);
	dim3 dimBlock(blockSize, 1, 1);

	Tensor4D<0, MAP_1, HEIGHT_OUT_1, WIDTH_OUT_1> output(batch, pDeviceOutput);
	ConstTensor4D<0, CHANNEL_1, HEIGHT_1, WIDTH_1> input(batch, pDeviceInput);
	ConstTensor4D<MAP_1, CHANNEL_1, K_1, K_1> mask(pDeviceMask);

	conv1<<<dimGrid, dimBlock, 0, stream>>>(output, input, mask);
}

void InvokeLayer2(float* pDeviceOutput, const float* pDeviceInput, const float* pDeviceMask,
	int batch, cudaStream_t stream)
{
	dim3 dimGrid(CeilDiv(batch * HEIGHT_OUT_2 * WIDTH_OUT_2, THREAD_BLOCK_BX), 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	Tensor4D<0, MAP_2, HEIGHT_OUT_2, WIDTH_OUT_2> output(batch, pDeviceOutput);
	ConstTensor4D<0, CHANNEL_2, HEIGHT_2, WIDTH_2> input(batch, pDeviceInput);
	ConstTensor4D<MAP_2, CHANNEL_2, K_2, K_2> mask(pDeviceMask);

	conv2<<<dimGrid, dimBlock, 0, stream>>>(output, input, mask);
}

void GPUInterface::conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput,
	const float* pDeviceMask, int batch, int map, int channel, int height, int width, int k)
{
}

constexpr int SEG_SIZE = 1000;

void GPUInterface::conv_forward_gpu_prolog(float* pHostOutput, float* pHostInput,
	float* pHostMask, float** ppDeviceOutput, float** ppDeviceInput, float** ppDeviceMask,
	int batch, int map, int channel, int height, int width, int k)
{
	const int heightOut = height - k + 1;
	const int widthOut = width - k + 1;

	size_t maskSize = map * channel * k * k * sizeof(float);
	size_t inputSize = batch * channel * height * width * sizeof(float);
	size_t outputSize = batch * map * heightOut * widthOut * sizeof(float);
	size_t inputSegSize = channel * height * width * sizeof(float);
	size_t outputSegSize = map * heightOut * widthOut * sizeof(float);

	// Pin memory on host
	CudaCheck(cudaHostRegister(pHostOutput, outputSize, cudaHostRegisterDefault));
	CudaCheck(cudaHostRegister(pHostInput, inputSize, cudaHostRegisterDefault));
	CudaCheck(cudaHostRegister(pHostMask, maskSize, cudaHostRegisterDefault));

	// Allocate device memory for the mask
	float* deviceMask;
	CudaCheck(cudaMalloc(&deviceMask, maskSize));
	CudaCheck(cudaMemcpy(deviceMask, pHostMask, maskSize, cudaMemcpyHostToDevice));

	cudaStream_t streams[3] = {};
	float* deviceInputPtrs[3] = {};
	float* deviceOutputPtrs[3] = {};
	for (int i = 0; i < 3; i++)
	{
		CudaCheck(cudaStreamCreate(&streams[i]));
		CudaCheck(cudaMalloc(&deviceInputPtrs[i], SEG_SIZE * inputSegSize));
		CudaCheck(cudaMalloc(&deviceOutputPtrs[i], SEG_SIZE * outputSegSize));
	}

	int nSegments = CeilDiv(batch, SEG_SIZE);
	for (int i = 0; i < nSegments; i += 2)
	{
		for (int j = 0; j < 2; j++)
		{
			int idx = (i + j) % 3;
			int remBatches = std::min(SEG_SIZE, batch - (i + j) * SEG_SIZE);
			if (remBatches > 0)
			{
				CudaCheck(cudaMemcpyAsync(deviceInputPtrs[idx], pHostInput, remBatches * inputSegSize,
					cudaMemcpyHostToDevice, streams[idx]));
				pHostInput += remBatches * channel * height * width;
			}
		}
		
		// Invoke kernels
		for (int j = 0; j < 2; j++)
		{
			int idx = (i + j) % 3;
			int remBatches = std::min(SEG_SIZE, batch - (i + j) * SEG_SIZE);
			if (remBatches > 0)
			{
				if (map == MAP_1)
				{
					InvokeLayer1(deviceOutputPtrs[idx], deviceInputPtrs[idx], deviceMask, 
						remBatches, streams[idx]);
				}
				else
				{
					InvokeLayer2(deviceOutputPtrs[idx], deviceInputPtrs[idx], deviceMask, 
						remBatches, streams[idx]);
				}
			}
		}
		
		for (int j = 0; j < 2; j++)
		{
			int idx = (i + j) % 3;
			int remBatches = std::min(SEG_SIZE, batch - (i + j) * SEG_SIZE);
			if (remBatches > 0)
			{
				CudaCheck(cudaMemcpyAsync(pHostOutput, deviceOutputPtrs[idx], remBatches * outputSegSize, 
					cudaMemcpyDeviceToHost, streams[idx]));
				pHostOutput += remBatches * map * heightOut * widthOut;
			}
		}
	}

	CudaCheck(cudaFree(deviceMask));
	for (int i = 0; i < 3; i++)
	{
		CudaCheck(cudaStreamDestroy(streams[i]));
		CudaCheck(cudaFree(deviceInputPtrs[i]));
		CudaCheck(cudaFree(deviceOutputPtrs[i]));
	}
}

void GPUInterface::conv_forward_gpu_epilog(float* pHostOutput, float* pDeviceOutput, float* pDeviceInput,
	float* pDeviceMask, int batch, int map, int channel, int height, int width, const int k)
{
}