#include "../gpu-new-forward.h"

constexpr int GEMM_TILE_SIZE = 16;
constexpr int OUTPUT_TILE_SIZE = 10;
constexpr int INPUT_TILE_SIZE = OUTPUT_TILE_SIZE + 7 - 1;

constexpr int UNROLL_BATCH = 5000;

// Mask:   M x (C x K x K)
// Input:  (C x K x K) x (H_o x W_o)
// Output: M x (H_o x W_o)
template<typename MaskType, typename InputType, typename OutputType>
__global__ void gemm(MaskType mask, InputType input, OutputType output)
{
	__shared__ float sharedA[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
	__shared__ float sharedB[GEMM_TILE_SIZE][GEMM_TILE_SIZE];

	constexpr int M = mask.sizeY;
	constexpr int N = input.sizeX;
	constexpr int K = mask.sizeX;

	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int b = blockIdx.z;

	float value = 0.0f;
	for (int start_k = 0; start_k < K; start_k += GEMM_TILE_SIZE)
	{
		// Load elements into shared memory
		const bool inBoundsA = (row < M) && (start_k + threadIdx.x < K);
		sharedA[threadIdx.y][threadIdx.x] = inBoundsA ? mask[row * K + start_k + threadIdx.x] : 0.0f;

		const bool inBoundsB = (start_k + threadIdx.y < K) && (col < N);
		sharedB[threadIdx.y][threadIdx.x] = inBoundsB ? input(0, b, start_k + threadIdx.y, col) : 0.0f;
		__syncthreads();

		for (int k = 0; k < GEMM_TILE_SIZE; k++)
		{
			value += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
		}
		__syncthreads();
	}

	if ((row < M) && (col < N))
	{
		output(0, b, row, col) = value;
	}
}

// Input:    C x H x W
// Unrolled: (C x K x K) x (W_o x H_o)
template<int K, typename UnrolledType, typename InputType>
__global__ void unroll(UnrolledType unrolled, InputType input)
{
	__shared__ float sharedInput[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

	constexpr int widthOut = input.sizeX - K + 1;
	constexpr int heightOut = input.sizeY - K + 1;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int w = blockIdx.x * OUTPUT_TILE_SIZE + tx;
	const int h = blockIdx.y * OUTPUT_TILE_SIZE + ty;

	const int b = blockIdx.z / input.sizeZ;
	const int c = blockIdx.z % input.sizeZ;

	if ((h < input.sizeY) && (w < input.sizeX))
	{
		sharedInput[ty][tx] = input(b, c, h, w);
	}
	else
	{
		sharedInput[ty][tx] = 0.0f;
	}
	__syncthreads();

	if ((tx < OUTPUT_TILE_SIZE) && (ty < OUTPUT_TILE_SIZE) && (h < heightOut) && (w < widthOut))
	{
		const int row = c * K * K;
		const int col = h * widthOut + w;
		for (int p = 0; p < K; p++)
		{
			for (int q = 0; q < K; q++)
			{
				unrolled(0, b, row + p * K + q, col) = sharedInput[ty + p][tx + q];
			}
		}
	}
}

void GPUInterface::conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput, const float* pDeviceMask,
	int batch, int map, int channel, int height, int width, int k)
{
	const int heightOut = height - k + 1;
	const int widthOut = width - k + 1;
	const int heightUnroll = channel * k * k;
	const int widthUnroll = widthOut * heightOut;

	// Create unrolled temporary allocation
	float* pDeviceUnrolled;
	InitTensor(1, UNROLL_BATCH, heightUnroll, widthUnroll, &pDeviceUnrolled);

	dim3 dimGridUnroll(CeilDiv(widthOut, OUTPUT_TILE_SIZE), CeilDiv(heightOut, OUTPUT_TILE_SIZE), 
		channel * UNROLL_BATCH);
	dim3 dimBlockUnroll(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1);

	dim3 dimGridGemm(CeilDiv(widthUnroll, GEMM_TILE_SIZE), CeilDiv(map, GEMM_TILE_SIZE), 
		UNROLL_BATCH);
	dim3 dimBlockGemm(GEMM_TILE_SIZE, GEMM_TILE_SIZE, 1);

	// Pass 1
	if (map == MAP_1)
	{
		constexpr int HeightUnroll = CHANNEL_1 * K_1 * K_1;
		constexpr int WidthUnroll = HEIGHT_OUT_1 * WIDTH_OUT_1;
		Tensor4D<1, UNROLL_BATCH, HeightUnroll, WidthUnroll> unrolled(pDeviceUnrolled);
		ConstTensor4D<1, 1, MAP_1, HeightUnroll> mask(pDeviceMask);

		for (int b = 0; b < batch; b += UNROLL_BATCH)
		{
			dimGridUnroll.z = channel * std::min(UNROLL_BATCH, batch - b);
			dimGridGemm.z = std::min(UNROLL_BATCH, batch - b);

			ConstTensor4D<UNROLL_BATCH, CHANNEL_1, HEIGHT_1, WIDTH_1> input(pDeviceInput);
			Tensor4D<1, UNROLL_BATCH, MAP_1, WidthUnroll> output(pDeviceOutput);

			// Call unroll kernel
			unroll<K_1><<<dimGridUnroll, dimBlockUnroll>>>(unrolled, input);

			// Call GEMM kernel
			gemm<<<dimGridGemm, dimBlockGemm>>>(mask, unrolled, output);

			// Update pointers
			pDeviceInput += UNROLL_BATCH * channel * height * width;
			pDeviceOutput += UNROLL_BATCH * map * heightOut * widthOut;
		}
	}
	// Pass 2
	else if (map == MAP_2)
	{
		constexpr int HeightUnroll = CHANNEL_2 * K_2 * K_2;
		constexpr int WidthUnroll = HEIGHT_OUT_2 * WIDTH_OUT_2;
		Tensor4D<1, UNROLL_BATCH, HeightUnroll, WidthUnroll> unrolled(pDeviceUnrolled);
		ConstTensor4D<1, 1, MAP_2, HeightUnroll> mask(pDeviceMask);

		for (int b = 0; b < batch; b += UNROLL_BATCH)
		{
			dimGridUnroll.z = channel * std::min(UNROLL_BATCH, batch - b);
			dimGridGemm.z = std::min(UNROLL_BATCH, batch - b);

			ConstTensor4D<UNROLL_BATCH, CHANNEL_2, HEIGHT_2, WIDTH_2> input(pDeviceInput);
			Tensor4D<1, UNROLL_BATCH, MAP_2, WidthUnroll> output(pDeviceOutput);

			// Call unroll kernel
			unroll<K_2><<<dimGridUnroll, dimBlockUnroll>>>(unrolled, input);

			// Call GEMM kernel
			gemm<<<dimGridGemm, dimBlockGemm>>>(mask, unrolled, output);

			// Update pointers
			pDeviceInput += UNROLL_BATCH * channel * height * width;
			pDeviceOutput += UNROLL_BATCH * map * heightOut * widthOut;
		}
	}

	DestroyTensor(pDeviceUnrolled);
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