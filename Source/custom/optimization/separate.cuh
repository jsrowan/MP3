#include "../gpu-new-forward.h"

constexpr int TILE_SIZE = 8;

// Output: B x M x H x W
// Input:  B x C x H x W
// Mask:   M x C x K x K
template<typename OutputType, typename InputType, typename MaskType>
__global__ void conv(OutputType output, InputType input, MaskType mask)
{
	constexpr int gridWidth = CeilDiv(output.sizeX, TILE_SIZE);

	// Insert your GPU convolution kernel code here
	const int b = blockIdx.z;
	const int m = blockIdx.x;
	const int h = (blockIdx.y / gridWidth) * TILE_SIZE + threadIdx.y;
	const int w = (blockIdx.y % gridWidth) * TILE_SIZE + threadIdx.x;

	float accum = 0.0f;
	for (int c = 0; c < mask.sizeZ; c++)
	{
		for (int p = 0; p < mask.sizeY; p++)
		{
			for (int q = 0; q < mask.sizeX; q++)
			{
				if ((h + p < input.sizeY) && (w + q < input.sizeX))
				{
					accum += input(b, c, h + p, w + q) * mask(m, c, p, q);
				}
			}
		}
	}
	if ((h < output.sizeY) && (w < output.sizeX))
	{
		output(b, m, h, w) = accum;
	}
}

void GPUInterface::conv_forward_gpu(float* pDeviceOutput, const float* pDeviceInput,
	const float* pDeviceMask, int batch, int map, int channel, int height, int width, int k)
{
	int heightOut = height - k + 1;
	int widthOut = width - k + 1;

	int gridHeight = CeilDiv(heightOut, TILE_SIZE);
	int gridWidth = CeilDiv(widthOut, TILE_SIZE);
	dim3 dimGrid(map, gridHeight * gridWidth, batch);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

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