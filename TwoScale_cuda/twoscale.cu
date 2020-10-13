#include "twoscale.h"

#define BLK_SZ 16
#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)

inline void __cudaCheckError(cudaError_t err, const char *file, int line)
{
	if(err != cudaSuccess)
	{
		cout << err << " in " << file << " at " << line << " line" << endl;
	}
}

BFilter::BFilter(int wid, int hei)
{
	cudaCheckError(cudaMalloc((void**)&d_imgIn_, sizeof(float) * wid * hei));
	cudaCheckError(cudaMalloc((void **)&d_imgOut_, sizeof(float) * wid * hei));
}

BFilter::~BFilter()
{
	if(d_imgIn_)
		cudaFree(d_imgIn_);
	if(d_imgOut_)
		cudaFree(d_imgOut_);
}

// do boxfilter on separable two dimension accumulation
// process row
// 加运算比移位运算优先级高
__device__ void d_boxfilter_x(float *Out, float *imgIn, int wid, int hei, int filterR)
{
	float scale = 1.0f / (float)((filterR << 1) + 1);
	//float scale = 0.0322581;
	float t;

	// do the left edge
	t = imgIn[0] * filterR;
	for(int x = 0; x < (filterR + 1); x++)
	{
		t += imgIn[x];
	}

	Out[0] = t * scale;

	for(int x = 1; x < (filterR + 1); x++)
	{
		t += imgIn[x + filterR];
		t -= imgIn[0];
		Out[x] = t * scale;
	}

	// main loop
	for(int x = (filterR + 1); x < (wid - filterR); x++)
	{
		t += imgIn[x + filterR];
		t -= imgIn[x - filterR - 1];
		Out[x] = t * scale;
	}

	// do the right edge
	for(int x = (wid - filterR); x < wid; x++)
	{
		t += imgIn[wid - 1];
		t -= imgIn[x - filterR - 1];
		Out[x] = t *  scale;
	}
}

// process column
__device__ void d_boxfilter_y(float *imgOut,float *imgIn, int wid, int hei, int filterR)
{
	float scale = 1.0f / (float)((filterR << 1) + 1);
	//float scale = 0.0322581;

	float t;

	// do the upper edge
	t = imgIn[0] * filterR;
	for(int y = 0; y < (filterR + 1); y++)
	{
		t += imgIn[y * wid];
	}

	imgOut[0] = 1.0 * t * scale;

	for(int y = 1; y < (filterR + 1); y++)
	{
		t += imgIn[(y + filterR) * wid];
		t -= imgIn[0];
		imgOut[y * wid] = t * scale;
	}

	// main loop
	for(int y = filterR + 1; y < hei - filterR; y++)
	{
		t += imgIn[(y + filterR) * wid];
		t -= imgIn[(y - filterR - 1) * wid];
		imgOut[y * wid] = t * scale;
	}

	// do the bottom dege
	for(int y = hei - filterR; y < hei; y++)
	{
		t += imgIn[(hei - 1) * wid];
		t -= imgIn[(y - filterR - 1) * wid];
		imgOut[y * wid] = t * scale;
	}
}

__global__ void d_boxfilter_x_global(float *Out, float *In, int wid, int hei, int filterR)
{
	unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
	if( y >= hei)
		return ;
	d_boxfilter_x(&Out[y * wid], &In[y * wid], wid, hei, filterR);
}

__global__ void d_boxfilter_y_global(float *Out, float *In, int wid, int hei, int filterR)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= wid)
		return ;

	d_boxfilter_y(&Out[x], &In[x], wid, hei, filterR);
}

__global__ void elemwiseSub_kernel(float *out, float *inA, float *inB, int wid, int hei)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	if(idx >= wid || idy >= hei)
		return ;

	int offset = idy * wid + idx;
	out[offset] = inA[offset] - inB[offset];
}

void BFilter::boxfilter(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR)
{
	int nthreads = 512;

	float *d_temp;
	cudaCheckError(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));

	cudaCheckError(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

	dim3 threadPerBlock(nthreads, 1);
	dim3 blockPerGrid;
	blockPerGrid.x = (hei + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;

	// only one iteration
	d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_temp, d_imgIn, wid, hei, filterR);
	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;
	d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut, d_temp, wid, hei, filterR);
	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	cudaCheckError(cudaDeviceSynchronize());

	if(d_temp)
		cudaFree(d_temp);
}

void BFilter::boxfilterTest(float *imgOut, float *imgIn, int wid, int hei, int filterR)
{
	int nthreads = 512;

	float *d_temp;
	cudaCheckError(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));

	cudaCheckError(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

	float *d_imgIn, *d_imgOut;
	cudaCheckError(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
	cudaCheckError(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));

	dim3 threadPerBlock(nthreads, 1);
	dim3 blockPerGrid;
	blockPerGrid.x = (hei + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;

	// only one iteration
	d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_temp, d_imgIn, wid, hei, filterR);
	//d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, filterR);
	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;
	d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut, d_temp, wid, hei, filterR);
	//d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, filterR);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;


	cudaCheckError(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
	cudaCheckError(cudaDeviceSynchronize());

	if(d_temp)
		cudaFree(d_temp);
	if(d_imgIn)
		cudaFree(d_imgIn);
	if(d_imgOut)
		cudaFree(d_imgOut);
}

/*
void BFilter::boxfilterTest(float *imgOut, float *imgIn, int wid, int hei, int filterR)
{
	int nthreads = 512;

	float *d_temp;
	cudaCheckError(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));

	cudaCheckError(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

	cudaCheckError(cudaMemcpy(d_imgIn_, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

	dim3 threadPerBlock(nthreads, 1);
	dim3 blockPerGrid;
	blockPerGrid.x = (hei + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;

	// only one iteration
	d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_temp, d_imgIn_, wid, hei, filterR);
	//d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, filterR);
	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;
	d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_temp, wid, hei, filterR);
	//d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, filterR);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;


	cudaCheckError(cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
	cudaCheckError(cudaDeviceSynchronize());

	if(d_temp)
		cudaFree(d_temp);
}
*/

// Version 1 : not consider about the global memory coalesce
void TScale::twoscaleTest(float *imgOutA, float *imgOutB, float *imgIn, int wid, int hei, int filterR)
{
	float *d_imgIn, *d_imgOutA, *d_imgOutB;
	cudaCheckError(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
	cudaCheckError(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMalloc((void **)&d_imgOutA, sizeof(float) * wid * hei));
	cudaCheckError(cudaMalloc((void **)&d_imgOutB, sizeof(float) * wid * hei));

	// get the low pass coefficients
	boxfilter(d_imgOutB, d_imgIn, wid, hei, filterR);

	// get the high pass coefficients
	dim3 threadPerBlock(BLK_SZ, BLK_SZ);
	dim3 blockPerGrid;
	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

	elemwiseSub_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOutA, d_imgIn, d_imgOutB, wid, hei);
	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	cudaCheckError(cudaMemcpy(imgOutB, d_imgOutB, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
	cudaCheckError(cudaMemcpy(imgOutA, d_imgOutA, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
}
