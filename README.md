# 快速笔记
## CUDA
- https://zhuanlan.zhihu.com/p/426978026
- https://developer.download.nvidia.cn/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
- 概念
    - Tensor Core 专用于加速深度学习模型训练和推理的计算单元，可在一个时钟周期内执行矩阵乘法和累加，提高计算效率
    - nvlink用于不同gpu之间通讯的总线以及通讯协议，传输带宽远高于pcie
    - pcie用于host和device之间的数据传输
- 常规编程流程
```C++
// 定义核函数
__global__
void add(int n, float *x, float *y) {
// Do add
}

// 主函数体
float *x, *y;
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));

// Run kernel on 1M elements on the GPU
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);

// wait gpu finish
cudaDeviceSynchronize();

cudaFree()
```

- 优化策略
1. 合理划分网格和块（需要query所用gpu情况进行参考）
2. 避免分支散列和条件判断，使同一warp中的所有线程执行相同代码路径 if (threadIdx.x < 64) {do} ，避免每个线程的频繁判断和执行命令切换 if (sdata[threadIdx.x] > 100) { do } else { do }
3. 使用共享内存（注意显卡支持的block内共享内存大小）
```C++
__global__ void kernel(float *d_data){
    __shared__ float s_data[1024];
```
4. 合并内存访问：遵循访存的空间局部性原理，保证内存访问时对齐且连续
```C++
__global__ void kernel(float *d_data) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    float value = d_data[tid];
}

__global__ void kernel(float *d_data) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float value = d_data[tid*stride];
}
```
5. 使用合适的内存类型：比如将常用的只读常量放到常量内存或者纹理内存中
```C++
__constant__ float const_data[256];
__global__ void kernel() {
    float data = const_data[threadIdx.x];
}
```
6. 减少Bank冲突：Bank conflict 发生在多个线程在同一时钟周期内访问共享内存的同一个bank，同一个bank在一个时钟周期内只能处理一个访问请求，所以出现bank冲突时会被串行化处理，从而降低内存访问效率。例如，共享内存被分为32个Bank，当前有一个线程块，里面有32个线程，也就是一个warp, 有如下代码：
```C++
unsigned int tid=threadIdx.x;
for(int s = 1; s<256;s++){
    int index = 2*tid;
    sdata[index] += sdata[index+1];
}
```
显然，线程0访问0号和1号地址，线程16访问32号和33号地址，此处出现明显bank0冲突

### reduce 优化
- https://zhuanlan.zhihu.com/p/426978026
- 优化思路：base算子 -> fix warp divergence 统一wrap到一个逻辑分支 -> 解决 bank conflict -> 减少idle线程 -> 展开最后一次迭代 -> 完全展开for -> 调整block-num -> 使用shuffle
- 在固定每个block的thread数量和数据总规模的情况下，block-num数量越多，并发量越高，但是同一时间处理的数据量也就越大；反之减小block-num以减小并发量，同一时间处理的数据量也变小。并行参数的选择是一个trade off
- base code（核函数参考zhihu链接）
```C++
#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
__global__ void reduce(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    // each thread loads one element from global to shared mem
    ...
}

bool check(float *out, float *res, int n) {
    for(int i = 0; i < n; i++){
        if(out[i] != res[i])
            return false;
    }
    return true;
}

bool print_array(float *arr, int n) {
    for(int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *a = (float*)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    int NUM_PER_BLOCK = THREAD_PER_BLOCK;
    // int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK;  // for reduce 3
    int block_num = N / NUM_PER_BLOCK;
    float *out = (float *)malloc(block_num * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (block_num * sizeof(float)));
    float *res = (float *)malloc(block_num * sizeof(float));

    for(int i = 0; i < N; i++) {
        a[i] = 1;
    }
    for(int i = 0; i < block_num; i++) {
        float cur = 0;
        for(int j = 0; j < NUM_PER_BLOCK; ++j) {
            cur += a[i * NUM_PER_BLOCK + j];
        }
        res[i] = cur;
    }
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce<<<Grid,Block>>>(d_a, d_out);
    cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out, res, block_num)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < block_num; i++){
            printf("%lf ", out[i]);
        }
        printf("\n");
    }
    // print_array(out, block_num);
    // print_array(res, block_num);
    cudaFree(d_a);
    cudaFree(d_out);
}
```
### 深蓝L2 矩阵乘法
- C = A * B, (A[M, K], B[K, N], C[M, N]), 计算流程
```shell
for i in [0, M)  # A的第i行，行中第k个，即第k列 (i, k)
    for j in [0, N)  # B的第j列，列中第k个，即第k行 (k, j)
        sum = 0
        for k in [0, K)
            sum += A[i * K + k] + B[k * N + j]
    C[i * N + j] = sum  # C(i, j) = sum
```
- CUDA上的矩阵乘法：在base的基础上，利用shared mem和分块矩阵，实现减少内存拷贝的目的，分块矩阵可以在通过shared_mem读取小块矩阵以减少访存时间
```shell
/* Accumulate C tile by tile. */
for tileIdx = 0 to (K/blockDim.x - 1) do
    /* Load one tile of A and one tile of B into shared mem */
    i <= blockIdx.y * blockDim.y + threadIdx.y // Row i of matrix A
    j <= tileIdx * blockDim.x + threadIdx.x // Column j of matrix A
    A_tile(threadIdx.y, threadIdx.x) <= A_gpu(i,j) // Load A(i,j) to shared mem
    B_tile(threadIdx.x, threadIdx.y) <= B_gpu(j,i) // Load B(j,i) to shared mem
    __sync() // Synchronize before computation

    /* Accumulate one tile of C from tiles of A and B in shared mem */
    for k = 0 to threadDim.x do
        accu <= accu + A_tile(threadIdx.y,k) * B_tile(k,threadIdx.x)
    end
    __sync()
end
```
- 设使用a*a维度小矩阵作为分块，for循环进行 **ceil(K / a)** 次计算，**row 或者 col = tile_idx * a + idx**
- 把小分块用shared memory保存
```C++
for (int i = 0; i < (int)(ceil((float)numAColumns / BLOCK_SIZE)); i++) {
    if (i*BLOCK_SIZE + tx < numAColumns && row < numARows)
        sharedM[ty][tx] = A[row*numAColumns + i * BLOCK_SIZE + tx];
    else
        sharedM[ty][tx] = 0.0;

    if (i*BLOCK_SIZE + ty < numBRows && col < numBColumns)
        sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*numBColumns + col];
    else
        sharedN[ty][tx] = 0.0;
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; j++) {
        Csub += sharedM[ty][j] * sharedN[j][tx];
    }
    __syncthreads();
}
```
- 原始矩阵乘法，全局访存2mnk次
- 分块平铺乘法：全局访存 2 * d_block * d_block * [m * n * k / d_block**3] = 2 * m * n * k / d_block
- 理论加速比d_block倍
### 深蓝L3 cuda stream & cuda event
- https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
- 流特征
    - 单线程内，默认流的执行是同步的，显式流的执行是异步的
    - 单线程内，编译加上--default-stream per-thread，默认/显式流的执行均为异步
    - 多线程，默认所有线程共享一个默认流
    - 编译加上--default-stream per-thread，每个线程都有一个默认流
- CUDA中的显式同步按粒度可以分为四类
    - device synchronize 影响很大
    - stream synchronize 影响单个流和CPU
    - event synchronize 影响CPU，更细粒度的同步
    - synchronizing across streams using an event
- 使用cuda event可以进行细粒度的流同步（类似future、promise）
    - cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event)
    - 指定 stream 等待特定的 event，该 event 可以关联到相同或者不同的 stream
- one thread stream
```C++
#include <iostream>
#include "cuda_runtime.h"

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}
```
- multithread stream
```C++
#include <pthread.h>
#include <stdio.h>
#include "cuda_runtime.h"

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void *launch_kernel(void *dummy)
{
    float *data;
    cudaMalloc(&data, N * sizeof(float));

    kernel<<<1, 64>>>(data, N);

    cudaStreamSynchronize(0);

    return NULL;
}

int main()
{
    const int num_threads = 8;

    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, launch_kernel, 0)) {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }

    cudaDeviceReset();

    return 0;
}
```
### 深蓝L4 cublas/cudnn/卷积
- cublas
```C++
// 准备 A, B, C 以及使用的线程网格、线程块的尺寸
// 创建句柄
cublasHandle_t handle;
cublasStatus_t cublasCreate(&handle);
// 调用计算函数
cublasStatus_t cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, *B, n, *A, k, &beta, *C, n);
// 销毁句柄
cublasStatus_t cublasDestroy(handle);
// 回收计算结果，顺序可以和销毁句柄交换
```
- cublasSgemm接口
    - https://blog.csdn.net/u011197534/article/details/78378536 测试用代码
    - https://www.cnblogs.com/cuancuancuanhao/p/7763256.html cublasSgemm接口存储型形式与转置的讲解
        - 关于主维 leading dimension。如果我们想要计算 Am×k Bk×n = Cm×n，那 m、n、k 三个参数已经固定了所有尺寸，为什么还要一组主维参数呢？看完了上面部分我们发现，输入的矩阵 A、B 在整个计算过程中会发生变形    ，包括行列优先变换和转置变换，所以需要一组参数告诉该函数，矩阵变形后应该具有什么样的尺寸。参考 CUDA 的教程 CUDA Toolkit Documentation v9.0.176，对这几个参数的说明比较到位。
        - 当参数 transa 为 CUBLAS_OP_N 时，矩阵 A 在 cuBLAS 中的尺寸实际为 lda × k，此时要求 lda ≥ max(1, m)（否则由该函数直接报错，输出全零的结果）
        - 当参数 transa 为 CUBLAS_OP_T 或 CUBLAS_OP_C 时，矩阵 A 在 cuBLAS 中的尺寸实际为 lda × m，此时要求 lda ≥ max(1, k) 。
        - transb 为 CUBLAS_OP_N 时，B 尺寸为 ldb × n，要求 ldb ≥ max(1, k)
        - transb 为 CUBLAS_OP_T 或 CUBLAS_OP_C 时，B尺寸为 ldb × k，此时要求 ldb ≥ max(1, n)
        - C 尺寸为 ldc × n，要求 ldc ≥ max(1, m)。
        - 可见，是否需要该函数帮我们转置矩阵 A 会影响 A 在函数中的存储。而主维正是在这一过程中起作用。
        - 以Am*k为例子，当 lda * k > m * k，cublas会给矩阵多出的行补0
    - cublas中矩阵的存储是列优先，开辟一段连续的内存，放入1,2,3,4,5,6,7,8,9，指定矩阵行和列均为3，则可表示矩阵[1,2,3 ; 4,5,6 ; 7,8,9]，然而，在使用cublas时，这样表示出来的矩阵应该是[1,4,7; 2,5,8; 3,6,9]
    - CUBLAS_OP_参数决定矩阵是否转置，即决定该矩阵是按照行优先还是列优先。当我们选择CUBLAS_OP_N 时表示不转置，按列优先存储；当我们选择CUBLAS_OP_T时表示需要转置，按行优先存储。
        - **CUBLAS_OP_参数只能决定A、B，作为结果的C一定是按照 m*n 的形式输出的**
    - 如果前边的参数是 CUBLAS_OP_T，那么leading dimesion 就是矩阵的列数，此时的矩阵是按照C语言以行优先的方式来存储的
    - 如果前边的参数是 CUBLAS_OP_N，那么leading dimesion 就是矩阵的行数，此时的矩阵保持CUBLAS的列优先存储方式
    - 对于 C = A X B, 有 C^T = B^T X A^T
```C++
cublasStatus_t cublasSgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha, const float *A, int lda, const float *B, int ldb,
    const float *beta, float*C, int ldc)
// 实现功能： C = alpha * op ( A ) * op ( B ) + beta * C
// 参数意义
// alpha和beta是标量， A B C是以列优先存储的矩阵
// 如果 transa 的参数是CUBLAS_OP_N 则op(A) = A ，如果是CUBLAS_OP_T 则op(A)=A的转置
// 如果 transb的参数是CUBLAS_OP_N 则op(B) = B ，如果是CUBLAS_OP_T 则op(B)=B的转置
// Lda/Ldb:A/B的leading dimension，若转置按行优先，则leading dimension为A/B的列数
// Ldc：C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数

// 定义测试矩阵的维度
int const A_ROW = 2;
int const A_COL = 3;
int const B_ROW = 3;
int const B_COL = 4;
```
- cudnn实现卷积
    - https://www.hbblog.cn/cuda%E7%9B%B8%E5%85%B3/2022%E5%B9%B407%E6%9C%8823%E6%97%A5%2023%E6%97%B617%E5%88%8614%E7%A7%92/ cuDNN API的使用与测试-以二维卷积+Relu激活函数为例
    - https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/index.html cudnn 8.9.7版本文档
    - conv2D cudnn C++ 代码 & python 代码
    - 关于 cudnnSetConvolution2dDescriptor 的参数  CUDNN_CROSS_CORRELATION 和 CUDNN_CONVOLUTION 的区别
        - 知乎参考 https://zhuanlan.zhihu.com/p/33194385
        - pytorch中卷积计算默认走的 CROSS_CORRELATION
```C++
// cudnn version: cudnn8.9.7
// nvcc conv_cudnn.cu -o conv_cudnn -lcudnn
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

#define CUDA_CALL(f) { \
  cudaError_t err = (f);  \
  if (err != cudaSuccess) { \
    std::cout \
      << "    Error occurred: " << err << std::end; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
      << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 2)
    px[tid] = k;
  else
    px[tid] = -k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid + 1;
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
    buffer.data(), data,
    n * c * h * w * sizeof(float),
    cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // iuput
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  // filter
  const int filt_k = 1;
  const int filtc = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  float *h_in_data = (float*)malloc(sizeof(float) * in_h * in_w);
  float *h_filt_data = (float*)malloc(sizeof(float) * filt_h * filt_w);

  // set matrix value
  for (int i = 0; i < in_h * in_w; i++) {
    h_in_data[i] = (float)(i + 1);
  }
  h_filt_data[0] = 1.0;
  h_filt_data[1] = 1.0;
  h_filt_data[2] = -1.0;
  h_filt_data[3] = -1.0;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));
  // cudaMemcpy(in_data, h_in_data, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyHostToDevice)

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));
  // cudaMemcpy(filt_data, h_filt_data, sizeof(float) * filt_k * filt_c * filt_h * filt_w, cudaMemcpyHostToDevice);

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std:cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    pad_h, pad_w, str_h, str_w, dil_h, dil_w,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
    conv_desc, in_desc, filt_desc,
    &out_n, &out_c, &out_h, &out_w));
  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std:cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgoPerf_t algo;
  int ret_cnt = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        1, &ret_cnt, &algo));
  std::cout << "Convolution algorithm: " << algo.algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo.algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_data, ws_size));
  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo.algo, ws_data, ws_size,
      &beta, out_desc, out_data));

  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w);
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnnDestroy(cudnn));
  return 0;
}
```
```python
import numpy as np
import torch.nn.functional as F
import torch

def test():
    input_tensor = torch.tensor([[[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]]]).float()
    weight_tensor = torch.tensor([[[[1,1],[-1,-1]]]]).float()
    # 关键方法 F.conv2d
    out_tensor_torch = F.conv2d(input_tensor, weight_tensor, stride=1, padding=1)
    print(input_tensor)
    print(weight_tensor)
    print(out_tensor_torch)

if __name__ == "__main__":
    test()
```
### 深蓝L5 TensorRt
- TensorRt核心是TRT，闭源
- TRT转换优化引擎 (离线，生成engine/model)
    - 1. 创建Builder
    - 2. 创建Network
    - 3. 使用API or Parser 构建network
    - 4. 优化网络
- TRT执行引擎 (在线)
    - 5. 序列化和反序列化模型
    - 6. 传输计算数据（host->device）
    - 7. 执行计算
    - 8. 传输计算结果（device->host）
- TensorRT模型转换
    - ONNX ：https://github.com/NVIDIA/TensorRT/tree/main/parsers
    - Pytorch：https://github.com/NVIDIA-AI-IOT/torch2trt
    - TensorFlow：https://github.com/tensorflow/tensorflow/tree/1cca70b80504474402215d2a4e55bc44621b691d/tensorflow/compiler/tf2tensorrt
    - Tencent Forward：https://github.com/Tencent/Forward
- SampleMnist
    - 可以使用官方的例子：TensorRT-8.XX/samples/sampleMnist
    - 自己编译，走cmake
    - 将官方的 sample_mnist.cpp 和依赖的 data common 目录放到自己路径
    - 要求配置好 cuda cudnn tensorrt
```cmake
PROJECT(SampleMnist)
cmake_minimum_required(VERSION 3.12)

SET(common_dir ${CMAKE_SOURCE_DIR}/../common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda/targets/x86_64-linux/include)
INCLUDE_DIRECTORIES(${cuda_dir})

set(ONNX_PARSE /usr/local/TensorRT/include)
INCLUDE_DIRECTORIES(${ONNX_PARSE})

SET(LOG_CPP ${common_dir}/logger.cpp)

SET(execute_name sample_mnist_test)
ADD_EXECUTABLE(${execute_name} sampleOnnxMNIST.cpp ${LOG_CPP})

find_library(LIBONNX_PATH nvonnxparser /usr/local/TensorRT/lib)
TARGET_LINK_LIBRARIES(${execute_name} ${LIBONNX_PATH})

find_library(LIBNVINFER nvinfer /usr/local/TensorRT/lib)
TARGET_LINK_LIBRARIES(${execute_name} ${LIBNVINFER})

find_library(LIBCUDART cudart /usr/local/cuda/lib64)
TARGET_LINK_LIBRARIES(${execute_name} ${LIBCUDART})
```
- 可视化 onnx
```shell
pip3 install netron
netron --host IP -p port mnist.onnx
```

### Transformers
- Transformers快速入门 https://transformers.run/

### 深蓝L6 Bert/Vit优化
- 下载 bert-base-uncased
    - https://huggingface.co/google-bert/bert-base-uncased/tree/main
    - 简单使用 https://blog.csdn.net/weixin_38481963/article/details/110535583
    - 需要安装 python的 transformers 库
```python
from transformers import BertTokenizer, BertModel

BERT_PATH="./bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertModel.from_pretrained(BERT_PATH, return_dict = True)
print(tokenizer.tokenize('I have a good time, thank you.'))

inputs = tokenizer("I have a good time, thank you.", return_tensors="pt")
outputs = model(**inputs)
print(inputs)
print("=========")
print(outputs)
```
