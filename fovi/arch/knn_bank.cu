#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T> __device__ float read_value(T x) { return (float)x; }
template <typename T> __device__ T write_value(float x) { return (T)x; }

// One owner per output: no atomics, no batch partitioning, FP32 reductions.
template <typename T>
__global__ void project(const T* x, const long long* levels,
                        const int* ptr, const int* ind, T* z,
                        int B, int C, int N, int P, int V, int stride) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= B*P*C*V) return;
    int v = q % V, c = (q/V) % C, p = (q/(V*C)) % P;
    int b = q/(P*C*V), level = (int)levels[b];
    int row = level*(P*V+1) + p*V+v;
    float sum = 0;
    for (int j=ptr[row]; j<ptr[row+1]; ++j)
        sum += read_value(x[(b*C+c)*N + ind[level*stride+j]]);
    z[q] = write_value<T>(sum);
}

// Transposed incidence lists make input gradients deterministic as well.
template <typename T>
__global__ void project_backward(const T* dz, const long long* levels,
                                 const int* ptr, const int* ind, T* dx,
                                 int B, int C, int N, int P, int V, int stride) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= B*C*N) return;
    int n=q%N, c=(q/N)%C, b=q/(N*C), level=(int)levels[b];
    int row=level*(N+1)+n;
    float sum=0;
    for (int j=ptr[row]; j<ptr[row+1]; ++j) {
        int dest=ind[level*stride+j], p=dest/V, v=dest%V;
        sum += read_value(dz[((b*P+p)*C+c)*V+v]);
    }
    dx[q]=write_value<T>(sum);
}
