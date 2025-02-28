
namespace suggestify
{
namespace kernels
{


static inline __device__ float __tanhf(float x)
{
#if (__CUDA_ARCH__ >= 750)
    float r = x;
    asm("tanh.approx.f32 %0, %0;" : "+f"(r));
    return r;
#else
    return tanhf(x);
#endif
}


}
}
