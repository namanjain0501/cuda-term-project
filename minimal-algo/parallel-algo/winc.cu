#include<bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
#define M 2
#define R 3

__global__ void apply_conv(float *img, float *kernels, int h, int w, int c, int n, int k, int r, int s, float *output)
{
    int h_o = h - r + 1;
    int w_o = w - s + 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x; //height
    int y = blockDim.y * blockIdx.y + threadIdx.y; //width
    int z = blockDim.z * blockIdx.z + threadIdx.z; // kernel no.
 
    if(x<h_o && y<w_o && z<k)
    {
        for(int i=0;i<n;i++)
        {
            // output[i][z][x][y] = 0;
            output[i*k*h_o*w_o + z*h_o*w_o + x*w_o + y] = 0;
            for(int x1=x;x1<(x+r);x1++)
            {
                for(int y1=y;y1<(y+s);y1++)
                {
                    for(int z1=0;z1<c;z1++)
                    {
                        output[i*k*h_o*w_o + z*h_o*w_o + x*w_o + y] += img[i*h*w*c + x1*w*c + y1*c + z1] * kernels[z*r*s*c + (x1-x)*s*c + (y1-y)*c + z1];
                    }
                }
            }
        }
    }
    
}

// filter size - r*s*c
// img size - h*w*c
// batch size - n
// no. of filters - k
// output size - n*k*(h-r+1)*(w-s+1)
float *forward_pass(float *img, float *kernels, int h, int w, int c, int n, int k, int r, int s)
{
    int h_o = h - r + 1;
    int w_o = w - s + 1;
    int m, r;

    m = M;
    r  = R;


    size_t size = n*k*h_o*w_o;
    float *output = (float *)malloc(size*sizeof(float));

    cudaError_t err = cudaSuccess;
    float *d_img = NULL;
    size = n*h*w*c* sizeof(float);
    err = cudaMalloc((void **)&d_img, size);

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate img memory(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy img vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_kernels = NULL;
    size = k*r*s*c * sizeof(float);
    err = cudaMalloc((void **)&d_kernels, size);

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate kernels memory(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_kernels, kernels, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy kernels vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_outputs = NULL;
    size = n*k*h_o*w_o * sizeof(float);
    err = cudaMalloc((void **)&d_outputs, size);

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate output memory(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    // h_o * w_o * k
    dim3 grid((h_o+31)/32, (w_o+31)/32, k);
    dim3 block(32, 32, 1);

    apply_conv<<<grid, block>>>(d_img, d_kernels, h, w, c, n, k, r, s, d_outputs);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch apply conv kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = n*k*h_o*w_o * sizeof(float);
    err = cudaMemcpy(output, d_outputs, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_img);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device img vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_kernels);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device kernel vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_outputs);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return output;
}

