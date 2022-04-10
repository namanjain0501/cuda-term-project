#include<bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
#define M 2
#define R 3

__global__ void apply_winograd(float ****img, float ****kernels, int h, int w, int c, int n, int k, int r, int m, float ****output)
{
    int P = n*ceil((double)h/m)*ceil((double)w/m);  // Number of image tiles
    int alpha = m+r-1;
    int h_o = h - r + 1;
    int w_o = w - r + 1;

   float **G = (float **) malloc(4*(sizeof(float*))); 
    for (int i = 0; i < 4; i++)
    {
        G[i] = (float *)malloc(3*(sizeof(float )));
        for(int j = 0 ; j < 3; j++ ) {
            G[i][j] = G_[i][j] ; 
        }
    }
    float **GT = (float **) malloc(3*(sizeof(float*))); 
    for (int i = 0; i < 3; i++)
    {
        GT[i] = (float *)malloc(4*(sizeof(float )));
        for(int j = 0 ; j < 4; j++ ) {
            GT[i][j] = GT_[i][j] ; 
        }
    }

    float **A = (float **) malloc(4*(sizeof(float*))); 
    for (int i = 0; i < 4; i++)
    {
        A[i] = (float *)malloc(2*(sizeof(float )));
        for(int j = 0 ; j < 2; j++ ) {
            A[i][j] = A_[i][j] ; 
        }
    }
    float **AT = (float **) malloc(2*(sizeof(float*))); 
    for (int i = 0; i < 2; i++)
    {
        AT[i] = (float *)malloc(4*(sizeof(float )));
        for(int j = 0 ; j < 4; j++ ) {
            AT[i][j] = AT_[i][j] ; 
        }
    }
    
    float **B = (float **) malloc(4*(sizeof(float*))); 
    for (int i = 0; i < 4; i++)
    {
        B[i] = (float *)malloc(4*(sizeof(float )));
        for(int j = 0 ; j < 4; j++ ) {
            B[i][j] = B_[i][j] ; 
        }
    }
    float **BT = (float **) malloc(4*(sizeof(float*))); 
    for (int i = 0; i < 4; i++)
    {
        BT[i] = (float *)malloc(4*(sizeof(float )));
        for(int j = 0 ; j < 4; j++ ) {
            BT[i][j] = BT_[i][j] ; 
        }
    }
    
    float ****Us = (float ****)malloc(k*(sizeof(float***))); 
    for (int i = 0; i < k; i++)
    {
        Us[i] = (float ***)malloc(c*(sizeof(float **)));
    }
    for(int i = 0 ; i  < k ; i++ ) {
        for(int  j = 0 ; j < c ; j++ ) {
            Us[i][j] = U (alpha, r , G , GT , kernels[i][j] ); 
        }
    }
    float ****Vs = (float ****)malloc(c*(sizeof(float***))); 
    for (int i = 0; i < c; i++)
    {
        Vs[i] = (float ***)malloc(P*(sizeof(float **)));
    }
     
    for(int i = 0 ; i  < c ; i++ ) {
        for(int  j = 0 ; j < P - 1 ; j++ ) {
            Vs[i][j] = V (alpha, B , BT , img[i][j] ); 
        }
    }
     float ****Y = (float ****)malloc(k*(sizeof(float***))); 
    for (int i = 0; i < k; i++)
    {
        Y[i] = (float ***)malloc(P*(sizeof(float **)));
        for(int j=0;j<P-1 ;j++)
        {
            Y[i][j] = (float **)malloc((m)*sizeof(float *));
            for(int x=0;x<m;x++){
                Y[i][j][x] = (float *)malloc((m)*sizeof(float));
                for(int y = 0 ; y < m ; y++ ) {
                    Y[i][j][x][y] = 0 ; 
                }
            }
        }
    }
    
    for(int i = 0 ; i < k ; i++ ) {
        for(int j = 0 ; j < P-1 ; j++ ) {
            for(int x = 0 ; x < c ; x++ ) {
                float ** mid = matrix_mult(Us[i][x] , alpha ,  alpha , Vs[x][j] , alpha ) ; 
                float ** left = matrix_mult(AT , m , alpha , mid , alpha ) ; //
                float ** Y_  = matrix_mult(left , m , alpha , A , m ) ;
                for(int ii = 0 ; ii < m ; ii++ ) {
                    for(int jj = 0 ; jj < m ; jj++ ) {
                        Y[i][j][ii][jj] += Y_[ii][jj] ; 
                    }
                }
            }
        }
    }

    for(int i = 0 ; i < k ; i++ ) {
        for(int j = 0 ; j < P -1 ; j++ ) {
            cout << "Y[" << i << "]["<< j<<"] : \n" ; 
                for(int ii = 0 ; ii < m ; ii++ ) {
                    for(int jj = 0 ; jj < m ; jj++ ) {
                        cout << Y[i][j][ii][jj] << " " ; 
                    }
                    cout << endl ; 
                }
            
        }
    }

}

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

    int h1;
    int h2;
    h1 = ceil(h_o/m);
    h2 = ceil(w_o/m);

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

