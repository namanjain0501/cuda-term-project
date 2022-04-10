

#include <bits/stdc++.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

using namespace std;
#define FractionsInG 0
#define FractionsInA 1
#define FractionsInB 2
#define FractionsInF 3

void apply_winograd(float ****img, float ***kernels,int h,int w,int n,int k,int r,int s,float ****output);
float **transpose (float **A, int m, int n);
float **matrix_mult (float **A, int m, int n, float **B, int r);

// filter size - r*r*c
// img size - h*w*c
// batch size - n
// no. of filters - k
// // output size - ??
// void forward_pass(float ****img, float ****kernels, int h, int w, int c, int n, int k, int r, int r, int m )
// {
//     int h_o = h - r + 1;
//     int w_o = w - s + 1;

//     float ****output = (float ****)malloc(n*sizeof(float ***));
//     for (int i = 0; i < n; i++)
//     {
//         output[i] = (float ***)malloc(k*sizeof(float **));
//         for (int j = 0; j < k; j++)
//         {
            
//             output[i][j] = (float **)malloc(h_o*sizeof(float *));
//             for (int x = 0; x < h_o; x++)
//             {
//                 output[i][j][x] = (float *)malloc(w_o*sizeof(float ));   
//             }
//         }
//     }

//     cudaError_t err = cudaSuccess;
//     float *d_img = NULL;
//     size_t size = n*h*w*c* sizeof(float);
//     err = cudaMalloc((void **)&d_img, size);

//     if(err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate img memory(error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to copy img vector from host to device (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     float *d_kernels = NULL;
//     size = k*r*s*c * sizeof(float);
//     err = cudaMalloc((void **)&d_kernels, size)

//     if(err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate kernels memory(error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaMemcpy(d_kernels, kernels, size, cudaMemcpyHostToDevice);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to copy kernels vector from host to device (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     float *d_outputs = NULL;
//     size = n*k*h_o*w_o * sizeof(float);
//     err = cudaMalloc((void **)&d_outputs, size);

//     if(err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate output memory(error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // h_o * w_o * k
//     dim3 grid((h_o+31)/32, (w_o+31)/32, k);
//     dim3 block(32, 32, 1);
//     apply_winograd<<<grid, block>>>(d_img, d_kernels, h, w, n, k, r, s, d_outputs);
//     err = cudaGetLastError();

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to launch apply winograd kernel (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     size = n*k*h_o*w_o * sizeof(float);
//     err = cudaMemcpy(output, d_outputs, size, cudaMemcpyDeviceToHost);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Free device global memory
//     err = cudaFree(d_img);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to free device img vector (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaFree(d_kernels);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to free device kernel vector (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaFree(d_outputs);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to free device output vector (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     return output;
// }

// Output Matrix At : m x n
// void def_At (float *a, int m, int n, float **At) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0;  j< n; j++) {
//             At[i][j] = pow(a[i], j);
//         }
//     }
// }

// // Output Matrix A : m x n
// void def_A (float *a, int m, int n, float **A) {
//     def_At (a, m-1, n, A);
//     for (int j = 0; j < n; j++) {
//         A[m-1][j] = (j==n-1)?1:0;
//     }
// }

// // Output Matrix T : n x (n+1)
// void def_T (float *a, int n, float **T) {
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             T[i][j] = (i==j)?1:0;
//         }
//     }
//     for (int i = 0; i < n; i++) {
//         T[i][n] = pow(-a[i], n);
//     }
// }

// // Output Matrix Lx : n x 1
// void def_Lx (float *a, int n, float **Lx) {
//     for (int i = 0; i < n; i++ ){
//         // TODO
        
//     }
// }

// // Output Matrix F : n x 1
// void def_F (float *a, int n, float **F) {
//     for (int i = 0; i < n; i++) {
//         F[i][0] = 1;
//         for (int j = 0; j < n; j++) {
//             F[i][0] *= (j!=i)?(a[i]-a[j]):1;
//         }
//     }
// }

// // Output Matrix Fdiag : n x n
// void def_Fdiag (float *a, int n, float **Fdiag) {
//     float **f = (float**) malloc (n*sizeof(float*));
//     for (int i = 0; i < n; i++) {
//         f[i] = (float*) malloc (n*sizeof(float));
//     }

//     def_F(a, n, f);

//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             Fdiag[i][j] = (i==j)?f[i][0]:0;
//         }
//     }
// }

// // Output Matrix FdiagPlus1 : n x n
// void def_FdiagPlus1 (float *a, int n, float **FdiagPlus1) {
//     def_Fdiag (a, n-1, FdiagPlus1);

//     for (int i = 0; i < n; i++) {
//         FdiagPlus1[i][n-1] = 0;
//     }

//     for (int j = 0; j < n; j++) {
//         FdiagPlus1[n-1][j] = (j==n-1)?1:0;
//     }
// }

// // Output Matrix L : n x n
// void def_L (float *a, int n, float **L) {
//     // TODO

// }

// // Output Matrix Bt : n x (n+1)
// void def_Bt (float *a, int n, float **Bt) {
//     float **L = (float**) malloc (n*sizeof(float*));
//     for (int i = 0; i < n; i++) {
//         L[i] = (float*) malloc (n*sizeof(float));
//     }
//     def_L (a, n, L);

//     float **T = (float**) malloc (n*sizeof(float*));
//     for (int i = 0; i < n; i++) {
//           T[i] = (float*) malloc ((n+1)*sizeof(float));
//     }
//     def_T (a, n, T);

//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n+1; j++) {
//             Bt[i][j] = 0
//             for (int k = 0; k < n+1; k++) {
//                 Bt[i][j] += L[i][k]*T[k][j];
//             }
//         }
//     }
// }

// // Output Matrix B : n x n
// void def_B (float *a, int n, float **B) {
//     def_B (a, n-1, B);

//     for (int j = 0; j < n; j++) {
//         B[n-1][j] = (j==n-1)?1:0;
//     }
// }

// Transposes matrix A of dimensions (m x n)
// Returns transpose of dimensions (n x m)
float **transpose (float **A, int m, int n) {
    float **T = (float**) malloc (n*sizeof(float*));
    for (int i = 0; i < n; i++) {
        T[i] = (float*) malloc (m*sizeof(float));
        for (int j = 0; j < m; j++) {
            T[i][j] = A[j][i];
        }
    }
    return T;
}

// Multiplies matrix A of dimensions (m x n) with matrix B of dimensions (n x r)
// Returns matrix of dimensions (m x r)
float **matrix_mult (float **A, int m, int n, float **B, int r) {
    float **M = (float**) malloc (m*sizeof(float*));
    for (int i = 0; i < m; i++) {
        M[i] = (float*) malloc (r*sizeof(float));
        for (int j = 0; j < r; j++) {
            M[i][j] = 0;
            for (int k = 0; k < n; k++) {
                M[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return M;
}

// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
// void getCofactor(float **A, float **temp, int N, int p, int q, int n) {
//     int i = 0, j = 0;
 
//     // Looping for each element of the matrix
//     for (int row = 0; row < n; row++)
//     {
//         for (int col = 0; col < n; col++)
//         {
//             //  Copying into temporary matrix only those element
//             //  which are not in given row and column
//             if (row != p && col != q)
//             {
//                 temp[i][j++] = A[row][col];
 
//                 // Row is filled, so increase row index and
//                 // reset col index
//                 if (j == n - 1)
//                 {
//                     j = 0;
//                     i++;
//                 }
//             }
//         }
//     }
// }
 
// // Recursive function for finding determinant of matrix A of dimension N x N
// // n is current dimension of A
// float determinant(float **A, int N, int n) {
//     float D = 0; // Initialize result
 
//     //  Base case : if matrix contains single element
//     if (n == 1)
//         return A[0][0];
 
//     float **temp; // To store cofactors
//     temp = (float**) malloc (N*sizeof(float*));
//     for (int i = 0; i < N; i++) {
//         temp[i] = (float*) malloc (N*sizeof(float));
//     }
 
//     int sign = 1;  // To store sign multiplier
 
//      // Iterate for each element of first row
//     for (int f = 0; f < n; f++)
//     {
//         // Getting Cofactor of A[0][f]
//         getCofactor(A, temp, N, 0, f, n);
//         D += sign * A[0][f] * determinant(temp, N, n - 1);
 
//         // terms are to be added with alternate sign
//         sign = -sign;
//     }
 
//     return D;
// }
 
// // Function to get adjoint of matrix A of size N x N in adj matrix.
// void adjoint(float **A, float **adj, int N) {
//     if (N == 1) {
//         adj[0][0] = 1;
//         return;
//     }
 
//     // temp is used to store cofactors of A[][]
//     int sign = 1;
//     float **temp = (float**) malloc (N*sizeof(float*));
//     for (int i = 0; i < N; i++) {
//         temp[i] = (float*) malloc (N*sizeof(float));
//     }
 
//     for (int i = 0; i < N; i++)
//     {
//         for (int j  =0; j < N; j++)
//         {
//             // Get cofactor of A[i][j]
//             getCofactor(A, temp, N, i, j, N);
 
//             // sign of adj[j][i] positive if sum of row
//             // and column indexes is even.
//             sign = ((i+j)%2==0)? 1: -1;
 
//             // Interchanging rows and columns to get the
//             // transpose of the cofactor matrix
//             adj[j][i] = (sign)*(determinant(temp, N, N-1));
//         }
//     }
// }
 
// // Function to calculate and store inverse, returns false if
// // matrix is singular
// bool inverse(float **A, float **inverse, int N) {
//     // Find determinant of A[][]
//     float det = determinant(A, N, N);
//     if (det == 0) {
//         return false;
//     }
 
//     // Find adjoint
//     float **adj = (float**) malloc (N*sizeof(float*));
//     for (int i = 0; i < n; i++) {
//         adj[i] = (float*)  malloc (N*sizeof(float));
//     }
//     adjoint(A, adj, N);
 
//     // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             inverse[i][j] = adj[i][j]/det;
//         }
//     }
 
//     return true;
// }

// // Populates AT, G, BT and f
// void findMatrices (float *a, int n, int r, float **AT, float **G, float **BT, float **f, int fractionsIn = FractionsInG) {
//     int alpha = n+r-1;
//     float **f = (float**) malloc (alpha*sizeof(float*));
//     for (int i = 0; i < alpha; i++) {
//         f[i] = (float*) malloc (alpha*sizeof(float));
//     }
//     def_FdiagPlus1 (a, alpha, f);

//     if (f[0][0] < 0) {
//         for (int j = 0; j < alpha; j++) {
//             f[0][j] *= -1;
//         }
//     }

//     float **AT, **BT, **G;
//     if (fractionsIn == FractionsInG) {
//         float **A = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             A[i] = (float*) malloc (n*sizeof(float));
//         }
//         def_A (a, alpha, n, A);
//         AT = transpose(A, alpha, n);

//         float **X = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             X[i] = (float*) malloc (r*sizeof(float));
//         }
//         def_A (a, alpha, r, X);
//         float **XT = transpose (X, alpha, r);
//         float **fT = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             fT[i] = (float*) malloc (alpha*sizeof(float));
//         }
//         inverse(f, fT, alpha);
//         float **Y = matrix_mult (XT, r, alpha, fT, alpha);
//         G = transpose (Y, r, alpha);

//         float **Z = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             Z[i] = (float*) malloc (alpha*sizeof(float));
//         }
//         def_B (a, alpha, Z);
//         float **ZT = transpose (Z, alpha, alpha);
//         BT = matrix_mult (f, alpha, alpha, ZT, alpha)

//     } else if (fractionsIn == FractionsInA) {
//         float **X = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             X[i] = (float*) malloc (alpha*sizeof(float));
//         }
//         def_B (a, alpha, X);
//         float **XT = transpose (X, alpha, alpha);
//         BT = matrix_mult (f, alpha, alpha, XT, alpha);

//         G = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             G[i] = (float*) malloc (r*sizeof(float));
//         }
//         def_A (a, alpha, r, G);

//         float **Y = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             Y[i] = (float*) malloc (n*sizeof(float));
//         }
//         def_A (a, alpha, n, Y);
//         float **YT = transpose (Y, alpha, n);
//         float **fT = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             fT[i] = (float*) malloc (alpha*sizeof(float));
//         }
//         inverse(f, fT, alpha);
//         AT = matrix_mult (YT, n, alpha, fT, alpha);

//     } else if (fractionsIn = FractionsInB) {
//         float **A = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             A[i] = (float*) malloc (n*sizeof(float));
//         }
//         def_A (a, alpha, n, A);
//         AT = transpose(A, alpha, n);

//         G = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             G[i] = (float*) malloc (r*sizeof(float));
//         }
//         def_A (a, alpha, r, G);

//         float **B = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             B[i] = (float*) malloc (alpha*sizeof(float));
//         }
//         def_B (a, alpha, B);
//         BT = transpose (B, alpha, alpha);

//     } else {
//         float **A = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             A[i] = (float*) malloc (n*sizeof(float));
//         }
//         def_A (a, alpha, n, A);
//         AT = transpose(A, alpha, n);

//         G = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             G[i] = (float*) malloc (r*sizeof(float));
//         }
//         def_A (a, alpha, r, G);

//         float **X = (float**) malloc (alpha*sizeof(float*));
//         for (int i = 0; i < alpha; i++) {
//             X[i] = (float*) malloc (alpha*sizeof(float));
//         }
//         def_B (a, alpha, X);
//         float **XT = transpose (X, alpha, alpha);
//         BT = matrix_mult (f, alpha, alpha, XT, alpha)
//     }
// }
float G_[4][3] = {{1,0,0} , {0.5, 0.5 , 0.5 } , {0.5 , -0.5 , 0.5 } , {0 , 0 , 1 }} ; 
float A_[4][2] = {{1,0} , {1,1} , {1,-1} , {0,-1}} ; 
float BT_[4][4] = {{1,0,-1,0} , {0,1,1,0} , {0,-1,1,0} , {0,1,0,-1}} ; 
float GT_[3][4] = {{1,0.5,0.5,0} , {0 , 0.5, -0.5 , 0 } , {0, 0.5 , 0.5 , 1 } } ;
float AT_[2][4] = {{1,1,1,0} , {0 , 1,-1,-1} } ; 
float B_[4][4] ={{1,0,0,0} , {0,1,-1,1} , {-1,1,1,0} , {0,0,0,-1}};
float ** U(int alpha , int r , float  **G , float  **GT , float ** filter) {
    
 float ** temp = matrix_mult (G,  alpha,  r, filter, r) ; 
 float ** temp1 = matrix_mult(temp , alpha , r , GT , alpha ) ; 
 return temp1 ; 
}
float ** V(int alpha  , float ** B , float ** BT , float ** tile) {
    
 float ** temp3 = matrix_mult(BT , alpha , alpha , tile , alpha) ; 
 float ** temp4 = matrix_mult(temp3 ,alpha, alpha , B, alpha ) ; 
 return temp4 ; 
}
int compute(float ** AT , float ** A , float **BT , float **B ,float **G , float ** GT , float ** filter , float ** tile,int m , int r)  
{
   // Y = AT[[GgGt][BTdB]]A 
 int alpha = m+r-1 ; 
 float ** temp = matrix_mult (G,  alpha,  r, filter, r) ; 
 float ** temp1 = matrix_mult(temp , alpha , r , GT , alpha ) ; 

 // TO BE CHECKED
  float ** temp3 = matrix_mult(BT , alpha , alpha , tile , alpha) ; 
 float ** temp4 = matrix_mult(temp3 ,alpha, alpha , B, alpha ) ; 
 //
 float ** mid = matrix_mult(temp1 , alpha ,  alpha , temp4 , alpha ) ; 
 float ** left = matrix_mult(AT , m , alpha , mid , alpha ) ; 
 float ** Y  = matrix_mult(left , m , alpha , A , m ) ; 
 int ans = 0 ; 
 
 for(int i = 0 ; i < m ; i++ ) {
     for(int j = 0 ; j < m ; j++ ) {
         ans += Y[i][j] ;
     }
 }
 return ans ; 
}
void apply_winograd(float ****img, float ****kernels, int h, int w, int c, int n, int k, int r, int m, float ****output)
{
    int P = n*ceil(h/m)*ceil(w/m);  // Number of image tiles
    int alpha = m+r-1;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            // TODO
        }
    }
    // TODO
    /*int h_o = h - r + 1;
    int w_o = w - s + 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x; //height
    int y = blockDim.y * blockIdx.y + threadIdx.y; //width
    int z = blockDim.z * blockIdx.z + threadIdx.z; // kernel no.
 
    if(x<h_o && y<w_o && z<k)
    {
        for(int i=0;i<n;i++)
        {
            output[i][z][x][y] = 0;
            for(int x1=x;x1<(x+r);x1++)
            {
                for(int y1=y;y1<(y+s);y1++)
                {
                    for(int z1=0;z1<c;z1++)
                    {
                        output[i][z][x][y] += img[i][x1][y1][z1] * kernels[z][x1-x][y1-y][z1];
                    }
                }
            }
        }
    }*/
//     float *a ; 
     
//    float **AT;
//    float **G ; 
   
//    float **BT ; 
//    float **f ;
//  // findMatrices (float *a, int n, int r, float **AT, float **G, float **BT, float **f, int fractionsIn = FractionsInG) {
//     findMatriceS(a , n , r , AT , G , BT , f) ; 
    
    
     

    // float ** GT = transpose(G,alpha , r ) ; 

    // // DISPERENCY CHECK ONCE 
    // float ** AT = transpose(A , alpha , m  ) ; 
    // //
    // float ** B = transpose(BT , alpha , alpha ) ; 

    // Y = AT[[GgGt][BTdB]]A 
    // g = filter d = tile 
    int h_o = h - r + 1;
    int w_o = w - r + 1;
    // dim3 grid((h_o+31)/32, (w_o+31)/32, k);
    // dim3 block(32, 32, 1);
    // compute(AT,A, BT , B , G , GT , filter , tile , m , r ) ; 

//     G_[4][3] = {{1,0,0} , {0.5, 0.5 , 0.5 } , {0.5 , -0.5 , 0.5 } , {0 , 0 , 1 }} ; 
// float A_[4][2] = {{1,0} , {1,1} , {1,-1} , {0,-1}} ; 
// float BT_[4][4] = {{1,0,-1,0} , {0,1,1,0} , {0,-1,1,0} , {0,1,0,-1}} ; 
// float GT_[3][4] = {{1,0.5,0.5,0} , {0 , 0.5, -0.5 , 0 } , {0, 0.5 , 0.5 , 1 } } ;
// float AT_[2][4] = {{1,1,1,0} , {0 , 1,-1,-1} } ; 
// float B_[4][4] ={{1,0,0,0} , {0,1,-1,1} , {-1,1,1,0} , {0,0,0,-1}};
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
        for(int  j = 0 ; j < P ; j++ ) {
            Vs[i][j] = V (alpha, B , BT , img[i][j] ); 
        }
    }
     float ****Y = (float ****)malloc(k*(sizeof(float***))); 
    for (int i = 0; i < k; i++)
    {
        Y[i] = (float ***)malloc(P*(sizeof(float **)));
        for(int j=0;j<P;j++)
        {
            Y[i][j] = (float **)malloc((m)*sizeof(float *));
            for(int x=0;x<m;x++){
                Y[i][j][x] = (float *)malloc((m)*sizeof(float));
            }
        {
    }
    for(int i = 0 ; i < k ; i++ ) {
        for(int j = 0 ; j < P ; j++ ) {
            
            float ** left = matrix_mult(AT , m , alpha , mid , alpha ) ; 
            float ** Y  = matrix_mult(left , m , alpha , A , m ) ; 
        }
    }
    // compute<<<grid, block>>>(d_img, d_kernels, h, w, n, k, r, s, d_outputs);
}
    
