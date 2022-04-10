#include <bits/stdc++.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

using namespace std;
#define FractionsInG 0
#define FractionsInA 1
#define FractionsInB 2
#define FractionsInF 3

void apply_winograd(float ****img, float ***kernels,int h,int w,int n,int k,int r,float ****output);
float **transpose (float **A, int m, int n);
float **matrix_mult (float **A, int m, int n, float **B, int r);

// Transposes matrix A of dimensions (m x n)
// Returns transpose of dimensions (n x m)
int cnt = 1  ; 
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

float G_[4][3] = {{1,0,0} , {0.5, 0.5 , 0.5 } , {0.5 , -0.5 , 0.5 } , {0 , 0 , 1 }} ; 
float A_[4][2] = {{1,0} , {1,1} , {1,-1} , {0,-1}} ; 
float BT_[4][4] = {{1,0,-1,0} , {0,1,1,0} , {0,-1,1,0} , {0,-1,0,1}} ; 
float GT_[3][4] = {{1,0.5,0.5,0} , {0 , 0.5, -0.5 , 0 } , {0, 0.5 , 0.5 , 1 } } ;
float AT_[2][4] = {{1,1,1,0} , {0 , 1,-1,-1} } ; 
float B_[4][4] ={{1,0,0,0} , {0,1,-1,-1} , {-1,1,1,0} , {0,0,0,1}};
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
    
