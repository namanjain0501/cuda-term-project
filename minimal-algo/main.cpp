#include<bits/stdc++.h>
#include "winc.cpp"

using namespace std;

int main()
{
    int n,h,w,c,k,r,m;
    cout<<"input in order n h w c k /r m/ "<<endl;
    // cin>>n>>h>>w>>c>>k;
    n =1 , h = 2 , w = 3 , c = 1; k =1 ;
    r = 3 ; m = 2 ; 

    // creating img of dim n * h * w * c  and randomly initialising it
    float ****img = (float ****)malloc(c*(sizeof(float***))); 
    for (int i = 0; i < c; i++)
    {
        img[i] = (float ***)malloc(n*(sizeof(float **)));
        for(int j=0;j<n;j++)
        {
            img[i][j] = (float **)malloc((h+r-1)*sizeof(float *));
            for(int x=0;x<h+r-1;x++)
            {
                img[i][j][x] = (float *)malloc((w+r-1)*sizeof(float));
                for(int y=0;y<w+r-1;y++)
                {
                    if(x < (r-1) /2 || x >= h+(r-1)/2 || y < (r-1)/2 || y  >= w+(r-1)/2 ) {
                        img[i][j][x][y] = 0 ; 
                    }
                    else  img[i][j][x][y] =  (float)(rand()%256);
                }
            }
        }
    }
    cout << "PADDED IMAGE MATRIX : " << endl ; 
    for (int i = 0; i < c; i++)
    {
        for(int j=0;j<n;j++)
        {
            for(int x=0;x<h+r-1;x++)
            {
                for(int y=0;y<w+r-1;y++)
                {
                    cout << img[i][j][x][y] << " " ; 
                }
                cout << endl ; 
            }
        }
    }
    // creating kernel of dim k * r * s * c  and randomly initialising it
    float ****kernels = (float ****)malloc(k*(sizeof(float***))); 
    for (int i = 0; i < k; i++)
    {
        kernels[i] = (float ***)malloc(c*(sizeof(float **)));
        for(int j=0;j<c;j++)
        {
            kernels[i][j] = (float **)malloc(r*sizeof(float *));
            for(int x=0;x<r;x++)
            {
                kernels[i][j][x] = (float *)malloc(r*sizeof(float));
                for(int y=0;y<r;y++)
                {
                    kernels[i][j][x][y] = (float)rand()/(RAND_MAX);
                }
            }
        }
    }
    for(int i = 0 ; i < k ; i++ ) {
        for(int j = 0 ; j < c ; j++ ) {
            cout << "KERNEL : " << endl ; 
            for(int x=0;x<r;x++)
            {
                
                for(int y=0;y<r;y++)
                {
                    cout << kernels[i][j][x][y] << " ";  
                }
                cout << endl ; 
            }
        }
    }
    float ****output ; 
    apply_winograd(img, kernels, h, w, c, n, k, r, m, output);

    return 0;
}
