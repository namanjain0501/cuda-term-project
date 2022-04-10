#include<bits/stdc++.h>
#include "kernel/conv.cu"

using namespace std;

int main()
{
    int n,h,w,c,k,r,s;
    cout<<"input in order n h w c k r s"<<endl;
    cin>>n>>h>>w>>c>>k>>r>>s;

    cout<<"input taken"<<endl;

    // creating img of dim n * h * w * c  and randomly initialising it
    float ****img = (float ****)malloc(n*(sizeof(float***))); 
    for (int i = 0; i < n; i++)
    {
        img[i] = (float ***)malloc(h*(sizeof(float **)));
        for(int j=0;j<h;j++)
        {
            img[i][j] = (float **)malloc(w*sizeof(float *));
            for(int x=0;x<w;x++)
            {
                img[i][j][x] = (float *)malloc(c*sizeof(float));
                for(int y=0;y<c;y++)
                {
                    img[i][j][x][y] = (float)(rand()%256);
                }
            }
        }
    }

    cout<<"img declared"<<endl;

    // creating kernel of dim k * r * s * c  and randomly initialising it
    float ****kernels = (float ****)malloc(k*(sizeof(float***))); 
    for (int i = 0; i < k; i++)
    {
        kernels[i] = (float ***)malloc(r*(sizeof(float **)));
        for(int j=0;j<r;j++)
        {
            kernels[i][j] = (float **)malloc(s*sizeof(float *));
            for(int x=0;x<s;x++)
            {
                kernels[i][j][x] = (float *)malloc(c*sizeof(float));
                for(int y=0;y<c;y++)
                {
                    kernels[i][j][x][y] = (float)rand()/(RAND_MAX);
                }
            }
        }
    }

    cout<<"kernel declared"<<endl;
    
    float ****output = forward_pass(img, kernels, h, w, c, n, k, r, s);

    cout<<"output: "<<output[0][0][0][0]<<endl;

    cout<<"forward_pass executed"<<endl;

    float output_cpu[n][k][h-r+1][w-s+1];

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<k;j++)
        {
            for(int x=0;x<(h-r+1);x++)
            {
                for(int y=0;y<(w-s+1);y++)
                {
                    for(int x1=x;x1<(x+r);x1++)
                    {
                        for(int y1=y;y1<(y+s);y1++)
                        {
                            for(int z1=0;z1<c;z1++)
                            {
                                output_cpu[i][j][x][y] += img[i][x1][y1][z1]*kernels[j][x1-x][y1-y][z1];
                            }
                        }
                    }
                    if(fabs(output_cpu[i][j][x][y]-output[i][j][x][y])>1e-4)
                    {   
                            cout<<"error"<<endl;
                            exit(0);
                    }
                }
            }
        }
    }
    cout<<"success"<<endl;

    return 0;
}
