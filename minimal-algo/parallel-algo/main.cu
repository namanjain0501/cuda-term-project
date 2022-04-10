#include<bits/stdc++.h>
#include "kernel/conv.cu"

using namespace std;

int main()
{
    int n,h,w,c,k,r,s;
    cout<<"input in order n h w c k r s"<<endl;
    cin>>n>>h>>w>>c>>k>>r>>s;

    size_t size = n*h*w*c*sizeof(float);    
    float *img = (float *)malloc(size);

    for(int i=0;i<n*h*w*c;i++)
        img[i]=(float)(rand()%256);

    size = k*r*s*c*sizeof(float);
    float *kernels = (float *)malloc(size);

    for(int i=0;i<k*r*s*c;i++)
        kernels[i]=(float)rand()/(RAND_MAX);

    float *output = forward_pass(img, kernels, h, w, c, n, k, r, s);

    int h_o = h-r+1;
    int w_o = w-s+1;

    float output_cpu[n][k][h_o][w_o];

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<k;j++)
        {
            for(int x=0;x<(h-r+1);x++)
            {
                for(int y=0;y<(w-s+1);y++)
                {
                    output_cpu[i][j][x][y]=0;
                    for(int x1=x;x1<(x+r);x1++)
                    {
                        for(int y1=y;y1<(y+s);y1++)
                        {
                            for(int z1=0;z1<c;z1++)
                            {
                                output_cpu[i][j][x][y] += img[i*h*w*c + x1*w*c + y1*c + z1]*kernels[j*r*s*c + (x1-x)*s*c + (y1-y)*c + z1];
                            }
                        }
                    }
                    if(fabs(output_cpu[i][j][x][y]-output[i*k*h_o*w_o + j*h_o*w_o + x*w_o + y])>1e-2)
                    {   
                        cout<<"expected : "<<output_cpu[i][j][x][y]<<endl;
                        cout<<"got: "<<output[i*k*h_o*w_o + j*h_o*w_o + x*w_o + y]<<endl;
                        cout<<"dim : "<<i<<" "<<j<<" "<<x<<" "<<y<<endl;
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
