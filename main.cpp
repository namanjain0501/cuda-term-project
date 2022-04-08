#include<bits/stdc++.h>

using namespace std;

int main()
{
    int n,h,w,c,k,r,s;
    cout<<"input in order n h w c k r s"<<endl;
    cin>>n>>h>>w>>c>>k>>r>>s;

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
                    img[i][j][x][y] = (float)rand()%256;
                }
            }
        }
    }

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

    float ****output = forward_pass(img, kernels, h, w, c, n, k, r, s);

    return 0;
}
