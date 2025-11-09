#include <stdio.h>
#include <cassert>
#include <iostream>


using std::cout;

struct pixel {
	unsigned char r,g,b;	
};


__global__ void drawImg(pixel* img, int wid, int ht, int maxIter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int xScreenPos = idx%wid;
    int yScreenPos = idx/wid;

    float x0 = ((float)xScreenPos / (float)wid)*4-2;
    float y0 = ((float)yScreenPos / (float)ht)*4-2;

    float x = 0;
    float y = 0;

    int i = 0;
    while(x*x+y*y <= 4 && i < maxIter)
    {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        i++;
    }
    int fullyInCheck = (i == maxIter) ? 0 : 1;

    img[idx].b = fullyInCheck*10*255*i/maxIter;
    img[idx].r = 0;
    img[idx].g = fullyInCheck*5*255*i/maxIter;

}

int main()
{
    int threadPBlock = 256;
    int width = 20000;
    int height = 20000;
    int maxItrations = 1000;
    pixel *h_img = (pixel*) malloc(sizeof(pixel) * width * height);
    pixel *d_img;

    cudaMalloc(&d_img, sizeof(pixel) * width * height);

    drawImg<<<width*height/threadPBlock, threadPBlock>>>(d_img, width, height, maxItrations);

    cudaMemcpy(h_img, d_img, sizeof(pixel) * width * height, cudaMemcpyDeviceToHost);

    printf("writting to file");

    FILE *fp = fopen("output.ppm", "wb");
    if (fp == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", 255); 

    for (int i = 0; i < width * height; i++)
    {
        u_int8_t red = (u_int8_t)(h_img[i].r);
        u_int8_t blue = (u_int8_t)(h_img[i].b);
        u_int8_t green = (u_int8_t)(h_img[i].g);

        fwrite(&red, sizeof(u_int8_t), 1, fp);
        fwrite(&green, sizeof(u_int8_t), 1, fp);
        fwrite(&blue, sizeof(u_int8_t), 1, fp);
    }

    fclose(fp);
    
    

    return 0;
}