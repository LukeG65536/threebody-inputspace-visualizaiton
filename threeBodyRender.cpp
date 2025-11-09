#include "hip/hip_runtime.h"
#include <stdio.h>
#include <cassert>
#include <iostream>
#define N 3

using std::cout;

struct pixel {
	unsigned char r,g,b;	
};

 
typedef struct vector2
{
    double x, y;
} v2;

struct bodyState 
{
    struct vector2 pos;
    struct vector2 vel;
    struct vector2 accel;
};

__device__ vector2* getAccels(bodyState *bodys);
__device__ vector2 getAccel(bodyState b1, bodyState b2);
__device__ vector2 v2Add(vector2 a, vector2 b);
__device__ vector2 v2Sub(v2 one, v2 two);
__device__ vector2 v2Scale(vector2 a, double s);
__device__ double v2Magnitude(vector2 a);
__device__ pixel getColorMax(bodyState *bodys);
__device__ pixel getColorMin(bodyState *bodys);
__device__ void updateBodys(bodyState *bodys, double dt);
__global__ void drawImg(pixel* img, bodyState *systems, vector2 *viewWindow, int wid, int ht, double dt, double time);
#define widd 5000
#define timee 10

int main()
{
    
    const int threadPBlock = 256;
    const int width = widd;
    const int height = widd;
    const int numPixel = width * height;
    vector2 *h_viewWindow = new vector2[2];
    vector2 *d_viewWindow;
    h_viewWindow[0] = {-5,-5};
    h_viewWindow[1] = {5,5};



    pixel *h_img = (pixel*) malloc(sizeof(pixel) * numPixel);
    pixel *d_img;

    bodyState *initState = new bodyState[N];
    initState[0] = {{-1,0},{0,-1},{0,0}};
    initState[1] = {{0,0},{0,0},{0,0}};
    initState[2] = {{1,0},{0,1},{0,0}};

    bodyState *h_systems = (bodyState*)malloc(numPixel * sizeof(bodyState) * N);
    bodyState *d_systems;


    for (int i = 0; i < numPixel; ++i) {
        for (int j = 0; j < N; ++j) {
            h_systems[i*N + j] = initState[j];
        }
    }

    

    hipMalloc(&d_img, sizeof(pixel) * width * height);
    hipMalloc(&d_systems, sizeof(bodyState) * N * numPixel);
    hipMalloc(&d_viewWindow, sizeof(vector2) * 2);

    
    hipMemcpy(d_systems, h_systems, sizeof(bodyState) * N * numPixel, hipMemcpyHostToDevice);
    hipMemcpy(d_viewWindow, h_viewWindow, sizeof(vector2) * 2, hipMemcpyHostToDevice);



    drawImg<<<numPixel/threadPBlock, threadPBlock>>>(d_img, d_systems, d_viewWindow, width, height, .01, timee);

    
    hipMemcpy(h_img, d_img, sizeof(pixel) * numPixel, hipMemcpyDeviceToHost);
    printf("writting to file");

    FILE *fp = fopen("output3.ppm", "wb");
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

__global__ void drawImg(pixel* img, bodyState *systems, vector2 *viewWindow, int wid, int ht, double dt, double time)
{
//assuming that systems is already assigned 
    int numPix = wid * ht;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * N;
    if (idx >= numPix) return;
    int xScreenPos = idx%wid;
    int yScreenPos = idx/wid;

    bodyState local[N];
    for (int j = 0; j < N; ++j) local[j] = systems[base + j];

    double xNorm = xScreenPos / (double)wid;
    double yNorm = yScreenPos / (double)ht;
    double xDist = viewWindow[1].x - viewWindow[0].x;
    double yDist = viewWindow[1].y - viewWindow[0].y;
    local[1].vel.x = xDist * xNorm + viewWindow[0].x;
    local[1].vel.y = yDist * yNorm + viewWindow[0].y;

    int numTicks = (int)(time/dt);

    for (int i = 0; i < numTicks; i++)
    {
        updateBodys(local, dt);
    }
    
    img[idx] = getColorMax(local);
    
}


__device__ pixel getColorMin(struct bodyState *bodys)
{
    double d1 = v2Magnitude(v2Sub(bodys[0].pos, bodys[1].pos));
    double d2 = v2Magnitude(v2Sub(bodys[1].pos, bodys[2].pos));
    double d3 = v2Magnitude(v2Sub(bodys[0].pos, bodys[2].pos));
    double min = fmin(fmin(d1, d2), d3);
    pixel p;
    p.r = 255 * min/d1;
    p.g = 255 * min/d2;
    p.b = 255 * min/d3;
    return p;
}
__device__ pixel getColorMax(struct bodyState *bodys)
{
    double d1 = v2Magnitude(v2Sub(bodys[0].pos, bodys[1].pos));
    double d2 = v2Magnitude(v2Sub(bodys[1].pos, bodys[2].pos));
    double d3 = v2Magnitude(v2Sub(bodys[0].pos, bodys[2].pos));
    double max = fmax(fmax(d1, d2), d3);
    pixel p;
    p.r = 255 * d1/max;
    p.g = 255 * d2/max;
    p.b = 255 * d3/max;
    return p;
}

__device__ void updateBodys(struct bodyState *bodys, double dt)
{
//trying velocity verlet https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    for (int i = 0; i < N; i++)
    {
        bodys[i].pos.x += bodys[i].vel.x * dt + bodys[i].accel.x * dt * dt * .5;
        bodys[i].pos.y += bodys[i].vel.y * dt + bodys[i].accel.y * dt * dt * .5;
    }
    v2 *newAccels = getAccels(bodys);
    for (int i = 0; i < N; i++)
    {
        bodys[i].vel.x += (newAccels[i].x+bodys[i].accel.x) * dt * .5;
        bodys[i].vel.y += (newAccels[i].y+bodys[i].accel.y) * dt * .5;
        bodys[i].accel = newAccels[i];
    }
    free(newAccels);
}

__device__ v2* getAccels(struct bodyState *bodys)
{

    v2 *accels = (v2*)malloc(N * sizeof(vector2));

    for (int i = 0; i < N; i++)
    {
        accels[i] = {0,0};
    }
    
   //calculate accels by looping through everu unordered index pair 
    for (int i = 0; i < N-1; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            v2 curAccel = getAccel(bodys[i], bodys[j]);
            accels[i] = v2Add(accels[i], curAccel);
            accels[j] = v2Add(accels[j], v2Scale(curAccel, -1));
        }
    }
    return accels;
}

//returns accel of body1 
__device__ v2 getAccel(struct bodyState b1, struct bodyState b2)
{
    double distx = b2.pos.x - b1.pos.x;
    double disty = b2.pos.y - b1.pos.y;
    double dist = sqrt(distx*distx + disty*disty);
    struct vector2 accel;
    //F is preportional to 1/(dist*dist)
    //to get from dist-->dixt x you multiply dist by (distx/dist)
    //meaning accel = (dixtx/dist)*(1/dist*dist)
    accel.x = (distx/(dist*dist*dist));
    accel.y = (disty/(dist*dist*dist));
    return accel;
    
}

__device__ v2 v2Add(v2 one, v2 two)
{
    v2 ret;
    ret.x = one.x + two.x;
    ret.y = one.y + two.y;
    return ret;
}

__device__ v2 v2Sub(v2 one, v2 two)
{
    return v2Add(one, v2Scale(two, -1));
}

__device__ double v2Magnitude(v2 one)
{
    return sqrt(one.x*one.x + one.y*one.y);
}

__device__ v2 v2Scale(v2 one, double scale)
{
    v2 ret;
    ret.x = one.x * scale;
    ret.y = one.y * scale;
    return ret;
}

