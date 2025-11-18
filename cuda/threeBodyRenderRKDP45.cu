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
};

struct bodyState 
{
    struct vector2 pos;
    struct vector2 vel;
};

__device__ vector2 getAccel(bodyState b1, bodyState b2);
__device__ vector2 v2Add(vector2 a, vector2 b);
__device__ vector2 v2Sub(vector2 one, vector2 two);
__device__ vector2 v2Scale(vector2 a, double s);
__device__ double v2Magnitude(vector2 a);
__device__ pixel getColor(bodyState *bodys);
__device__ pixel getColor2(bodyState *bodys);

__device__ void getDState(const bodyState *bodys, bodyState* res, vector2* accels);
__device__ void scaleDState(bodyState *dState, bodyState *res, double scale);
__device__ void addDState(bodyState *state, bodyState *dState, double dStateScale, bodyState * res);
__device__ void getAccels(const bodyState *bodys, vector2* accels);
__device__ double tryRKDP45Step(bodyState *bodys, double dt, double tol);

__device__ double updateBodys(bodyState *bodys, double dt, double lastDt);
__global__ void drawImg(pixel* img, bodyState *systems, bodyState *local, vector2 *viewWindow, double *lastDt, int wid, int ht, double time, bool rst);

#define widd 500
#define timee 10
#define tolerance 1e-8
#define minStep 0.0001
#define winSize 2
#define frameRate 5
#define modifying vel

#define color1R 255
#define color1G 000
#define color1B 000

#define color2R 000
#define color2G 255
#define color2B 000

#define color3R 000
#define color3G 000
#define color3B 255

void writeFrame(pixel* img, int num, int width, int height)
{
    char filename[15]; //img0000.ppm
    sprintf(filename, "out/img%4d.ppm", num);

    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", 255); 

    for (int i = 0; i < width * height; i++)
    {
        u_int8_t red = (u_int8_t)(img[i].r);
        u_int8_t blue = (u_int8_t)(img[i].b);
        u_int8_t green = (u_int8_t)(img[i].g);

        fwrite(&red, sizeof(u_int8_t), 1, fp);
        fwrite(&green, sizeof(u_int8_t), 1, fp);
        fwrite(&blue, sizeof(u_int8_t), 1, fp);
    }

    fclose(fp);
}


int main()
{
    //settings for img render
    const int threadPBlock = 256;
    const int width = widd;
    const int height = widd;
    const int numPixel = width * height;
    vector2 *h_viewWindow = new vector2[2];
    vector2 *d_viewWindow;
    h_viewWindow[0] = {-winSize, -winSize};
    h_viewWindow[1] = {winSize, winSize};



    // pixel *h_img = (pixel*) malloc(sizeof(pixel) * numPixel);
    pixel *img;
    //trying unified memory
    
    bodyState *initState = new bodyState[N];
    initState[0] = {{-1,0},{0,0}};
    initState[1] = {{0,0},{0,0}};
    initState[2] = {{1,0},{0,0}};
    
    bodyState *h_systems = (bodyState*)malloc(numPixel * sizeof(bodyState) * N);
    bodyState *d_systems;
    
    bodyState *d_local;
    double *lastDt;
    
    
    for (int i = 0; i < numPixel; ++i) {
        for (int j = 0; j < N; ++j) {
            h_systems[i*N + j] = initState[j]; //making all the systems start at initstate
        }
    }


    //cuda memory stuff
    int id = cudaGetDevice(&id);
    
    cudaMallocManaged(&img, sizeof(pixel) * numPixel);

    // cudaMalloc(&d_img, sizeof(pixel) * width * height);
    cudaMalloc(&d_systems, sizeof(bodyState) * N * numPixel);
    cudaMalloc(&d_viewWindow, sizeof(vector2) * 2);
    cudaMalloc(&d_local, sizeof(bodyState) * N * numPixel);
    cudaMalloc(&lastDt, sizeof(double) * numPixel);
    
    cudaMemcpy(d_systems, h_systems, sizeof(bodyState) * N * numPixel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_viewWindow, h_viewWindow, sizeof(vector2) * 2, cudaMemcpyHostToDevice);
    
    
    // int numFrames = frameRate * timee;
    // double timeStep = 1.0/((double)frameRate);
    // drawImg<<<numPixel/threadPBlock, threadPBlock>>>(img, d_systems, d_local, d_viewWindow, lastDt, width, height, timeStep, true);
    
    // for (int i = 0; i < numFrames; i++)
    // {
    //     drawImg<<<numPixel/threadPBlock, threadPBlock>>>(img, d_systems, d_local, d_viewWindow, lastDt, width, height, timeStep, false);
    //     cudaMemPrefetchAsync(img, sizeof(pixel)*numPixel, cudaCpuDeviceId);
    //     cudaDeviceSynchronize();
    //     writeFrame(img, i, width, height);
    //     printf("wrote frame %4d\n", i);
    // }
        
    drawImg<<<numPixel/threadPBlock, threadPBlock>>>(img, d_systems, d_local, d_viewWindow, lastDt, width, height, timee, true);
    
    cudaMemPrefetchAsync(img, sizeof(pixel)*numPixel, cudaCpuDeviceId);
    cudaDeviceSynchronize();
    writeFrame(img, 67, width, height);


    return 0;
}


__global__ void drawImg(pixel* img, bodyState *systems, bodyState *localp, vector2 *viewWindow, double *lastDt, int wid, int ht, double time, bool rst)
{

// bodyState local[N];

    int numPix = wid * ht;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * N;
    if (idx >= numPix) return;
    bodyState *local = &localp[idx * N];

    if(rst) 
    {
        int xScreenPos = idx%wid;
        int yScreenPos = idx/wid;
        double xNorm = xScreenPos / (double)wid;
        double yNorm = yScreenPos / (double)ht;
        double xDist = viewWindow[1].x - viewWindow[0].x;
        double yDist = viewWindow[1].y - viewWindow[0].y;
        for (int j = 0; j < N; ++j) local[j] = systems[base + j];
        local[1].modifying.x = xDist * xNorm + viewWindow[0].x;
        local[1].modifying.y = yDist * yNorm + viewWindow[0].y;
        lastDt[idx] = 0.01;
    }


    lastDt[idx] = updateBodys(local, time, lastDt[idx]);
    
    img[idx] = getColor2(local);
    
}

__device__ pixel getColor(bodyState *bodys)
{
    double d1 = v2Magnitude(v2Sub(bodys[0].pos, bodys[1].pos));
    double d2 = v2Magnitude(v2Sub(bodys[1].pos, bodys[2].pos));
    double d3 = v2Magnitude(v2Sub(bodys[0].pos, bodys[2].pos));
    double max = fmax(fmax(d1, d2), d3);
    pixel p;
    double w1 = d1/max;
    double w2 = d2/max;
    double w3 = d3/max;
    p.r = sqrt((color1R * color1R * w1 + color2R * color2R * w2 + color3R * color3R * w3)/(w1 + w2 + w3));
    p.g = sqrt((color1G * color1G * w1 + color2G * color2G * w2 + color3G * color3G * w3)/(w1 + w2 + w3));
    p.b = sqrt((color1B * color1B * w1 + color2B * color2B * w2 + color3B * color3B * w3)/(w1 + w2 + w3));
    return p;
}


__device__ pixel getColor2(bodyState *bodys)
{
    double d1 = v2Magnitude(v2Sub(bodys[0].pos, bodys[1].pos));
    double d2 = v2Magnitude(v2Sub(bodys[1].pos, bodys[2].pos));
    double d3 = v2Magnitude(v2Sub(bodys[0].pos, bodys[2].pos));
    double max = fmax(fmax(d1, d2), d3);
    pixel p;
    p.r = 255 * d1/max;
    p.g = 255 * d2/max;
    p.b = 255 * d3/max;
    // double w1 = d1/max;
    // double w2 = d2/max;
    // double w3 = d3/max;
    
    return p;
}



__device__ double updateBodys(bodyState *bodys, double tf, double lastDt)
{

    double time = 0;
    double dt = lastDt;
    while(time <= tf){
        time += dt;
        dt = tryRKDP45Step(bodys, dt, tolerance);
    }
    return dt;
}

__device__ double tryRKDP45Step(bodyState *bodys, double dt, double tol)
{
    bodyState k1[N];
    bodyState k2[N];
    bodyState k3[N];
    bodyState k4[N];
    bodyState k5[N];
    bodyState k6[N];
    bodyState k7[N];

    bodyState y4[N];
    bodyState y5[N];

    vector2 accels[N];
    //magic numbers from https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method 
    //k1 = d/dx(bodys)
    getDState(bodys, k1, accels);
    scaleDState(k1, k1, dt); //k1 *=dt


    //k2 = y+k1*c
    addDState(bodys, k1, 1.0/5.0, k2);
    getDState(k2, k2, accels); //k2 = d/dx(k2)
    scaleDState(k2, k2, dt); //k2 *=dy

    //k3
    addDState(bodys, k1, 3.0/40.0, k3);
    addDState(k3, k2, 9.0/40, k3);
    getDState(k3, k3, accels);
    scaleDState(k3, k3, dt);

    //k4
    addDState(bodys, k1, 44.0/45.0, k4);
    addDState(k4, k2, -56.0/15.0, k4);
    addDState(k4, k3, 32.0/9.0, k4);
    getDState(k4, k4, accels);
    scaleDState(k4, k4, dt);

    //k5
    addDState(bodys, k1, 19372.0/6561.0, k5);
    addDState(k5, k2, -25360.0/2187.0, k5);
    addDState(k5, k3, 64448.0/6561.0, k5);
    addDState(k5, k4, -212.0/729.0, k5);
    getDState(k5, k5, accels);
    scaleDState(k5, k5, dt);

    //k6
    addDState(bodys, k1, 9017.0/3168.0, k6);
    addDState(k6, k2, -355.0/33.0,k6);
    addDState(k6, k3, 46732.0/5247.0,k6);
    addDState(k6, k4, 49.0/176.0,k6);
    addDState(k6, k5, -5103.0/18656.0,k6);
    getDState(k6, k6, accels);
    scaleDState(k6, k6, dt);

    //k7 and y4
    addDState(bodys, k1, 35.0/384.0, y4);
    addDState(y4, k3, 500.0/1113.0, y4);
    addDState(y4, k4, 125.0/192.0, y4);
    addDState(y4, k5, -2187.0/6784.0, y4);
    addDState(y4, k6, 11.0/84.0, y4);
    getDState(y4, k7, accels);
    scaleDState(k7, k7, dt);

    //y5
    addDState(bodys, k1, 5179.0/57600.0, y5);
    addDState(y5, k3, 7571.0/16695.0, y5);
    addDState(y5, k4, 393.0/640.0, y5);
    addDState(y5, k5, -92097.0/339200.0, y5);
    addDState(y5, k6, 187.0/2100.0, y5);
    addDState(y5, k7, 1.0/40.0, y5);

    double maxErr = 0.0;

    for (int i=0;i<N;i++) 
    {
        double dx = y5[i].pos.x - y4[i].pos.x;
        double dy = y5[i].pos.y - y4[i].pos.y;
        double dvx = y5[i].vel.x - y4[i].vel.x;
        double dvy = y5[i].vel.y - y4[i].vel.y;
        double e = sqrt(dx*dx + dy*dy + dvx*dvx + dvy*dvy);
        if (e > maxErr) maxErr = e;
    }
    
    double potential = dt*pow((tol/maxErr), 1.0/5.0); 

    if (tol == -1 || maxErr < tol) //if the error is acceptable apply and slightly increase step size
    {
        for (int i = 0; i < N; i++)
        {
            bodys[i] = y5[i];
        }
        
    
        return potential * 1.1;
        
    }
    
    if(potential < minStep){ //if were getting way to small just force an update and continue
        potential = minStep;
        
        for (int i = 0; i < N; i++)
        {
            bodys[i] = y5[i];
        }
    }
    return potential * .9;
}


__device__ void scaleDState(bodyState *dState, bodyState *res, double scale)
{
    for (int i = 0; i < N; i++)
    {
        res[i].pos.x = dState[i].pos.x * scale;
        res[i].pos.y = dState[i].pos.y * scale;
        res[i].vel.x = dState[i].vel.x * scale;
        res[i].vel.y = dState[i].vel.y * scale;
    }
}

__device__ void addDState(bodyState *state, bodyState *dState, double dStateScale, bodyState * res)
{
    for (int i = 0; i < N; i++)
    {
        res[i].pos.x = state[i].pos.x + dState[i].pos.x * dStateScale;
        res[i].pos.y = state[i].pos.y + dState[i].pos.y * dStateScale;
        res[i].vel.x = state[i].vel.x + dState[i].vel.x * dStateScale;
        res[i].vel.y = state[i].vel.y + dState[i].vel.y * dStateScale;
    }
    
}

__device__ void getDState(const bodyState *bodys, bodyState* res, vector2* accels)
{
    getAccels(bodys, accels);
    for (int i = 0; i < N; i++)
    {
        res[i].pos = bodys[i].vel;
        res[i].vel = accels[i];
    }
    
}

__device__ void getAccels(const bodyState *bodys, vector2* accels)
{
    for (int i = 0; i < N; i++)
    {
        accels[i] = {0,0};
    }
    
   //calculate accels by looping through everu unordered index pair 
    for (int i = 0; i < N-1; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            vector2 curAccel = getAccel(bodys[i], bodys[j]);
            accels[i] = v2Add(accels[i], curAccel);
            accels[j] = v2Add(accels[j], v2Scale(curAccel, -1));
        }
    }
}

//returns accel of body1 
__device__ vector2 getAccel(struct bodyState b1, struct bodyState b2)
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

__device__ vector2 v2Add(vector2 one, vector2 two)
{
    vector2 ret;
    ret.x = one.x + two.x;
    ret.y = one.y + two.y;
    return ret;
}

__device__ vector2 v2Sub(vector2 one, vector2 two)
{
    return v2Add(one, v2Scale(two, -1));
}

__device__ double v2Magnitude(vector2 one)
{
    return sqrt(one.x*one.x + one.y*one.y);
}

__device__ vector2 v2Scale(vector2 one, double scale)
{
    vector2 ret;
    ret.x = one.x * scale;
    ret.y = one.y * scale;
    return ret;
}

