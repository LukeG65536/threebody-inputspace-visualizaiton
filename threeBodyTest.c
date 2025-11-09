#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define N 3
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

v2* getAccels(struct bodyState *bodys);
v2 getAccel(struct bodyState b1, struct bodyState b2);
v2 v2Add(v2 a, v2 b);
v2 v2Scale(v2 a, double s);
v2 v2Sub(v2 a, v2 b);
double v2Magnitude(v2 a);
void updateBodys(struct bodyState *bodys, double dt);


int main() 
{

    struct bodyState *bodys;
    
    bodys = (struct bodyState*)malloc(sizeof(struct bodyState) * N);


    for (int i = 0; i < N; i++)
    {
        bodys[i].accel.x = 0;
        bodys[i].accel.y = 0;
        bodys[i].vel.x = 0;
        bodys[i].vel.y = 0;
    }
    bodys[0].pos.x = -1;
    bodys[0].pos.y = 0;
    bodys[1].pos.x = 0;
    bodys[1].pos.y = 0;
    bodys[2].pos.x = 1;
    bodys[2].pos.y = 0;
   
    updateBodys(bodys, 0.01);
    printf("%f\n", bodys[0].vel.x);

    free(bodys);

    return 0;
     
}

void getColor(struct bodyState *bodys)
{
    double d1 = v2Magnitude(v2Sub(bodys[0].pos, bodys[1].pos));
    double d2 = v2Magnitude(v2Sub(bodys[1].pos, bodys[2].pos));
    double d3 = v2Magnitude(v2Sub(bodys[0].pos, bodys[2].pos));
    double max = fmax(fmax(d1, d2), d3);
    double r = 255 * d1/max;
    double g = 255 * d2/max;
    double b = 255 * d3/max;
}

void updateBodys(struct bodyState *bodys, double dt)
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

v2* getAccels(struct bodyState *bodys)
{

    v2 *accels = calloc(N, sizeof(v2));
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
v2 getAccel(struct bodyState b1, struct bodyState b2)
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

v2 v2Add(v2 one, v2 two)
{
    v2 ret;
    ret.x = one.x + two.x;
    ret.y = one.y + two.y;
    return ret;
}

v2 v2Sub(v2 one, v2 two)
{
    return v2Add(one, v2Scale(two, -1));
}

double v2Magnitude(v2 one)
{
    return sqrt(one.x*one.x + one.y*one.y);
}

v2 v2Scale(v2 one, double scale)
{
    v2 ret;
    ret.x = one.x * scale;
    ret.y = one.y * scale;
    return ret;
}