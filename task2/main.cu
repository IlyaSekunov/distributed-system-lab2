#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define G 6.67430e-11f
#define DT 3600.0f
#define BLOCK_SIZE 256
#define SOFTENING 1e-10f

typedef struct {
  float mass;
  float x, y, z;
  float vx, vy, vz;
  float ax, ay, az;
} Body;

int read_input(const char *filename, Body *bodies) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Error opening file %s\n", filename);
    exit(1);
  }

  int n;
  if (fscanf(file, "%d", &n) != 1) {
    fprintf(stderr, "Error reading number of bodies\n");
    exit(1);
  }

  for (int i = 0; i < n; i++) {
    double mass, x, y, z, vx, vy, vz;
    if (fscanf(file, "%lf %lf %lf %lf %lf %lf %lf",
               &mass, &x, &y, &z, &vx, &vy, &vz) != 7) {
      fprintf(stderr, "Error reading body %d data\n", i);
      exit(1);
    }
    bodies[i].mass = (float)mass;
    bodies[i].x = (float)x;
    bodies[i].y = (float)y;
    bodies[i].z = (float)z;
    bodies[i].vx = (float)vx;
    bodies[i].vy = (float)vy;
    bodies[i].vz = (float)vz;
    bodies[i].ax = 0.0f;
    bodies[i].ay = 0.0f;
    bodies[i].az = 0.0f;
  }

  fclose(file);
  return n;
}

__global__ void compute_accelerations_kernel(Body *bodies, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  float ax = 0.0f, ay = 0.0f, az = 0.0f;
  float xi = bodies[i].x;
  float yi = bodies[i].y;
  float zi = bodies[i].z;

  for (int j = 0; j < n; j++) {
    if (i != j) {
      float dx = bodies[j].x - xi;
      float dy = bodies[j].y - yi;
      float dz = bodies[j].z - zi;

      float dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
      float dist = sqrtf(dist_sq);
      float inv_dist3 = 1.0f / (dist_sq * dist);

      float force = G * bodies[j].mass * inv_dist3;

      ax += force * dx;
      ay += force * dy;
      az += force * dz;
    }
  }

  bodies[i].ax = ax;
  bodies[i].ay = ay;
  bodies[i].az = az;
}

__global__ void update_positions_kernel(Body *bodies, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  bodies[i].vx += bodies[i].ax * dt;
  bodies[i].vy += bodies[i].ay * dt;
  bodies[i].vz += bodies[i].az * dt;

  bodies[i].x += bodies[i].vx * dt;
  bodies[i].y += bodies[i].vy * dt;
  bodies[i].z += bodies[i].vz * dt;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <tend> <input_file>\n", argv[0]);
    return 1;
  }

  double tend = atof(argv[1]);
  const char *input_filename = argv[2];

  Body bodies[10005];
  int n = read_input(input_filename, bodies);

  FILE *output = fopen("trajectories.csv", "w");
  if (!output) {
    fprintf(stderr, "Error creating output file\n");
    return 1;
  }

  fprintf(output, "t");
  for (int i = 0; i < n; i++) {
    fprintf(output, " x%d y%d", i + 1, i + 1);
  }
  fprintf(output, "\n");

  fprintf(output, "%.6f", 0.0);
  for (int i = 0; i < n; i++) {
    fprintf(output, " %.6e %.6e", bodies[i].x, bodies[i].y);
  }
  fprintf(output, "\n");

  float t = 0.0f;
  int steps = (int)(tend / DT);

  Body *d_bodies;
  cudaMalloc(&d_bodies, n * sizeof(Body));
  cudaMemcpy(d_bodies, bodies, n * sizeof(Body), cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int step = 0; step < steps; step++) {
    compute_accelerations_kernel<<<grid, block>>>(d_bodies, n);
    cudaDeviceSynchronize();

    update_positions_kernel<<<grid, block>>>(d_bodies, n, DT);
    cudaDeviceSynchronize();

    t += DT;

    cudaMemcpy(bodies, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost);

    fprintf(output, "%.6f", t);
    for (int i = 0; i < n; i++) {
      fprintf(output, " %.6e %.6e", bodies[i].x, bodies[i].y);
    }
    fprintf(output, "\n");
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float exec_time_ms;
  cudaEventElapsedTime(&exec_time_ms, start, stop);

  fclose(output);

  printf("Final simulation time: %.2f seconds\n", t);
  printf("Execution time: %.6f ms\n", exec_time_ms);
  printf("Results file: trajectories.csv\n");

  cudaFree(d_bodies);

  return 0;
}