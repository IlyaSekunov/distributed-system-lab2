#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define G 6.67430e-11
#define DT 3600.0

typedef struct {
  double mass;
  double x, y, z;
  double vx, vy, vz;
  double ax, ay, az;
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
    if (fscanf(file, "%lf %lf %lf %lf %lf %lf %lf",
               &bodies[i].mass,
               &bodies[i].x, &bodies[i].y, &bodies[i].z,
               &bodies[i].vx, &bodies[i].vy, &bodies[i].vz) != 7) {
      fprintf(stderr, "Error reading body %d data\n", i);
      exit(1);
    }
    bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;
  }

  fclose(file);
  return n;
}

int main(int argc, char *argv[]) {
  double tend = atof(argv[1]);
  const char *input_filename = argv[2];
  int num_threads = atoi(argv[3]);

  omp_set_num_threads(num_threads);

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

  // Initial state (t = 0)
  fprintf(output, "%.6f", 0.0);
  for (int i = 0; i < n; i++) {
    fprintf(output, " %.6e %.6e", bodies[i].x, bodies[i].y);
  }
  fprintf(output, "\n");

  double t = 0.0;
  int steps = (int)(tend / DT);

  double start_time = omp_get_wtime();

  for (int step = 0; step < steps; step++) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      double ax = 0.0, ay = 0.0, az = 0.0;

      for (int j = 0; j < n; j++) {
        if (i != j) {
          double dx = bodies[j].x - bodies[i].x;
          double dy = bodies[j].y - bodies[i].y;
          double dz = bodies[j].z - bodies[i].z;

          double dist_sq = dx * dx + dy * dy + dz * dz;
          double dist = sqrt(dist_sq);

          if (dist > 1e-10) {
            double force = G * bodies[j].mass / (dist_sq * dist);
            ax += force * dx;
            ay += force * dy;
            az += force * dz;
          }
        }
      }
      bodies[i].ax = ax;
      bodies[i].ay = ay;
      bodies[i].az = az;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      bodies[i].vx += bodies[i].ax * DT;
      bodies[i].vy += bodies[i].ay * DT;
      bodies[i].vz += bodies[i].az * DT;

      bodies[i].x += bodies[i].vx * DT;
      bodies[i].y += bodies[i].vy * DT;
      bodies[i].z += bodies[i].vz * DT;
    }

    t += DT;

    fprintf(output, "%.6f", t);
    for (int i = 0; i < n; i++) {
      fprintf(output, " %.6e %.6e", bodies[i].x, bodies[i].y);
    }
    fprintf(output, "\n");
  }

  double end_time = omp_get_wtime();
  double exec_time = end_time - start_time;

  fclose(output);

  printf("Final simulation time: %.2f seconds\n", t);
  printf("Threads count: %d\n", num_threads);
  printf("Execution time: %.6f ms\n", exec_time * 1000);
  printf("Results file: trajectories.csv\n");

  return 0;
}