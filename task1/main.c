#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define MAX_ITER 1000

typedef struct {
  double x, y;
} Point;

bool is_in_mandelbrot(double complex c) {
  double complex z = 0 + 0 * I;
  for (int i = 0; i < MAX_ITER; i++) {
    z = z * z + c;
    if (cabs(z) >= 2.0) return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  int threads_count = atoi(argv[1]);
  int points_count = atoi(argv[2]);

  omp_set_num_threads(threads_count);

  double x_min = -2.0;
  double x_max = 1.0;
  double y_min = -1.5;
  double y_max = 1.5;

  Point *mandelbrot_points = (Point*)malloc(points_count * sizeof(Point));
  int points_found = 0;

  double start_time = omp_get_wtime();
  #pragma omp parallel
  {
    srand(time(NULL));

    Point *local_points = (Point*)malloc(points_count * sizeof(Point));
    int local_count = 0;

    #pragma omp for
    for (int i = 0; i < points_count; i++) {
      double x = (double)rand() / RAND_MAX * (x_max - x_min) + x_min;
      double y = (double)rand() / RAND_MAX * (y_max - y_min) + y_min;

      double complex c = x + y * I;
      if (is_in_mandelbrot(c)) {
        local_points[local_count].x = x;
        local_points[local_count].y = y;
        local_count++;
      }
    }

    #pragma omp critical
    {
      for (int i = 0; i < local_count; i++) {
        mandelbrot_points[points_found++] = local_points[i];
      }
    }

    free(local_points);
  }

  double end_time = omp_get_wtime();
  double execution_time = end_time - start_time;

  char filename[100];
  sprintf(filename, "mandelbrot_%dthreads_%dpoints.csv", threads_count, points_count);

  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    printf("Ошибка создания файла\n");
    free(mandelbrot_points);
    return 1;
  }

  fprintf(file, "X,Y\n");

  for (int i = 0; i < points_found; i++) {
    fprintf(file, "%.15f,%.15f\n", mandelbrot_points[i].x, mandelbrot_points[i].y);
  }

  fclose(file);

  printf("========================================\n");
  printf("Program Execution Results:\n");
  printf("========================================\n");
  printf("Number of threads: %d\n", threads_count);
  printf("Total generated points: %d\n", points_count);
  printf("Points in Mandelbrot set: %d (%.2f%%)\n",
         points_found, (double)points_found / points_count * 100);
  printf("Execution time: %.4f seconds\n", execution_time);
  printf("Output file: %s\n", filename);
  printf("========================================\n");

  free(mandelbrot_points);
  return 0;
}