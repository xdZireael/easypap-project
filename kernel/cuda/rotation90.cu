#include "cuda_kernels.cuh"
#include "cppdefs.h"

EXTERN __global__ void rotation90_kernel_cuda(unsigned *image, unsigned *alt_image, unsigned DIM) {
  unsigned i = get_i ();
  unsigned j = get_j ();
  next_img(DIM - i - 1, j) = cur_img(j, i);
}