#ifndef OCL_IS_DEF
#define OCL_IS_DEF

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "error.h"
#include "monitoring.h"
#include "ezp_gpu_event.h"

void ocl_init (int show_config, int silent);
void ocl_build_program (int list_variants);
void ocl_alloc_buffers (void);
void ocl_send_data (void);
void ocl_retrieve_data (void);
void ocl_establish_bindings (void);
void ocl_update_texture (void);
unsigned easypap_number_of_gpus_ocl (void);
size_t ocl_get_max_workgroup_size (void);
const char *ocl_GetError (cl_int error);

#define check(err, format, ...)                                                \
  do {                                                                         \
    if (err != CL_SUCCESS)                                                     \
      exit_with_error (format " [OCL err %d: %s]", ##__VA_ARGS__, err,         \
                       ocl_GetError (err));                                    \
  } while (0)

typedef struct
{
  cl_command_queue q;
  cl_device_id device;
  cl_mem curb, nextb;
  cl_kernel kernel;
} ocl_gpu_t;

extern ocl_gpu_t ocl_gpu[];
extern unsigned ocl_nb_gpus;

extern cl_context context;
extern cl_program program;

#define ocl_queue(gpu) ocl_gpu[gpu].q
#define ocl_device(gpu) ocl_gpu[gpu].device
#define ocl_cur_buffer(gpu) ocl_gpu[gpu].curb
#define ocl_next_buffer(gpu) ocl_gpu[gpu].nextb
#define ocl_compute_kernel(gpu) ocl_gpu[gpu].kernel


#endif
