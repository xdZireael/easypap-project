#ifndef TRACE_DATA_IS_DEF
#define TRACE_DATA_IS_DEF

#include <stdint.h>

#include "list.h"
#include "trace_common.h"
#include "perfcounter.h"
#include "ezv.h"

typedef int64_t perfcounter_array_t[EASYPAP_NB_COUNTERS];

typedef struct
{
  uint64_t start_time, end_time;
  unsigned x, y, w, h;
  int task_type;
  int task_id;
  unsigned iteration;
  perfcounter_array_t counters;
  struct list_head cpu_chain;
} trace_task_t;

typedef struct
{
  uint64_t start_time, end_time;
#ifdef ENABLE_PER_ITERATION_STATS
  perfcounter_array_t *perfcounter_cpu_scores; // counter for each cpu
  perfcounter_array_t perfcounter_scores;      // counter for iteration
#endif
  uint64_t correction, gap;
  struct list_head chain;
  trace_task_t **first_cpu_task;
} trace_iteration_t;

typedef struct
{
  unsigned num;
  unsigned dimensions;
  unsigned nb_cores;
  unsigned nb_gpu;
  unsigned first_iteration;
  unsigned nb_iterations;
  unsigned has_cache_data;
  char *label;
  char *mesh_file;
  ezv_palette_name_t palette;
  char **task_ids;
  unsigned task_ids_count;
  struct list_head *per_cpu;
  trace_iteration_t *iteration;
} trace_t;

#define MAX_TRACES 2

extern trace_t trace[MAX_TRACES];
extern unsigned nb_traces;
extern unsigned trace_data_align_mode;

void trace_data_init (trace_t *tr, unsigned num);
void trace_data_set_nb_threads (trace_t *tr, unsigned nb_cores,
                                unsigned nb_gpu);
void trace_data_set_dim (trace_t *tr, unsigned dim);
void trace_data_set_do_cache (trace_t *tr, unsigned use_cache);
void trace_data_set_first_iteration (trace_t *tr, unsigned it);
void trace_data_set_label (trace_t *tr, char *label);
void trace_data_set_meshfile (trace_t *tr, char *filename);
void trace_data_set_palette (trace_t *tr, ezv_palette_name_t palette);

void trace_data_alloc_task_ids (trace_t *tr, unsigned count);
void trace_data_add_taskid (trace_t *tr, char *id);

void trace_data_add_task (trace_t *tr, uint64_t start_time, uint64_t end_time,
                          unsigned x, unsigned y, unsigned w, unsigned h,
                          unsigned iteration, unsigned cpu,
                          task_type_t task_type, int task_id,
                          int64_t *counters);

void trace_data_start_iteration (trace_t *tr, uint64_t start_time);
void trace_data_end_iteration (trace_t *tr, uint64_t end_time);

void trace_data_no_more_data (trace_t *tr);

void trace_data_sync_iterations (void);

void trace_data_finalize (void);

#define for_all_tasks(tr, cpu, var)                                            \
  list_for_each_entry (trace_task_t, var, (tr)->per_cpu + (cpu), cpu_chain)

int trace_data_search_iteration (trace_t *tr, long t);
int trace_data_search_next_iteration (trace_t *tr, long t);
int trace_data_search_prev_iteration (trace_t *tr, long t);

#define iteration_start_time(tr, it)                                           \
  (trace_data_align_mode                                                       \
       ? (tr)->iteration[it].start_time + (tr)->iteration[it].correction       \
       : (tr)->iteration[it].start_time)

#define iteration_end_time(tr, it)                                             \
  (trace_data_align_mode                                                       \
       ? (tr)->iteration[it].end_time + (tr)->iteration[it].correction +       \
             (tr)->iteration[it].gap                                           \
       : (tr)->iteration[it].end_time)

#define task_start_time(tr, t)                                                 \
  (trace_data_align_mode                                                       \
       ? (t)->start_time + (tr)->iteration[(t)->iteration].correction          \
       : (t)->start_time)

#define task_end_time(tr, t)                                                   \
  (trace_data_align_mode                                                       \
       ? (t)->end_time + (tr)->iteration[(t)->iteration].correction            \
       : (t)->end_time)

#endif
