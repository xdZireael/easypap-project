#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#include "error.h"
#include "trace_data.h"
#include "ezv.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

trace_t trace[MAX_TRACES];
unsigned nb_traces             = 0;
unsigned trace_data_align_mode = 0;

#define REMOVE_OVERHEAD

#ifdef REMOVE_OVERHEAD

static long overhead           = 0;
static long end_last_iteration = 0;
static long fixed_gap          = 0;

#define shift(t) ((t)-overhead)

#else

#define shift(t) (t)
#endif

static LIST_HEAD (tmp_list);
static trace_iteration_t *current_it = NULL;

void trace_data_init (trace_t *tr, unsigned num)
{
  overhead           = 0;
  end_last_iteration = 0;
  fixed_gap          = 0;
  current_it         = NULL;

  tr->num             = num;
  tr->nb_cores        = 1;
  tr->nb_gpu          = 0;
  tr->per_cpu         = NULL;
  tr->nb_iterations   = 0;
  tr->first_iteration = 1;
  tr->label           = NULL;
  tr->mesh_file       = NULL;
  tr->palette         = EZV_PALETTE_LINEAR;
  tr->task_ids        = NULL;
  tr->task_ids_count  = 0;
  tr->has_cache_data  = 0;
}

void trace_data_set_nb_threads (trace_t *tr, unsigned nb_cores, unsigned nb_gpu)
{
  tr->nb_cores = nb_cores + nb_gpu;
  tr->nb_gpu   = nb_gpu;
  tr->per_cpu  = malloc (tr->nb_cores * sizeof (trace_task_t));
  for (int i = 0; i < tr->nb_cores; i++)
    INIT_LIST_HEAD (tr->per_cpu + i);
}

void trace_data_set_dim (trace_t *tr, unsigned dim)
{
  tr->dimensions = dim;
}

void trace_data_set_do_cache (trace_t *tr, unsigned use_cache)
{
  tr->has_cache_data = use_cache;
}

void trace_data_set_first_iteration (trace_t *tr, unsigned it)
{
  tr->first_iteration = it;
}

void trace_data_set_label (trace_t *tr, char *label)
{
  tr->label = malloc (strlen (label) + 1);
  strcpy (tr->label, label);
}

void trace_data_set_meshfile (trace_t *tr, char *filename)
{
  tr->mesh_file = malloc (strlen (filename) + 1);
  strcpy (tr->mesh_file, filename);
}

void trace_data_set_palette (trace_t *tr, ezv_palette_name_t palette)
{
  tr->palette = palette;
}

static int next_id[MAX_TRACES] = {0, 0};

void trace_data_alloc_task_ids (trace_t *tr, unsigned count)
{
  tr->task_ids       = calloc (count, sizeof (char *));
  tr->task_ids_count = count;
}

void trace_data_add_taskid (trace_t *tr, char *id)
{
  int i           = next_id[tr->num]++;
  tr->task_ids[i] = malloc (strlen (id) + 1);
  strcpy (tr->task_ids[i], id);
}

void trace_data_add_task (trace_t *tr, uint64_t start_time, uint64_t end_time,
                          unsigned x, unsigned y, unsigned w, unsigned h,
                          unsigned iteration, unsigned cpu,
                          task_type_t task_type, int task_id, int64_t *counters)
{
  trace_task_t *t = malloc (sizeof (trace_task_t));

  t->start_time = shift (start_time);
  t->end_time   = shift (end_time);
  t->x          = x;
  t->y          = y;
  t->w          = w;
  t->h          = h;
  t->iteration  = iteration;
  t->task_type  = task_type;
  t->task_id    = task_id;
  if (tr->has_cache_data) {
    if (counters == NULL) {
      for (int c = 0; c < EASYPAP_NB_COUNTERS; c++)
        t->counters[c] = 0;
    } else {
      for (int c = 0; c < EASYPAP_NB_COUNTERS; c++) {
        t->counters[c] = counters[c];
#ifdef ENABLE_PER_ITERATION_STATS
        current_it->perfcounter_cpu_scores[cpu][c] += counters[c];
        current_it->perfcounter_scores[c] += counters[c];
#endif
      }
    }
  }

  list_add_tail (&t->cpu_chain, tr->per_cpu + cpu);

  if (current_it->first_cpu_task[cpu] == NULL)
    current_it->first_cpu_task[cpu] = t;
}

static void trace_data_display_all (trace_t *tr)
{
  // We go through a given range of iterations
  for (int it = 0; it < tr->nb_iterations; it++) {
    printf ("*** Iteration %d ***\n", it + 1);

    for (int c = 0; c < tr->nb_cores; c++) {
      printf ("On CPU %d :\n", c);

      // We get a pointer on the first task of the current iteration executed by
      // CPU 'c'
      trace_task_t *first = tr->iteration[it].first_cpu_task[c];

      if (first != NULL)
        // We follow the list of tasks, starting from this first task
        list_for_each_entry_from (trace_task_t, t, tr->per_cpu + c, first,
                                  cpu_chain)
        {
          // We stop if we encounter a task belonging to a greater iteration
          if (t->iteration > it + 1)
            break;

          printf ("Task: time [%" PRIu64 "-%" PRIu64 "], tile [%d, %d, %d, %d], iteration %d\n",
                  task_start_time (tr, t), task_end_time (tr, t), t->x, t->y,
                  t->w, t->h, t->iteration);
        }
    }
  }
}

void trace_data_start_iteration (trace_t *tr, uint64_t start_time)
{
  tr->nb_iterations++;

  // printf ("Iteration %d : start %lu -> ", tr->nb_iterations, start_time);
#ifdef REMOVE_OVERHEAD
  overhead += shift (start_time) - end_last_iteration - fixed_gap;
#endif

  current_it                 = malloc (sizeof (trace_iteration_t));
  current_it->correction     = 0;
  current_it->gap            = 0;
  current_it->start_time     = shift (start_time);
  current_it->first_cpu_task = calloc (tr->nb_cores, sizeof (trace_task_t *));

#ifdef ENABLE_PER_ITERATION_STATS
  if (tr->has_cache_data) {
    current_it->perfcounter_cpu_scores =
        calloc (tr->nb_cores, sizeof (perfcounter_array_t));
    for (int i = 0; i < EASYPAP_NB_COUNTERS; i++) {
      current_it->perfcounter_scores[i] = 0;
    }
  } else {
    current_it->perfcounter_cpu_scores = NULL;
  }
#endif

  list_add_tail (&current_it->chain, &tmp_list);
}

void trace_data_end_iteration (trace_t *tr, uint64_t end_time)
{
  current_it->end_time = shift (end_time);
#ifdef REMOVE_OVERHEAD
  end_last_iteration = current_it->end_time;
  if (tr->nb_iterations == 1) {
    // gap = 10% of first iteration
    // fixed_gap = (current_it->end_time - current_it->start_time) * 10 / 100;
    fixed_gap = 1;
  }
#endif
  // printf ("Iteration %d : end %lu -> %lu\n", tr->nb_iterations, end_time,
  // end_last_iteration);
}

void trace_data_no_more_data (trace_t *tr)
{
  unsigned cpt  = 0;
  tr->iteration = malloc (tr->nb_iterations * sizeof (trace_iteration_t));

  // Now we know the number of iterations: we copy the list' elements
  // into an array, and we delete the list
  list_for_each_entry_safe (trace_iteration_t, pos, &tmp_list, chain)
  {
    list_del (&pos->chain);
    tr->iteration[cpt] = *pos;
    free (pos);
    cpt++;
  }

  // We now finalize 'first_cpu_task' arrays so that
  // iteration[i].first_cpu_task[c] points on the first task the starting time
  // of which is greater or equal to start time of iteration i. In other words,
  // iteration[i].first_cpu_task[c] can point on a task belonging to an
  // iteration j > i.
  if (tr->nb_iterations > 1) {
    for (int i = tr->nb_iterations - 2; i >= 0; i--) {
      for (int c = 0; c < tr->nb_cores; c++)
        if (tr->iteration[i].first_cpu_task[c] == NULL)
          tr->iteration[i].first_cpu_task[c] =
              tr->iteration[i + 1].first_cpu_task[c];
    }
  }
}

void trace_data_finalize (void)
{
  // TODO: Free all memory!
}

int trace_data_search_iteration (trace_t *tr, long t)
{
  unsigned first = 0;
  unsigned last  = tr->nb_iterations - 1;

  while (first < last) {
    unsigned middle = (first + last) / 2;
    if (t > iteration_end_time (tr, middle))
      first = middle + 1;
    else
      last = middle;
  }

  if (t >= iteration_start_time (tr, first) &&
      t <= iteration_end_time (tr, first))
    return first;
  else
    return -1;
}

int trace_data_search_next_iteration (trace_t *tr, long t)
{
  unsigned first = 0;
  unsigned last  = tr->nb_iterations - 1;

  while (first < last) {
    unsigned middle = (first + last) / 2;
    if (t > iteration_end_time (tr, middle))
      first = middle + 1;
    else
      last = middle;
  }

  return first;
}

int trace_data_search_prev_iteration (trace_t *tr, long t)
{
  unsigned first = 0;
  unsigned last  = tr->nb_iterations - 1;

  while (first < last) {
    unsigned middle = (first + last) / 2;
    if (t >= iteration_start_time (tr, middle + 1))
      first = middle + 1;
    else
      last = middle;
  }

  return first;
}

void trace_data_sync_iterations (void)
{
  if (nb_traces == 1) {
    trace_data_align_mode = 1;
    return;
  }

  long cur_correction[MAX_TRACES] = {0};
  unsigned min_it = min (trace[0].nb_iterations, trace[1].nb_iterations);

  for (int it = 0; it < min_it; it++) {
    // We apply correctiun accumulated in previous iterations
    trace[0].iteration[it].correction = cur_correction[0];
    trace[1].iteration[it].correction = cur_correction[1];

    // We compare durations
    long d0 =
        trace[0].iteration[it].end_time - trace[0].iteration[it].start_time;
    long d1 =
        trace[1].iteration[it].end_time - trace[1].iteration[it].start_time;
    if (d0 < d1) {
      trace[0].iteration[it].gap = d1 - d0;
      // We update accumulated adjustment
      cur_correction[0] += trace[0].iteration[it].gap;
    } else {
      trace[1].iteration[it].gap = d0 - d1;
      // We update accumulated adjustment
      cur_correction[1] += trace[1].iteration[it].gap;
    }
  }
  // We check if one trace has more iterations
  int remaining = (trace[0].nb_iterations > trace[1].nb_iterations) ? 0 : 1;

  for (int it = min_it; it < trace[remaining].nb_iterations; it++) {
    trace[remaining].iteration[it].correction = cur_correction[remaining];
  }
}