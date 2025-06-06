
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "ezv.h"
#include "ezv_event.h"

// settings
const unsigned int SCR_WIDTH  = 1024;
const unsigned int SCR_HEIGHT = 768;

#define MAX_CTX 2

static mesh3d_obj_t mesh;
static ezv_ctx_t ctx[MAX_CTX] = {NULL, NULL};
static unsigned nb_ctx           = 1;
static int hud                   = -1;

static void do_pick (void)
{
  int p = ezv_perform_1D_picking (ctx, nb_ctx);

  if (p == -1)
    ezv_hud_off (ctx[0], hud);
  else {
    ezv_hud_on (ctx[0], hud);
    ezv_hud_set (ctx[0], hud, "Cell: %d", p);
  }
}

static void process_events (void)
{
  SDL_Event event;

  int r = ezv_get_event (&event, 1);

  if (r > 0) {
    int pick;
    ezv_process_event (ctx, nb_ctx, &event, NULL, &pick);
    if (pick)
      do_pick ();
  }
}

static void load_mesh (int argc, char *argv[], mesh3d_obj_t *mesh)
{
  if (argc > 1) {
    unsigned group = 1;

    if (argc > 2)
      group = atoi (argv[2]);

    if (!strcmp (argv[1], "--help") || !strcmp (argv[1], "-h")) {
      printf ("Usage: %s [ -t | -c | -cv | -tv | -cy ] "
              "[cell_size]\n",
              argv[0]);
      exit (0);
    } else if (!strcmp (argv[1], "--torus") || !strcmp (argv[1], "-t"))
      mesh3d_obj_build_torus_surface (mesh, group);
    else if (!strcmp (argv[1], "--cube") || !strcmp (argv[1], "-c"))
      mesh3d_obj_build_cube (mesh, group);
    else if (!strcmp (argv[1], "--cubus-volumus") || !strcmp (argv[1], "-cv"))
      mesh3d_obj_build_cube_volume (mesh, group);
    else if (!strcmp (argv[1], "--torus-volumus") || !strcmp (argv[1], "-tv"))
      mesh3d_obj_build_torus_volume (mesh, 32, 16, 16);
    else if (!strcmp (argv[1], "--cylinder-volumus") ||
             !strcmp (argv[1], "-cy"))
      mesh3d_obj_build_cylinder_volume (mesh, 400, 200);
    else if (!strcmp (argv[1], "--wall") ||
             !strcmp (argv[1], "-w"))
      mesh3d_obj_build_wall (mesh, 8);
    else
      mesh3d_obj_load (argv[1], mesh);
  } else
    mesh3d_obj_build_default (mesh);
}

int main (int argc, char *argv[])
{
  ezv_init ();

  mesh3d_obj_init (&mesh);
  load_mesh (argc, argv, &mesh);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = ezv_ctx_create (EZV_CTX_TYPE_MESH3D, "Mesh", SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT,
                              EZV_ENABLE_PICKING | EZV_ENABLE_HUD |
                                  EZV_ENABLE_CLIPPING);
  hud    = ezv_hud_alloc (ctx[0]);
  ezv_hud_on (ctx[0], hud);

  // Attach mesh
  ezv_mesh3d_set_mesh (ctx[0], &mesh);

  ezv_use_data_colors_predefined (ctx[0], EZV_PALETTE_RAINBOW);

  float *values = malloc (mesh.nb_cells * sizeof (float));

  // Color cells according to their position within the bounding box
  const int COORD = 2;
  for (int c = 0; c < mesh.nb_cells; c++) {
    bbox_t box;
    mesh3d_obj_get_bbox_of_cell (&mesh, c, &box);
    float z   = (box.min[COORD] + box.max[COORD]) / 2.0f;
    values[c] = (z - mesh.bbox.min[COORD]) /
                (mesh.bbox.max[COORD] - mesh.bbox.min[COORD]);
  }

  ezv_set_data_colors (ctx[0], values);

  // render loop
  while (1) {
    process_events ();
    ezv_render (ctx, nb_ctx);
  }

  free (values);

  return 0;
}
