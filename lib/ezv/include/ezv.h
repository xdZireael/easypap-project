#ifndef EZV_H
#define EZV_H

#include "ezv_hud.h"
#include "ezv_palette.h"
#include "ezv_rgba.h"

#include "ezv_img2d.h"
#include "ezv_mesh3d.h"
#include "ezv_mon.h"

typedef enum
{
  EZV_CTX_TYPE_MESH3D,
  EZV_CTX_TYPE_IMG2D,
  EZV_CTX_TYPE_MONITOR
} ezv_ctx_type_t;

#define EZV_ENABLE_PICKING 1
#define EZV_ENABLE_VSYNC 2
#define EZV_ENABLE_HUD 4
#define EZV_ENABLE_CLIPPING 8
#define EZV_HIDE_WINDOW 16

struct ezv_ctx_s;
typedef struct ezv_ctx_s *ezv_ctx_t;

void ezv_init (const char *prefix);
void ezv_load_opengl (void);
unsigned ezv_get_max_display_height (void);

ezv_ctx_t ezv_ctx_create (ezv_ctx_type_t ctx_type, const char *win_title, int x,
                          int y, int w, int h, int flags);

ezv_ctx_type_t ezv_ctx_type (ezv_ctx_t ctx);
char *ezv_ctx_typestr (ezv_ctx_t ctx);
unsigned ezv_ctx_width (ezv_ctx_t ctx);
unsigned ezv_ctx_height (ezv_ctx_t ctx);

void ezv_ctx_raise (ezv_ctx_t ctx);
void ezv_ctx_show (ezv_ctx_t ctx);
void ezv_ctx_hide (ezv_ctx_t ctx);

void ezv_switch_to_context (ezv_ctx_t ctx);
void ezv_ctx_destroy (ezv_ctx_t ctx);
void ezv_toggle_clipping (ezv_ctx_t ctx[], unsigned nb_ctx);

void ezv_use_cpu_colors (ezv_ctx_t ctx);
void ezv_reset_cpu_colors (ezv_ctx_t ctx);

void ezv_set_cpu_color_1D (ezv_ctx_t ctx, unsigned offset, unsigned size,
                           uint32_t color);
void ezv_set_cpu_color_2D (ezv_ctx_t ctx, unsigned x, unsigned width,
                           unsigned y, unsigned height, uint32_t color);

void ezv_use_data_colors (ezv_ctx_t ctx, float *data, unsigned size);
void ezv_use_data_colors_predefined (ezv_ctx_t ctx, ezv_palette_name_t palette);

void ezv_set_data_colors (ezv_ctx_t ctx, void *values);

// Virtual methods
void ezv_render (ezv_ctx_t ctx[], unsigned nb_ctx);
void ezv_reset_view (ezv_ctx_t ctx[], unsigned nb_ctx);
void ezv_get_shareable_buffer_ids (ezv_ctx_t ctx, int buffer_ids[]);
void ezv_set_data_brightness (ezv_ctx_t ctx, float brightness);

// Helpers
void ezv_helper_ctx_next_coord (ezv_ctx_t ctx[], unsigned ctx_no,
                                unsigned *xwin, unsigned *ywin);

#endif
