
#ifndef TEST_TEST_H
#pragma once
#include <sanitizer.h>
#include <sanitizer_result.h>
#include <stdint.h>
#include <dlfcn.h>
#include "stdbool.h"
#include <stdio.h>
// #include "debug-info.h"
#define STANDALONE


#include <cuda.h>
#include <link.h>
#include <stdbool.h>

// #include <stdio.h>
#define SANITIZER_API_DEBUG 1
#if SANITIZER_API_DEBUG
#define PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINT(...)
#endif

#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#define PRINT_INFO(...) fprintf(stderr, __VA_ARGS__)





#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

// @findhao: when you are going to change the bellowing functions, you have to change both of it in cplusplus and else.
#ifdef __cplusplus

#else


typedef struct gpu_patch_buffer {
  volatile uint32_t full;
  volatile uint32_t analysis;
  volatile uint32_t head_index;
  volatile uint32_t tail_index;
  uint32_t size;
  uint32_t num_threads;  // If num_threads == 0, the kernel is finished
  uint32_t block_sampling_offset;
  uint32_t block_sampling_frequency;
  uint32_t type;
  uint32_t flags;  // read or write or both
  void *records;
  void *aux;
} gpu_patch_buffer_t;


static Sanitizer_StreamHandle priority_stream_handle_global = NULL;
static CUstream priority_stream_global = NULL;

static void sanitizer_load_callback(CUcontext context, CUmodule module,
                                    const void *cubin, size_t cubin_size);
static void sanitizer_unload_callback(const void *module, const void *cubin,
                                      size_t cubin_size);
static void sanitizer_buffer_init(CUcontext context);
static Sanitizer_StreamHandle sanitizer_priority_stream_get(CUcontext context);
static void sanitizer_subscribe_callback(void *userdata,
                                         Sanitizer_CallbackDomain domain,
                                         Sanitizer_CallbackId cbid,
                                         const void *cbdata);

#endif


EXTERNC int sanitizer_callbacks_subscribe();

EXTERNC void sanitizer_callbacks_unsubscribe();


//*****************************************************************************
// interface operations
//*****************************************************************************

// returns 0 on success
int cuda_bind(
    void);


void
cuda_priority_stream_create_with_param(CUstream* stream);



#define TEST_TEST_H

#endif // TEST_TEST_H
