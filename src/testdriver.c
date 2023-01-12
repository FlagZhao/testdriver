// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2019, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

//***************************************************************************
//
// File:
//   sanitizer-api.c
//
// Purpose:
//   implementation of wrapper around NVIDIA's Sanitizer API
//
//***************************************************************************

#include "testdriver.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#ifndef HPCRUN_STATIC_LINK
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>


#include <errno.h>     // errno
#include <fcntl.h>     // open
#include <stdio.h>     // sprintf
#include <sys/stat.h>  // mkdir
#include <unistd.h>

#ifndef HPCRUN_STATIC_LINK
#ifndef __USE_GNU
#define __USE_GNU /* must define on Linux to get RTLD_NEXT from <dlfcn.h> */
#define SELF_DEFINED__USE_GNU
#endif
#include <dlfcn.h>
#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <link.h>          // dl_iterate_phdr
#undef __USE_GNU


#include <linux/limits.h>  // PATH_MAX
#include <string.h>        // strstr
#endif

#include <cuda.h>
#include <cuda_runtime.h>


//*****************************************************************************
// macros
//*****************************************************************************
#define DYN_FN_NAME(f) f##_fn

#define CUDA_FN_NAME(f) DYN_FN_NAME(f)

#define CUDA_FN(fn, args) \
  static CUresult(*CUDA_FN_NAME(fn)) args

#define CUDA_RUNTIME_FN(fn, args) \
  static cudaError_t(*CUDA_FN_NAME(fn)) args

#define CHK_DLOPEN(h, lib, flags) \
  void *h = dlopen(lib, flags);   \
  if (!h) {                       \
    return -1;                    \
  }

#define CHK_DLSYM(h, fn)             \
  {                                  \
    dlerror();                       \
    DYN_FN_NAME(fn) = dlsym(h, #fn); \
    if (DYN_FN_NAME(fn) == 0) {      \
      return -1;                     \
    }                                \
  }

#define HPCRUN_CUDA_API_CALL(fn, args)                \
  {                                                   \
    CUresult error_result = CUDA_FN_NAME(fn) args;    \
    if (error_result != CUDA_SUCCESS) {               \
      fprintf(stderr, "cuda api %s returned %d", #fn, \
              (int)error_result);                     \
      exit(-1);                                       \
    }                                                 \
  }

#define HPCRUN_CUDA_RUNTIME_CALL(fn, args)                \
  {                                                       \
    cudaError_t error_result = CUDA_FN_NAME(fn) args;     \
    if (error_result != cudaSuccess) {                    \
      fprintf(stderr, "cuda runtime %s returned %d", #fn, \
              (int)error_result);                         \
      exit(-1);                                           \
    }                                                     \
  }


static __thread bool cuda_internal = false;

#ifndef HPCRUN_STATIC_LINK

CUDA_FN(cuDeviceGetAttribute, (int *pi, CUdevice_attribute attrib, CUdevice dev));

CUDA_FN(cuCtxGetCurrent, (CUcontext * ctx));

CUDA_FN(cuCtxSetCurrent, (CUcontext ctx));

CUDA_RUNTIME_FN(cudaGetDevice, (int *device_id));

CUDA_RUNTIME_FN(cudaRuntimeGetVersion, (int *runtimeVersion));

CUDA_FN(cuCtxGetStreamPriorityRange, (int *leastPriority, int *greatestPriority));

CUDA_FN(cuStreamCreateWithPriority, (CUstream * phStream, unsigned int flags, int priority););

CUDA_FN(cuStreamCreate, (CUstream * phStream, unsigned int Flags););

CUDA_FN(cuStreamSynchronize, (CUstream hStream););

CUDA_FN(cuMemcpyDtoHAsync, (void *dst, CUdeviceptr src, size_t byteCount, CUstream stream););

CUDA_FN(cuMemcpyHtoDAsync, (CUdeviceptr dst, void *src, size_t byteCount, CUstream stream););

CUDA_FN(cuModuleLoad, (CUmodule * module, const char *fname););

CUDA_FN(cuModuleGetFunction, (CUfunction * hfunc, CUmodule hmod, const char *name););

CUDA_FN(cuLaunchKernel, (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                         unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra););

CUDA_FN(cuFuncSetAttribute, (CUfunction hfunc, CUfunction_attribute attrib, int value););

#endif


static __thread bool sanitizer_stop_flag = false;


static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_device = NULL;


static bool __buffer_inited = false;


void sanitizer_global_priority_stream_init(CUcontext context){
  CUstream *temp = (CUstream*)malloc(sizeof(CUstream));
  if(priority_stream_global == NULL){
    cuda_priority_stream_create_with_param(temp);
    priority_stream_global = *temp;
  }
}


// only subscribed by the main thread
static Sanitizer_SubscriberHandle sanitizer_subscriber_handle;

typedef void (*sanitizer_error_callback_t)(const char *type, const char *fn,
                                           const char *error_string);

static void sanitizer_error_callback_dummy(const char *type, const char *fn,
                                           const char *error_string);

static sanitizer_error_callback_t sanitizer_error_callback =
    sanitizer_error_callback_dummy;



//#define DYN_FN_NAME(f) f ## _fn
#define SANITIZER_FN_NAME(f) f

#define SANITIZER_FN(fn, args) \
  static SanitizerResult(*SANITIZER_FN_NAME(fn)) args

#define TEST_SANITIZER_CALL(fn, args)              \
  {                                                      \
    SanitizerResult status = SANITIZER_FN_NAME(fn) args; \
    if (status != SANITIZER_SUCCESS) {                   \
      sanitizer_error_report(status, #fn);               \
    }                                                    \
  }

#define TEST_SANITIZER_CALL_NO_CHECK(fn, args) \
  {                                                  \
    SANITIZER_FN_NAME(fn)                            \
    args;                                            \
  }


//----------------------------------------------------------
// sanitizer function pointers for late binding
//----------------------------------------------------------

static void sanitizer_error_callback_dummy  // __attribute__((unused))
    (const char *type, const char *fn, const char *error_string) {
  PRINT("Sanitizer-> %s: function %s failed with error %s\n", type, fn,
        error_string);
  exit(-1);
}

static void sanitizer_error_report(SanitizerResult error, const char *fn) {
  const char *error_string;
  SANITIZER_FN_NAME(sanitizerGetResultString)
  (error, &error_string);
  sanitizer_error_callback("Sanitizer result error", fn, error_string);
}

void memcpy_debug(Sanitizer_StreamHandle priority_stream, char* msg){

    PRINT("\n\ndebugging: %s\n",msg);
    gpu_patch_buffer_t host_test;
    PRINT("Host buffer addr is %p\nDevice buffer addr is %p\nStream is %p\n",&host_test,sanitizer_gpu_patch_buffer_device,priority_stream);
    TEST_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,(&host_test,
                                                          sanitizer_gpu_patch_buffer_device,
                                                          sizeof(gpu_patch_buffer_t),priority_stream));
    PRINT("Test memcpy Flags:%d   >>>>>>>>       msg:%s\n\n",host_test.full,msg);

}

//----------------------------------------------------------
// sampling
//----------------------------------------------------------

static void sanitizer_load_callback(CUcontext context, CUmodule module,
                                    const void *cubin, size_t cubin_size) {
  sanitizer_buffer_init(context);
}

static void sanitizer_unload_callback(const void *module, const void *cubin,
                                      size_t cubin_size) {
}

static void sanitizer_buffer_init(CUcontext context) {
  if (sanitizer_gpu_patch_buffer_device != NULL) {
    // All entries have been initialized
    return;
  }

  // Get cached entry
  Sanitizer_StreamHandle priority_stream =
      sanitizer_priority_stream_get(context);

  sanitizer_gpu_patch_buffer_device = NULL;

  if (sanitizer_gpu_patch_buffer_device == NULL) {
    // Allocated buffer
    // gpu_patch_buffer
    TEST_SANITIZER_CALL(sanitizerAlloc,
                              (context,
                               (void **)(&(sanitizer_gpu_patch_buffer_device)),
                               sizeof(gpu_patch_buffer_t)));
    TEST_SANITIZER_CALL(sanitizerMemset,
                              (sanitizer_gpu_patch_buffer_device, 0,
                               sizeof(gpu_patch_buffer_t), priority_stream));

    PRINT("Sanitizer-> Allocate gpu_patch_buffer %p, size %zu\n",
          sanitizer_gpu_patch_buffer_device, sizeof(gpu_patch_buffer_t));

    __buffer_inited = 1;

    // Ensure data copy is done
    TEST_SANITIZER_CALL(sanitizerStreamSynchronize, (priority_stream));
  }
  
}

static Sanitizer_StreamHandle sanitizer_priority_stream_get(CUcontext context) {
  
  PRINT("Context is %p\n", context);

  if(priority_stream_global == NULL){
    sanitizer_global_priority_stream_init(context);
  }

  if (priority_stream_handle_global == NULL) {
    // First time
    // create priority stream
    TEST_SANITIZER_CALL(
        sanitizerGetStreamHandle,
        (context, priority_stream_global, &priority_stream_handle_global));
  }

  return priority_stream_handle_global;
}


void sanitizer_callbacks_unsubscribe() {
  TEST_SANITIZER_CALL(sanitizerUnsubscribe,
                            (sanitizer_subscriber_handle));
  TEST_SANITIZER_CALL(
      sanitizerSubscribe,
      (&sanitizer_subscriber_handle, sanitizer_subscribe_callback, NULL));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_LAUNCH));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_RESOURCE));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMCPY));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMSET));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_SYNCHRONIZE));
}

static void sanitizer_subscribe_callback(void *userdata,
                                         Sanitizer_CallbackDomain domain,
                                         Sanitizer_CallbackId cbid,
                                         const void *cbdata) {

  if (!sanitizer_stop_flag) {
    sanitizer_stop_flag = true;
  }

  PRINT("Sanitizer> Domain is %d cbid is %d\n",domain,cbid);


  if (domain == SANITIZER_CB_DOMAIN_RESOURCE) {
    switch (cbid) {
      case SANITIZER_CBID_RESOURCE_MODULE_LOADED: {
        // single thread
        Sanitizer_ResourceModuleData *md = (Sanitizer_ResourceModuleData *)cbdata;
        sanitizer_load_callback(md->context, md->module, md->pCubin,
                                md->cubinSize);
        break;
      }
      case SANITIZER_CBID_RESOURCE_MODULE_UNLOAD_STARTING: {
        // single thread
        Sanitizer_ResourceModuleData *md = (Sanitizer_ResourceModuleData *)cbdata;
        sanitizer_unload_callback(md->module, md->pCubin, md->cubinSize);
        if (__buffer_inited){

          // print context addr
          PRINT("Sanitizer-> Context %p \n", md->context);
          // use memcpy_debug
          char msg[100] = "SANITIZER_CBID_RESOURCE_MODULE_UNLOAD_STARTING";
          Sanitizer_StreamHandle priority_stream = sanitizer_priority_stream_get(md->context);
          memcpy_debug(priority_stream, msg);
        }
        break;
      }
      default: {
        break;
      }
    }
  }else if (domain == SANITIZER_CB_DOMAIN_MEMCPY) {
    Sanitizer_MemcpyData *md = (Sanitizer_MemcpyData *)cbdata;
    Sanitizer_StreamHandle priority_stream = sanitizer_priority_stream_get(md->srcContext);
    TEST_SANITIZER_CALL(sanitizerMemset,
                              (sanitizer_gpu_patch_buffer_device, 0,
                               sizeof(gpu_patch_buffer_t), priority_stream));
    PRINT("Memset Done\n");
    memcpy_debug(priority_stream,"SANITIZER_CB_DOMAIN_MEMCPY");
  }
}


__attribute__((constructor))
int sanitizer_callbacks_subscribe() {
  pid_t pid = getpid();
  if (cuda_bind()) {
    PRINT_ERR("Test driver-> unable to bind to NVIDIA CUDA library%s\n", dlerror());
  }
  PRINT("PID: %d\n", pid);

  TEST_SANITIZER_CALL(
      sanitizerSubscribe,
      (&sanitizer_subscriber_handle, sanitizer_subscribe_callback, NULL));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_LAUNCH));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_UVM));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_RESOURCE));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMCPY));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMSET));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_DRIVER_API));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_RUNTIME_API));
  TEST_SANITIZER_CALL(
      sanitizerEnableDomain,
      (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_SYNCHRONIZE));
  return 0;
}


int cuda_bind(
    void) {
#ifndef HPCRUN_STATIC_LINK
  // dynamic libraries only availabile in non-static case
  CHK_DLOPEN(cuda, "libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  CHK_DLSYM(cuda, cuDeviceGetAttribute);
  CHK_DLSYM(cuda, cuCtxGetCurrent);
  CHK_DLSYM(cuda, cuCtxSetCurrent);
  CHK_DLOPEN(cudart, "libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  CHK_DLSYM(cudart, cudaGetDevice);
  CHK_DLSYM(cudart, cudaRuntimeGetVersion);
  CHK_DLSYM(cuda, cuCtxGetStreamPriorityRange);
  CHK_DLSYM(cuda, cuStreamCreateWithPriority);
  CHK_DLSYM(cuda, cuStreamCreate);
  CHK_DLSYM(cuda, cuStreamSynchronize);
  CHK_DLSYM(cuda, cuMemcpyDtoHAsync);
  CHK_DLSYM(cuda, cuMemcpyHtoDAsync);
  CHK_DLSYM(cuda, cuModuleLoad);
  CHK_DLSYM(cuda, cuModuleGetFunction);
  CHK_DLSYM(cuda, cuLaunchKernel);
  CHK_DLSYM(cuda, cuFuncSetAttribute);
  return 0;
#else
  return -1;
#endif  // ! HPCRUN_STATIC_LINK
}

void
cuda_priority_stream_create_with_param(CUstream* stream) {
#ifndef HPCRUN_STATIC_LINK
  cuda_internal = true;
  int priority_high=0, priority_low=0;
  // CUstream stream;
  HPCRUN_CUDA_API_CALL(cuCtxGetStreamPriorityRange,
                       (&priority_low, &priority_high));
  HPCRUN_CUDA_API_CALL(cuStreamCreateWithPriority,
                       (stream, CU_STREAM_NON_BLOCKING, priority_high));
  cuda_internal = false;
  printf("stream using is %p \n",stream);
  // return stream;
#else
  return NULL;
#endif
}


