/*
 * Tencent is pleased to support the open source community by making TKEStack
 * available.
 *
 * Copyright (C) 2012-2019 Tencent. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations under the License.
 */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include "include/cuda-helper.h"
#include "include/hijack.h"
#include "include/nvml-helper.h"

extern resource_data_t g_vcuda_config;
extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];
extern char pid_path[];

static pthread_once_t g_init_set = PTHREAD_ONCE_INIT;
static pthread_once_t g_register_set = PTHREAD_ONCE_INIT;

// 当前已使用的cuda算力，有一个线程去循环维护这个值的变化
static volatile int g_cur_cuda_cores = 0;
static volatile int g_total_cuda_cores = 0;

static int g_max_thread_per_sm = 0;
static int g_sm_num = 0;

static int g_block_x = 1, g_block_y = 1, g_block_z = 1;
static uint32_t g_block_locker = 0;

static const struct timespec g_cycle = {
    .tv_sec = 0,
    .tv_nsec = TIME_TICK * MILLISEC,
};

static const struct timespec g_wait = {
    .tv_sec = 0,
    .tv_nsec = 120 * MILLISEC,
};

/** pid mapping related */
static int g_pids_table[MAX_PIDS];
static int g_pids_table_size;

/** internal function definition */
static void register_to_remote();

static void active_utilization_notifier();

static void *utilization_watcher(void *);

static void get_used_gpu_utilization(int, void *);

static void initialization();

static void rate_limiter(int, int);

static void change_token(int);

static const char *nvml_error(nvmlReturn_t);

static const char *cuda_error(CUresult, const char **);

static int int_match(const void *, const void *);

static int delta(int, int, int);

/** export function definition */
CUresult cuDriverGetVersion(int *driverVersion);
// 驱动初始化
CUresult cuInit(unsigned int flag);
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags); 
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags, void *symbolStatus);           
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes);
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes);
CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels);
// 根据设备号获取设备的总显存大小
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
// 获取显存信息，空闲、最大显存
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
CUresult cuMemGetInfo(size_t *free, size_t *total);
// 执行cuda内核函数
// 检索每线程默认流版本
// 某些 CUDA 驱动程序 API 可以配置为具有默认流或每线程默认流语义。
// 具有每线程默认流语义的驱动程序 API 的名称中带有后缀_ptsz或_ptds。
// 例如，cuLaunchKernel具有名为cuLaunchKernel_ptsz的每线程默认流变体
CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra);
// 执行cuda内核函数 启动具有给定执行配置的内核。
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra);
CUresult cuLaunchKernelEx(CUlaunchConfig *config, CUfunction f, 
                        void **kernelParams, void **extra);
CUresult cuLaunchKernelEx_ptsz(CUlaunchConfig *config, CUfunction f, 
                        void **kernelParams, void **extra);
CUresult cuLaunch(CUfunction f);
CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX, 
                                  unsigned int gridDimY, unsigned int gridDimZ, 
                                  unsigned int blockDimX, unsigned int blockDimY,
                                  unsigned int blockDimZ, unsigned int sharedMemBytes, 
                                  CUstream hStream, void **kernelParams);
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                  unsigned int gridDimY, unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY,
                                  unsigned int blockDimZ, unsigned int sharedMemBytes,
                                  CUstream hStream, void **kernelParams);
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream);
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr);
CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle);
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr);
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);

entry_t cuda_hooks_entry[] = {
    {.name = "cuDriverGetVersion", .fn_ptr = cuDriverGetVersion},
    {.name = "cuInit", .fn_ptr = cuInit},
    {.name = "cuGetProcAddress", .fn_ptr = cuGetProcAddress},
    {.name = "cuGetProcAddress_v2", .fn_ptr = cuGetProcAddress_v2},
    {.name = "cuMemAllocManaged", .fn_ptr = cuMemAllocManaged},
    {.name = "cuMemAlloc_v2", .fn_ptr = cuMemAlloc_v2},
    {.name = "cuMemAlloc", .fn_ptr = cuMemAlloc},
    {.name = "cuMemAllocPitch_v2", .fn_ptr = cuMemAllocPitch_v2},
    {.name = "cuMemAllocPitch", .fn_ptr = cuMemAllocPitch},
    {.name = "cuArrayCreate_v2", .fn_ptr = cuArrayCreate_v2},
    {.name = "cuArrayCreate", .fn_ptr = cuArrayCreate},
    {.name = "cuArray3DCreate_v2", .fn_ptr = cuArray3DCreate_v2},
    {.name = "cuArray3DCreate", .fn_ptr = cuArray3DCreate},
    {.name = "cuMipmappedArrayCreate", .fn_ptr = cuMipmappedArrayCreate},
    {.name = "cuDeviceTotalMem_v2", .fn_ptr = cuDeviceTotalMem_v2},
    {.name = "cuDeviceTotalMem", .fn_ptr = cuDeviceTotalMem},
    {.name = "cuMemGetInfo_v2", .fn_ptr = cuMemGetInfo_v2},
    {.name = "cuMemGetInfo", .fn_ptr = cuMemGetInfo},
    {.name = "cuLaunchKernel_ptsz", .fn_ptr = cuLaunchKernel_ptsz},
    {.name = "cuLaunchKernel", .fn_ptr = cuLaunchKernel},
    {.name = "cuLaunchKernelEx_ptsz", .fn_ptr = cuLaunchKernelEx_ptsz},
    {.name = "cuLaunchKernelEx", .fn_ptr = cuLaunchKernelEx},
    {.name = "cuLaunch", .fn_ptr = cuLaunch},
    {.name = "cuLaunchCooperativeKernel_ptsz", .fn_ptr = cuLaunchCooperativeKernel_ptsz},
    {.name = "cuLaunchCooperativeKernel", .fn_ptr = cuLaunchCooperativeKernel},
    {.name = "cuLaunchGrid", .fn_ptr = cuLaunchGrid},
    {.name = "cuLaunchGridAsync", .fn_ptr = cuLaunchGridAsync},
    {.name = "cuFuncSetBlockShape", .fn_ptr = cuFuncSetBlockShape},
    // TODO 虚拟内存管理
    {.name = "cuMemAddressFree", .fn_ptr = cuMemAddressFree},
    {.name = "cuMemExportToShareableHandle", .fn_ptr = cuMemExportToShareableHandle},
    {.name = "cuMemGetAccess", .fn_ptr = cuMemGetAccess},
    {.name = "cuMemGetAllocationGranularity", .fn_ptr = cuMemGetAllocationGranularity},
    {.name = "cuMemGetAllocationPropertiesFromHandle", .fn_ptr = cuMemGetAllocationPropertiesFromHandle},
    {.name = "cuMemImportFromShareableHandle", .fn_ptr = cuMemImportFromShareableHandle},
    {.name = "cuMemMapArrayAsync", .fn_ptr = cuMemMapArrayAsync},
    {.name = "cuMemRelease", .fn_ptr = cuMemRelease},
    {.name = "cuMemRetainAllocationHandle", .fn_ptr = cuMemRetainAllocationHandle},
    {.name = "cuMemSetAccess", .fn_ptr = cuMemSetAccess},
    {.name = "cuMemUnmap", .fn_ptr = cuMemUnmap},
    {.name = "cuMemAddressReserve", .fn_ptr = cuMemAddressReserve},
    {.name = "cuMemMap", .fn_ptr = cuMemMap},
    {.name = "cuMemCreate", .fn_ptr = cuMemCreate},
};

const int cuda_hook_nums =
    sizeof(cuda_hooks_entry) / sizeof(cuda_hooks_entry[0]);

/** dynamic rate control */
typedef struct {
  int user_current;
  int sys_current;
  int valid;
  uint64_t checktime;
  int sys_process_num;
} utilization_t;

/** helper function */
int int_match(const void *a, const void *b) {
  const int *ra = (const int *)a;
  const int *rb = (const int *)b;

  if (*ra < *rb) {
    return -1;
  }

  if (*ra > *rb) {
    return 1;
  }

  return 0;
}

void atomic_action(const char *filename, atomic_fn_ptr fn_ptr,
                          void *arg) {
  int fd;

  fd = open(filename, O_RDONLY);
  if (unlikely(fd == -1)) {
    LOGGER(FATAL, "can't open %s, error %s", filename, strerror(errno));
  }

  fn_ptr(fd, arg);

  close(fd);
}

const char *nvml_error(nvmlReturn_t code) {
  const char *(*err_fn)(nvmlReturn_t) = NULL;

  err_fn = nvml_library_entry[NVML_ENTRY_ENUM(nvmlErrorString)].fn_ptr;
  if (unlikely(!err_fn)) {
    LOGGER(FATAL, "can't find nvmlErrorString");
  }

  return err_fn(code);
}

const char *cuda_error(CUresult code, const char **p) {
  CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, code, p);

  return *p;
}

static void change_token(int delta) {
  int cuda_cores_before = 0, cuda_cores_after = 0;

  LOGGER(5, "delta: %d, curr: %d", delta, g_cur_cuda_cores);
  do {
    cuda_cores_before = g_cur_cuda_cores;
    cuda_cores_after = cuda_cores_before + delta;

    if (unlikely(cuda_cores_after > g_total_cuda_cores)) {
      cuda_cores_after = g_total_cuda_cores;
    }
  } while (!CAS(&g_cur_cuda_cores, cuda_cores_before, cuda_cores_after));
}

// 计算资源限制：cuda 核心数
static void rate_limiter(int grids, int blocks) {
  int before_cuda_cores = 0;
  int after_cuda_cores = 0;
  int kernel_size = grids;

  LOGGER(5, "grid: %d, blocks: %d", grids, blocks);
  LOGGER(5, "launch kernel %d, curr core: %d", kernel_size, g_cur_cuda_cores);
  // 当开启了vcuda config配置，才开始计算限制
  if (g_vcuda_config.enable) {
    do {
    CHECK:
      before_cuda_cores = g_cur_cuda_cores;
      LOGGER(8, "current core: %d", g_cur_cuda_cores);
      // 当前cuda核心数小于0，则循环等待
      if (before_cuda_cores < 0) {
        // 睡十毫秒
        nanosleep(&g_cycle, NULL);
        goto CHECK;
      }
      // 更新后cuda核心数 = 当前cuda核心数 - 网格数
      after_cuda_cores = before_cuda_cores - kernel_size;
      // CAS更新当前cuda核心数
    } while (!CAS(&g_cur_cuda_cores, before_cuda_cores, after_cuda_cores));
  }
}

static int delta(int up_limit, int user_current, int share) {
  int utilization_diff =
      abs(up_limit - user_current) < 5 ? 5 : abs(up_limit - user_current);
  int increment =
      g_sm_num * g_sm_num * g_max_thread_per_sm / 256 * utilization_diff / 10;

  /* Accelerate cuda cores allocation when utilization vary widely */
  if (utilization_diff > up_limit / 2) {
    increment = increment * utilization_diff * 2 / (up_limit + 1);
  }

  if (unlikely(increment < 0)) {
    LOGGER(3, "overflow: %d, current sm: %d, thread_per_sm: %d, diff: %d",
           increment, g_sm_num, g_max_thread_per_sm, utilization_diff);
  }

  if (user_current <= up_limit) {
    share = share + increment > g_total_cuda_cores ? g_total_cuda_cores
                                                   : share + increment;
  } else {
    share = share - increment < 0 ? 0 : share - increment;
  }

  return share;
}

// #lizard forgives
static void *utilization_watcher(void *arg UNUSED) {
  utilization_t top_result = {
      .user_current = 0,
      .sys_current = 0,
      .sys_process_num = 0,
  };
  int sys_free = 0;
  int share = 0;
  int i = 0;
  int avg_sys_free = 0;
  int pre_sys_process_num = 1;
  int up_limit = g_vcuda_config.utilization;

  LOGGER(5, "start %s", __FUNCTION__);
  LOGGER(4, "sm: %d, thread per sm: %d", g_sm_num, g_max_thread_per_sm);
  while (1) {
    nanosleep(&g_wait, NULL);
    do {
      // 加载pid文件调用get_used_gpu_utilization函数，计算当前已使用的gpu利用率
      atomic_action(pid_path, get_used_gpu_utilization, (void *)&top_result);
    } while (!top_result.valid);
    // 最大利用率-系统利用率=系统可用利用率
    sys_free = MAX_UTILIZATION - top_result.sys_current;
    // 两种算力隔离策略
    if (g_vcuda_config.hard_limit) {
      /* Avoid usage jitter when application is initialized*/
      if (top_result.sys_process_num == 1 &&
          top_result.user_current < up_limit / 10) {
        g_cur_cuda_cores =
            delta(g_vcuda_config.utilization, top_result.user_current, share);
        continue;
      }
      share = delta(g_vcuda_config.utilization, top_result.user_current, share);
    } else {
      // 当前进程数和检测到的进程数不一致时
      if (pre_sys_process_num != top_result.sys_process_num) {
        /* When a new process comes, all processes are reset to initial value*/
        // 出现了新的进程，则将变量重置为初始值
        if (pre_sys_process_num < top_result.sys_process_num) {
          share = g_max_thread_per_sm;
          up_limit = g_vcuda_config.utilization;
          i = 0;
          avg_sys_free = 0;
        }
        // 修改 当前进程数
        pre_sys_process_num = top_result.sys_process_num;
      }

      /* 1.Only one process on the GPU
       * Allocate cuda cores according to the limit value.
       *
       * 2.Multiple processes on the GPU
       * First, change the up_limit of the process according to the
       * historical resource utilization. Second, allocate the cuda
       * cores according to the changed limit value.*/
      
      /* 1.GPU上只有一个进程
       * 根据限制值分配cuda核心。
       */
      if (top_result.sys_process_num == 1) {
        share = delta(g_vcuda_config.limit, top_result.user_current, share);
      } else {
       /* 2.GPU上有多个进程
        * 首先，根据历史资源利用率更改进程的up_limit。
        * 然后，根据更改后的极限值分配cuda核心 
        */
        i++;
        avg_sys_free += sys_free;
        if (i % CHANGE_LIMIT_INTERVAL == 0) {
          if (avg_sys_free * 2 / CHANGE_LIMIT_INTERVAL > USAGE_THRESHOLD) {
            up_limit = up_limit + g_vcuda_config.utilization / 10 >
                               g_vcuda_config.limit
                           ? g_vcuda_config.limit
                           : up_limit + g_vcuda_config.utilization / 10;
          }
          i = 0;
        }
        avg_sys_free = i % (CHANGE_LIMIT_INTERVAL / 2) == 0 ? 0 : avg_sys_free;
        share = delta(up_limit, top_result.user_current, share);
      }
    }

    change_token(share);

    LOGGER(4, "util: %d, up_limit: %d,  share: %d, cur: %d",
           top_result.user_current, up_limit, share, g_cur_cuda_cores);
  }
}

static void active_utilization_notifier() {
  pthread_t tid;

  pthread_create(&tid, NULL, utilization_watcher, NULL);

#ifdef __APPLE__
  pthread_setname_np("utilization_watcher");
#else
  pthread_setname_np(tid, "utilization_watcher");
#endif
}

static void get_used_gpu_utilization(int fd, void *arg) {
  nvmlProcessUtilizationSample_t processes_sample[MAX_PIDS];
  int processes_num = MAX_PIDS;
  unsigned int running_processes = MAX_PIDS;
  nvmlProcessInfo_t pids_on_device[MAX_PIDS];
  nvmlDevice_t dev;
  utilization_t *top_result = (utilization_t *)arg;
  nvmlReturn_t ret;
  struct timeval cur;
  size_t microsec;
  int codec_util = 0;

  int i;
  // 获取第一个设备
  ret =
      NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetHandleByIndex, 0, &dev);
  if (unlikely(ret)) {
    LOGGER(4, "nvmlDeviceGetHandleByIndex: %s", nvml_error(ret));
    return;
  }
  // 根据设备找到设备上运行中的进程
  ret =
      NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses,
                      dev, &running_processes, pids_on_device);
  if (unlikely(ret)) {
    LOGGER(4, "nvmlDeviceGetComputeRunningProcesses: %s", nvml_error(ret));
    return;
  }
  top_result->sys_process_num = running_processes;

  // 加载容器pid文件获取容器pid
  load_pids_table(fd, NULL);
  gettimeofday(&cur, NULL);
  microsec = (cur.tv_sec - 1) * 1000UL * 1000UL + cur.tv_usec;
  top_result->checktime = microsec;
  // 获取当前设备的线程利用率
  ret = NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetProcessUtilization,
                        dev, processes_sample, &processes_num, microsec);
  if (unlikely(ret)) {
    LOGGER(4, "nvmlDeviceGetProcessUtilization: %s", nvml_error(ret));
    return;
  }
  
  top_result->user_current = 0;
  top_result->sys_current = 0;
  for (i = 0; i < processes_num; i++) {
    if (processes_sample[i].timeStamp >= top_result->checktime) {
      top_result->valid = 1;
      top_result->sys_current += GET_VALID_VALUE(processes_sample[i].smUtil);

      codec_util = GET_VALID_VALUE(processes_sample[i].encUtil) +
                   GET_VALID_VALUE(processes_sample[i].decUtil);
      top_result->sys_current += CODEC_NORMALIZE(codec_util);

      LOGGER(8, "try to find %d from pid tables", processes_sample[i].pid);
      if (likely(bsearch(&processes_sample[i].pid, g_pids_table,
                         (size_t)g_pids_table_size, sizeof(int), int_match))) {
        top_result->user_current += GET_VALID_VALUE(processes_sample[i].smUtil);

        codec_util = GET_VALID_VALUE(processes_sample[i].encUtil) +
                     GET_VALID_VALUE(processes_sample[i].decUtil);
        top_result->user_current += CODEC_NORMALIZE(codec_util);
      }
    }
  }

  LOGGER(5, "sys utilization: %d", top_result->sys_current);
  LOGGER(5, "used utilization: %d", top_result->user_current);
}

void load_pids_table(int fd, void *arg UNUSED) {
  int item = 0;
  int rsize = 0;
  int i = 0;

  for (item = 0; item < MAX_PIDS; item++) {
    rsize = (int)read(fd, g_pids_table + item, sizeof(int));
    if (unlikely(rsize != sizeof(int))) {
      break;
    }
  }

  for (i = 0; i < item; i++) {
    LOGGER(8, "pid: %d", g_pids_table[i]);
  }

  g_pids_table_size = item;

  LOGGER(8, "read %d items from %s", g_pids_table_size, pid_path);
}

// 查找对应的gpu设备上的内存使用
// static void get_used_gpu_memory(int fd, unsigned int index, void *arg) {
//   size_t *used_memory = arg;

//   nvmlDevice_t dev;
//   nvmlProcessInfo_t pids_on_device[MAX_PIDS];
//   unsigned int size_on_device = MAX_PIDS;
//   int ret;

//   unsigned int i;

//   load_pids_table(fd, NULL);
//   // 查找index 0的设备
//   ret =
//       NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetHandleByIndex, index, &dev);
//   if (unlikely(ret)) {
//     LOGGER(4, "nvmlDeviceGetHandleByIndex can't find device %d, return %d", index, ret);
//     *used_memory = g_vcuda_config.gpu_memory;
//     return;
//   }

//   ret =
//       NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses,
//                       dev, &size_on_device, pids_on_device);
//   if (unlikely(ret)) {
//     LOGGER(4,
//            "nvmlDeviceGetComputeRunningProcesses can't get pids on device %d, "
//            "return %d", index, ret);
//     *used_memory = g_vcuda_config.gpu_memory;
//     return;
//   }

//   for (i = 0; i < size_on_device; i++) {
//     LOGGER(4, "summary: %d used %lld", pids_on_device[i].pid,
//            pids_on_device[i].usedGpuMemory);
//   }

//   for (i = 0; i < size_on_device; i++) {
//     // 二分查找g_pids_table下的pid，匹配上之后将该进程使用的显存相加，得到已使用显存
//     if (bsearch(&pids_on_device[i].pid, g_pids_table, (size_t)g_pids_table_size,
//                 sizeof(int), int_match)) {
//       LOGGER(4, "%d use memory: %lld", pids_on_device[i].pid,
//              pids_on_device[i].usedGpuMemory);
//       *used_memory += pids_on_device[i].usedGpuMemory;
//     }
//   }

//   LOGGER(4, "total used memory: %zu", *used_memory);
// }

// 根据当前容器的pid得到在 device0 上已使用的设备内存
void get_used_gpu_memory(int fd, void *arg) {
  size_t *used_memory = arg;

  nvmlDevice_t dev;
  nvmlProcessInfo_t pids_on_device[MAX_PIDS];
  unsigned int size_on_device = MAX_PIDS;
  int ret;

  unsigned int i;

  load_pids_table(fd, NULL);
  // 查找index 0的设备
  ret =
      NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetHandleByIndex, 0, &dev);
  if (unlikely(ret)) {
    LOGGER(4, "nvmlDeviceGetHandleByIndex can't find device 0, return %d", ret);
    *used_memory = g_vcuda_config.gpu_memory;
    return;
  }

  ret =
      NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses,
                      dev, &size_on_device, pids_on_device);
  if (unlikely(ret)) {
    LOGGER(4,
           "nvmlDeviceGetComputeRunningProcesses can't get pids on device 0, "
           "return %d",
           ret);
    *used_memory = g_vcuda_config.gpu_memory;
    return;
  }

  for (i = 0; i < size_on_device; i++) {
    LOGGER(5, "summary: %d used %lld", pids_on_device[i].pid,
           pids_on_device[i].usedGpuMemory);
  }

  for (i = 0; i < size_on_device; i++) {
    // 二分查找g_pids_table下的pid，匹配上之后将该进程使用的显存相加，得到已使用显存
    if (bsearch(&pids_on_device[i].pid, g_pids_table, (size_t)g_pids_table_size,
                sizeof(int), int_match)) {
      LOGGER(5, "%d use memory: %lld", pids_on_device[i].pid,
             pids_on_device[i].usedGpuMemory);
      *used_memory += pids_on_device[i].usedGpuMemory;
    }
  }

  LOGGER(5, "total used memory: %zu", *used_memory);
}

void search_on_pids_table(unsigned int *infoCount, nvmlProcessInfo_t *infos){
  nvmlProcessInfo_t pids_on_device[MAX_PIDS];
  unsigned int index = 0;
  unsigned int i;
  for (i = 0; i < *infoCount; i++) {
    // 二分查找g_pids_table下的pid，匹配上之后将该进程使用的显存相加，得到已使用显存
    if (bsearch(&infos[i].pid, g_pids_table, (size_t)g_pids_table_size,
                sizeof(int), int_match)) {
      LOGGER(4, "search_on_pids_table: %d", infos[i].pid);
      pids_on_device[index] = infos[i];
      index++;
    }
  }
  *infoCount = index;
  memcpy(infos, pids_on_device, sizeof(nvmlProcessInfo_t) * index);
}

// #lizard forgives
static void register_to_remote() {
  nvmlPciInfo_t pci_info;
  nvmlDevice_t nvml_dev;
  int ret;
  // 查询device 0设备
  ret = NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetHandleByIndex, 0,
                        &nvml_dev);
  if (unlikely(ret)) {
    LOGGER(FATAL, "can't find device 0, error %s",
           nvml_error((nvmlReturn_t)ret));
  }
  // 根据device 0设备查询它的pci信息
  ret = NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetPciInfo, nvml_dev,
                        &pci_info);
  if (unlikely(ret)) {
    LOGGER(FATAL, "can't find device 0, error %s",
           nvml_error((nvmlReturn_t)ret));
  }
  // 将pci总线id赋值给g_vcuda_config.bus_id
  strncpy(g_vcuda_config.bus_id, pci_info.busId,
          NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
  // 注册到远程服务获取数据
  // TODO 这里的调用可能存在bug，container_name已不再使用，调用将会失败
  register_to_remote_with_data(g_vcuda_config.bus_id, g_vcuda_config.pod_uid,
                               g_vcuda_config.container_name, "");
}

static void initialization() {
  int ret;
  const char *cuda_err_string = NULL;
  // 初始化驱动
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, 0);
  if (unlikely(ret)) {
    LOGGER(FATAL, "cuInit error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }
  // 获取device 0设备上的多处理器数量
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetAttribute, &g_sm_num,
                        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
  if (unlikely(ret)) {
    LOGGER(FATAL, "can't get processor number, error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }
  // 获取device 0设备每个处理器的最大驻留线程数
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetAttribute,
                        &g_max_thread_per_sm,
                        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 0);
  if (unlikely(ret)) {
    LOGGER(FATAL, "can't get max thread per processor, error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }
  // 处理器数量 * 最大驻留线程数 * 32 = 最大cuda核心数
  g_total_cuda_cores = g_max_thread_per_sm * g_sm_num * FACTOR;
  LOGGER(4, "total cuda cores: %d", g_total_cuda_cores);
  active_utilization_notifier();
}

/** hijack entrypoint */
CUresult cuDriverGetVersion(int *driverVersion) {
  CUresult ret;

  load_necessary_data();
  if (!is_custom_config_path()) {
    pthread_once(&g_register_set, register_to_remote);
  }
  pthread_once(&g_init_set, initialization);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDriverGetVersion, driverVersion);
  if (unlikely(ret)) {
    goto DONE;
  }

DONE:
  return ret;
}

CUresult cuInit(unsigned int flag) {
  CUresult ret;
  // 加载必要数据
  load_necessary_data();
  if (!is_custom_config_path()) {
    // 注册到远程
    pthread_once(&g_register_set, register_to_remote);
  }
  // 开启线程 初始化
  pthread_once(&g_init_set, initialization);
  // 调用官方api初始化驱动
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, flag);

  if (unlikely(ret)) {
    goto DONE;
  }

DONE:
  return ret;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags) {
  CUresult ret;
  int i;

  load_necessary_data();
  if (!is_custom_config_path()) {
    pthread_once(&g_register_set, register_to_remote);
  }
  pthread_once(&g_init_set, initialization);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn,
                        cudaVersion, flags);
  if (ret == CUDA_SUCCESS) {
    for (i = 0; i < cuda_hook_nums; i++) {
      if (!strcmp(symbol, cuda_hooks_entry[i].name)) {
        LOGGER(5, "Match hook %s", symbol);
        *pfn = cuda_hooks_entry[i].fn_ptr;
        break;
      }
    }
  }

  return ret;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags, void *symbolStatus) {
  CUresult ret;
  int i;

  load_necessary_data();
  if (!is_custom_config_path()) {
    pthread_once(&g_register_set, register_to_remote);
  }
  pthread_once(&g_init_set, initialization);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn,
                        cudaVersion, flags, symbolStatus);
  if (ret == CUDA_SUCCESS) {
    for (i = 0; i < cuda_hook_nums; i++) {
      if (!strcmp(symbol, cuda_hooks_entry[i].name)) {
        LOGGER(5, "Match hook %s", symbol);
        *pfn = cuda_hooks_entry[i].fn_ptr;
        break;
      }
    }
  }

  return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemAllocManaged");
  size_t used = 0;
  size_t request_size = bytesize;
  CUresult ret;

  if (g_vcuda_config.enable) {
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        flags);
DONE:
  return ret;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemAlloc_v2");
  size_t used = 0;
  size_t request_size = bytesize;
  CUresult ret;

  if (g_vcuda_config.enable) {
    
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAlloc_v2, dptr, bytesize);
DONE:
  return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemAlloc");
  size_t used = 0;
  size_t request_size = bytesize;
  CUresult ret;

  if (g_vcuda_config.enable) {
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAlloc, dptr, bytesize);
DONE:
  return ret;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemAllocPitch_v2");
  size_t used = 0;
  size_t request_size = ROUND_UP(WidthInBytes * Height, ElementSizeBytes);
  CUresult ret;

  if (g_vcuda_config.enable) {
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocPitch_v2, dptr, pPitch,
                        WidthInBytes, Height, ElementSizeBytes);
DONE:
  return ret;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemAllocPitch");
  size_t used = 0;
  size_t request_size = ROUND_UP(WidthInBytes * Height, ElementSizeBytes);
  CUresult ret;

  if (g_vcuda_config.enable) {
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocPitch, dptr, pPitch,
                        WidthInBytes, Height, ElementSizeBytes);
DONE:
  return ret;
}

static size_t get_array_base_size(int format) {
  size_t base_size = 0;

  switch (format) {
  case CU_AD_FORMAT_UNSIGNED_INT8:
  case CU_AD_FORMAT_SIGNED_INT8:
    base_size = 8;
    break;
  case CU_AD_FORMAT_UNSIGNED_INT16:
  case CU_AD_FORMAT_SIGNED_INT16:
  case CU_AD_FORMAT_HALF:
    base_size = 16;
    break;
  case CU_AD_FORMAT_UNSIGNED_INT32:
  case CU_AD_FORMAT_SIGNED_INT32:
  case CU_AD_FORMAT_FLOAT:
    base_size = 32;
    break;
  default:
    base_size = 32;
  }

  return base_size;
}

static CUresult
cuArrayCreate_helper(const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  size_t used = 0;
  size_t base_size = 0;
  size_t request_size = 0;
  CUresult ret = CUDA_SUCCESS;

  if (g_vcuda_config.enable) {
    base_size = get_array_base_size(pAllocateArray->Format);
    request_size = base_size * pAllocateArray->NumChannels *
                   pAllocateArray->Height * pAllocateArray->Width;

    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

DONE:
  return ret;
}

CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  LOGGER(5, "Call Rewrite Function: cuArrayCreate_v2\n");
  CUresult ret;

  ret = cuArrayCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS) {
    goto DONE;
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate_v2, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = cuArrayCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS) {
    goto DONE;
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

static CUresult
cuArray3DCreate_helper(const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  size_t used = 0;
  size_t base_size = 0;
  size_t request_size = 0;
  CUresult ret = CUDA_SUCCESS;

  if (g_vcuda_config.enable) {
    base_size = get_array_base_size(pAllocateArray->Format);
    request_size = base_size * pAllocateArray->NumChannels *
                   pAllocateArray->Height * pAllocateArray->Width *
                   pAllocateArray->Depth;

    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

DONE:
  return ret;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = cuArray3DCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS) {
    goto DONE;
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate_v2, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = cuArray3DCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS) {
    goto DONE;
  }
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

CUresult
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMipmappedArrayCreate");
  size_t used = 0;
  size_t base_size = 0;
  size_t request_size = 0;
  CUresult ret;

  if (g_vcuda_config.enable) {
    base_size = get_array_base_size(pMipmappedArrayDesc->Format);
    request_size = base_size * pMipmappedArrayDesc->NumChannels *
                   pMipmappedArrayDesc->Height * pMipmappedArrayDesc->Width *
                   pMipmappedArrayDesc->Depth;

    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    if (unlikely(used + request_size > g_vcuda_config.gpu_memory)) {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMipmappedArrayCreate, pHandle,
                        pMipmappedArrayDesc, numMipmapLevels);
DONE:
  return ret;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  // 当开启了vgpu，则返回受限制的显存大小
  if (g_vcuda_config.enable) {
    *bytes = g_vcuda_config.gpu_memory;

    return CUDA_SUCCESS;
  }
  // 否则，直接调用api查询实际大小
  return CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (g_vcuda_config.enable) {
    *bytes = g_vcuda_config.gpu_memory;

    return CUDA_SUCCESS;
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem, bytes, dev);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemGetInfo_v2");
  size_t used = 0;

  if (g_vcuda_config.enable) {
    // 获取已使用的显卡内存
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);

    *total = g_vcuda_config.gpu_memory;
    // 当已使用大于分配量，可用量为0，否则为 分配的显存总量 - 已使用的显存
    *free =
        used > g_vcuda_config.gpu_memory ? 0 : g_vcuda_config.gpu_memory - used;

    return CUDA_SUCCESS;
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total);
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
  LOGGER(5, "Call Rewrite Function: %s\n", "cuMemGetInfo");
  size_t used = 0;
  // 当开启vcuda配置时
  if (g_vcuda_config.enable) {
    // 读取pid 配置文件 /etc/vcuda/pids.config
    atomic_action(pid_path, get_used_gpu_memory, (void *)&used);
    // 最大可用显存限制为 vcuda 分配时的大小
    *total = g_vcuda_config.gpu_memory;
    // 可用显存为 限制的最大显存 - 已使用显存
    *free =
        used > g_vcuda_config.gpu_memory ? 0 : g_vcuda_config.gpu_memory - used;

    return CUDA_SUCCESS;
  }
  // 当没有开启vcuda，直接调用原生的cuda api
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo, free, total);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra) {
  // 计算资源限制，根据网格数 更新当前cuda核心数
  rate_limiter(gridDimX * gridDimY * gridDimZ,
               blockDimX * blockDimY * blockDimZ);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel_ptsz, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
  // 计算资源限制，根据网格数 更新当前cuda核心数
  rate_limiter(gridDimX * gridDimY * gridDimZ,
               blockDimX * blockDimY * blockDimZ);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernelEx(CUlaunchConfig *config, CUfunction f, 
                        void **kernelParams, void **extra) {
  // TODO 利用率限制                      
  rate_limiter(config->gridDimX *config->gridDimY * config->gridDimZ,
               config->blockDimX * config->blockDimY * config->blockDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernelEx, 
                         config, f, kernelParams, extra);
}

CUresult cuLaunchKernelEx_ptsz(CUlaunchConfig *config, CUfunction f, 
                        void **kernelParams, void **extra) {
  // TODO 利用率限制         
  rate_limiter(config->gridDimX *config->gridDimY * config->gridDimZ,
               config->blockDimX * config->blockDimY * config->blockDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernelEx_ptsz, 
                         config, f, kernelParams, extra);
}

CUresult cuLaunch(CUfunction f) {
  // 计算资源限制，将当前cuda核心数减1
  rate_limiter(1, g_block_x * g_block_y * g_block_z);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  // 计算资源限制，根据网格数 更新当前cuda核心数
  rate_limiter(gridDimX * gridDimY * gridDimZ,
               blockDimX * blockDimY * blockDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel_ptsz, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams) {
  rate_limiter(gridDimX * gridDimY * gridDimZ,
               blockDimX * blockDimY * blockDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  rate_limiter(grid_width * grid_height, g_block_x * g_block_y * g_block_z);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f, grid_width,
                         grid_height);
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  rate_limiter(grid_width * grid_height, g_block_x * g_block_y * g_block_z);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGridAsync, f, grid_width,
                         grid_height, hStream);
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  if (g_vcuda_config.enable) {
    while (!CAS(&g_block_locker, 0, 1)) {
    }

    g_block_x = x;
    g_block_y = y;
    g_block_z = z;

    LOGGER(5, "Set block shape: %d, %d, %d", x, y, z);

    while (!CAS(&g_block_locker, 1, 0)) {
    }
  }
  return CUDA_ENTRY_CALL(cuda_library_entry, cuFuncSetBlockShape, hfunc, x, y,
                         z);
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemCreate, handle, size, prop,flags);
}

// 映射内存
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemMap, ptr, size, offset, handle, flags);
}
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemAddressReserve, ptr, size, alignment, addr, flags);
}
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemAddressFree, ptr, size);
}
CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemExportToShareableHandle, shareableHandle, handle, handleType, flags);
}
CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetAccess, flags, location, ptr);
}
CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetAllocationGranularity, granularity, prop, option);
}
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetAllocationPropertiesFromHandle, prop, handle);
}
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemImportFromShareableHandle, handle, osHandle, shHandleType);
}
CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemMapArrayAsync, mapInfoList, count, hStream);
}
CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemRelease, handle);
}
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemRetainAllocationHandle, handle, addr);
}
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemSetAccess, ptr, size, desc, count);
}
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    return CUDA_ENTRY_CALL(cuda_library_entry, cuMemUnmap, ptr, size);
}
