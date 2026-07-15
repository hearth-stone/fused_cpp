/*
 * ACL 核心绑定策略控制
 *
 * 提供运行时动态控制 GEMM 计算的核心绑定策略（线程数和绑定的 CPU 核心范围），
 * 支持通过 pthread_setaffinity_np 绑定线程到指定核心范围，
 * 并通过 ACL Scheduler 设置并行线程数。
 *
 * 编译条件：核心绑定功能仅在 Linux 上可用，ACL 线程数设置仅在 AArch64 上可用。
 */

#include <torch/extension.h>
#include <tuple>
#include <mutex>
#include <cstdio>

#ifdef __linux__
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <cerrno>
#include <cstring>
#endif

#ifdef __aarch64__
#include "arm_compute/runtime/Scheduler.h"
using namespace arm_compute;
#endif

// 模块级静态变量，保存当前核心绑定配置
static std::mutex g_affinity_mutex;
static int g_core_start = -1;  // -1 表示未设置
static int g_core_end = -1;
static int g_num_threads = -1;  // -1 表示使用默认值

void set_acl_thread_affinity(int64_t core_start, int64_t core_end, int64_t num_threads) {
  std::lock_guard<std::mutex> lock(g_affinity_mutex);

  // 校验参数
  if (core_start >= core_end) {
    fprintf(stderr,
            "WARNING: set_acl_thread_affinity: core_start(%ld) >= "
            "core_end(%ld)，忽略此次设置\n",
            static_cast<long>(core_start), static_cast<long>(core_end));
    return;
  }

#ifdef __linux__
  long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (core_end > num_cpus) {
    fprintf(stderr,
            "WARNING: set_acl_thread_affinity: core_end(%ld) 超出系统核心数"
            "(%ld)，忽略此次设置\n",
            static_cast<long>(core_end), num_cpus);
    return;
  }

  // 通过 pthread_setaffinity_np 绑定当前线程到指定核心范围
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int64_t i = core_start; i < core_end; ++i) {
    CPU_SET(static_cast<int>(i), &mask);
  }
  int ret = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (ret != 0) {
    fprintf(stderr, "WARNING: pthread_setaffinity_np(cores %ld-%ld) 失败: %s\n", static_cast<long>(core_start),
            static_cast<long>(core_end - 1), strerror(ret));
    return;
  }

  // 更新保存的配置
  g_core_start = static_cast<int>(core_start);
  g_core_end = static_cast<int>(core_end);
#else
  fprintf(stderr,
          "WARNING: set_acl_thread_affinity: 核心绑定功能仅在 Linux 上可用，"
          "仅设置线程数\n");
  g_core_start = static_cast<int>(core_start);
  g_core_end = static_cast<int>(core_end);
#endif

  // 设置 ACL Scheduler 线程数
  if (num_threads > 0) {
#ifdef __aarch64__
    Scheduler::get().set_num_threads(static_cast<unsigned int>(num_threads));
#endif
    g_num_threads = static_cast<int>(num_threads);
  }
}

std::tuple<int64_t, int64_t, int64_t> get_acl_thread_affinity() {
  std::lock_guard<std::mutex> lock(g_affinity_mutex);
  return std::make_tuple(static_cast<int64_t>(g_core_start), static_cast<int64_t>(g_core_end),
                         static_cast<int64_t>(g_num_threads));
}
