#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifndef FUSED_CPP_ENABLE_PROFILING
#define FUSED_CPP_ENABLE_PROFILING 1
#endif

namespace fused_cpp::profile {

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint now() {
  return Clock::now();
}

inline uint64_t now_ns() {
  const auto t = now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t).count());
}

inline double elapsed_ms(TimePoint start) {
  return std::chrono::duration<double, std::milli>(now() - start).count();
}

inline void add_elapsed(double* field, TimePoint start) {
  *field += elapsed_ms(start);
}

#if FUSED_CPP_ENABLE_PROFILING

inline bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return !(value[0] == '\0' || std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 ||
           std::strcmp(value, "FALSE") == 0);
}

#define FUSED_CPP_PROFILE_START(name) \
  ::fused_cpp::profile::TimePoint name = ::fused_cpp::profile::now()
#define FUSED_CPP_PROFILE_RESTART(name) name = ::fused_cpp::profile::now()
#define FUSED_CPP_PROFILE_ADD_IF_ENABLED(enabled, field, start) \
  do {                                                         \
    if (enabled) {                                             \
      ::fused_cpp::profile::add_elapsed(&(field), start);      \
    }                                                          \
  } while (0)
#define FUSED_CPP_PROFILE_ADD_IF_PTR(ptr, start)          \
  do {                                                    \
    if ((ptr) != nullptr) {                               \
      ::fused_cpp::profile::add_elapsed((ptr), start);    \
    }                                                     \
  } while (0)
#define FUSED_CPP_PROFILE_FIELD_PTR(enabled, field) ((enabled) ? &(field) : nullptr)
#define FUSED_CPP_PROFILE_IF_ENABLED(enabled, ...) \
  do {                                            \
    if (enabled) {                                \
      __VA_ARGS__;                                \
    }                                             \
  } while (0)

#else

#define FUSED_CPP_PROFILE_START(name) ((void)0)
#define FUSED_CPP_PROFILE_RESTART(name) ((void)0)
#define FUSED_CPP_PROFILE_ADD_IF_ENABLED(enabled, field, start) ((void)0)
#define FUSED_CPP_PROFILE_ADD_IF_PTR(ptr, start) ((void)0)
#define FUSED_CPP_PROFILE_FIELD_PTR(enabled, field) nullptr
#define FUSED_CPP_PROFILE_IF_ENABLED(enabled, ...) ((void)0)

#endif  // FUSED_CPP_ENABLE_PROFILING

}  // namespace fused_cpp::profile
