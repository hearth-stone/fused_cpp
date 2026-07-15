// Measure concurrent packed-weight scan bandwidth without GEMM instructions.
//
// Each expert stream is split across a contiguous CPU team. Every worker owns
// disjoint cache lines from that stream, matching the fused MoE N-split weight
// ownership. Repeated passes model the weight revisit for successive M panels.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

constexpr std::size_t kCacheLine = 64;

struct Options {
    std::vector<int> cpu_ids;
    std::vector<int> groups;
    std::size_t stream_bytes = 16ULL << 20;
    int passes = 170;
    int warmup = 1;
    int runs = 7;
    std::string output_csv;
};

std::vector<int> parse_list(const std::string& text, bool allow_ranges) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) continue;
        const std::size_t dash = item.find('-');
        if (allow_ranges && dash != std::string::npos) {
            const int first = std::stoi(item.substr(0, dash));
            const int last = std::stoi(item.substr(dash + 1));
            if (first < 0 || last < first) {
                throw std::invalid_argument("invalid range: " + item);
            }
            for (int value = first; value <= last; ++value) {
                values.push_back(value);
            }
        } else {
            values.push_back(std::stoi(item));
        }
    }
    if (values.empty()) throw std::invalid_argument("empty integer list");
    return values;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg(argv[index]);
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[index];
        };
        if (arg == "--cpu-ids") {
            options.cpu_ids = parse_list(value(), true);
        } else if (arg == "--groups") {
            options.groups = parse_list(value(), false);
        } else if (arg == "--stream-mib") {
            options.stream_bytes = static_cast<std::size_t>(
                std::stod(value()) * static_cast<double>(1ULL << 20));
        } else if (arg == "--passes") {
            options.passes = std::stoi(value());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--runs") {
            options.runs = std::stoi(value());
        } else if (arg == "--output-csv") {
            options.output_csv = value();
        } else if (arg == "--help") {
            std::cout
                << "Usage: bench_weight_scan --cpu-ids 0-95 "
                   "--groups 1,2,3,4,6,8,12,16,24,32 [options]\n"
                << "  --stream-mib MiB  bytes in one expert stage (default 16)\n"
                << "  --passes N        repeated M-panel scans (default 170)\n"
                << "  --warmup N        untimed runs (default 1)\n"
                << "  --runs N          timed runs (default 7)\n"
                << "  --output-csv PATH optional structured output\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.cpu_ids.empty() || options.groups.empty()) {
        throw std::invalid_argument("--cpu-ids and --groups are required");
    }
    if (options.stream_bytes < kCacheLine ||
        options.stream_bytes % kCacheLine != 0) {
        throw std::invalid_argument("stream size must be a multiple of 64 bytes");
    }
    if (options.passes <= 0 || options.runs <= 0 || options.warmup < 0) {
        throw std::invalid_argument("passes/runs must be positive; warmup >= 0");
    }
    for (int groups : options.groups) {
        if (groups <= 0 || groups > static_cast<int>(options.cpu_ids.size())) {
            throw std::invalid_argument("groups must be in [1, number of CPUs]");
        }
    }
    return options;
}

void pin_thread(int cpu_id) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (error != 0) {
        throw std::runtime_error(
            "pthread_setaffinity_np failed: " + std::string(std::strerror(error)));
    }
#else
    (void)cpu_id;
#endif
}

class Barrier {
  public:
    explicit Barrier(int participants)
        : participants_(participants), remaining_(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const int generation = generation_;
        if (--remaining_ == 0) {
            ++generation_;
            remaining_ = participants_;
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [&] { return generation_ != generation; });
    }

  private:
    const int participants_;
    int remaining_;
    int generation_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
};

std::vector<int> balanced_shape(int cores, int groups) {
    const int base = cores / groups;
    const int extra = cores % groups;
    std::vector<int> shape(groups, base);
    for (int index = 0; index < extra; ++index) ++shape[index];
    return shape;
}

struct WorkerRange {
    std::size_t first_line;
    std::size_t line_count;
};

WorkerRange worker_range(
    int worker, const std::vector<int>& shape, std::size_t lines_per_stream) {
    int begin = 0;
    for (int group = 0; group < static_cast<int>(shape.size()); ++group) {
        const int end = begin + shape[group];
        if (worker < end) {
            const int local = worker - begin;
            const std::size_t base = lines_per_stream / shape[group];
            const std::size_t extra = lines_per_stream % shape[group];
            const std::size_t count = base + (local < static_cast<int>(extra));
            const std::size_t local_begin =
                static_cast<std::size_t>(local) * base +
                std::min<std::size_t>(local, extra);
            return {
                static_cast<std::size_t>(group) * lines_per_stream + local_begin,
                count,
            };
        }
        begin = end;
    }
    throw std::logic_error("worker does not belong to a group");
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const std::size_t low = static_cast<std::size_t>(position);
    const std::size_t high = std::min(low + 1, values.size() - 1);
    const double weight = position - static_cast<double>(low);
    return values[low] * (1.0 - weight) + values[high] * weight;
}

struct Result {
    int groups;
    int min_team_threads;
    int max_team_threads;
    std::size_t working_set_bytes;
    double median_ms;
    double p10_ms;
    double p90_ms;
    double bandwidth_gbs;
};

Result run_point(const Options& options, int groups) {
    const int cores = static_cast<int>(options.cpu_ids.size());
    const std::vector<int> shape = balanced_shape(cores, groups);
    const std::size_t total_bytes = options.stream_bytes * groups;
    const std::size_t total_lines = total_bytes / kCacheLine;
    const std::size_t lines_per_stream = options.stream_bytes / kCacheLine;
    void* allocation = nullptr;
    if (posix_memalign(&allocation, kCacheLine, total_bytes) != 0) {
        throw std::bad_alloc();
    }
    auto* data = static_cast<std::uint64_t*>(allocation);
    Barrier start_barrier(cores + 1);
    Barrier finish_barrier(cores + 1);
    std::atomic<std::uint64_t> checksum{0};
    const int iterations = options.warmup + options.runs;

    std::vector<std::thread> workers;
    workers.reserve(cores);
    for (int worker = 0; worker < cores; ++worker) {
        workers.emplace_back([&, worker] {
            pin_thread(options.cpu_ids[worker]);
            const WorkerRange range = worker_range(worker, shape, lines_per_stream);
            for (std::size_t line = range.first_line;
                 line < range.first_line + range.line_count; ++line) {
                data[line * (kCacheLine / sizeof(std::uint64_t))] =
                    static_cast<std::uint64_t>(line + 1);
            }
            std::uint64_t local = 0;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                start_barrier.wait();
                const volatile std::uint64_t* volatile_data = data;
                for (int pass = 0; pass < options.passes; ++pass) {
                    for (std::size_t line = range.first_line;
                         line < range.first_line + range.line_count; ++line) {
                        local += volatile_data[
                            line * (kCacheLine / sizeof(std::uint64_t))];
                    }
                }
                finish_barrier.wait();
            }
            checksum.fetch_add(local, std::memory_order_relaxed);
        });
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(options.runs);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        start_barrier.wait();
        finish_barrier.wait();
        const auto end = std::chrono::steady_clock::now();
        if (iteration >= options.warmup) {
            samples_ms.push_back(
                std::chrono::duration<double, std::milli>(end - begin).count());
        }
    }
    for (auto& worker : workers) worker.join();
    std::free(allocation);
    if (checksum.load(std::memory_order_relaxed) == 0 || total_lines == 0) {
        throw std::runtime_error("invalid scan checksum");
    }

    const double median_ms = percentile(samples_ms, 0.5);
    const double scanned_bytes =
        static_cast<double>(total_bytes) * options.passes;
    return {
        groups,
        *std::min_element(shape.begin(), shape.end()),
        *std::max_element(shape.begin(), shape.end()),
        total_bytes,
        median_ms,
        percentile(samples_ms, 0.1),
        percentile(samples_ms, 0.9),
        scanned_bytes / (median_ms * 1.0e6),
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::cout << "groups minT maxT workset_MiB median_ms p10_ms p90_ms GB/s\n";
        std::vector<Result> results;
        for (int groups : options.groups) {
            Result result = run_point(options, groups);
            results.push_back(result);
            std::cout << result.groups << ' ' << result.min_team_threads << ' '
                      << result.max_team_threads << ' '
                      << result.working_set_bytes / static_cast<double>(1ULL << 20)
                      << ' ' << result.median_ms << ' ' << result.p10_ms << ' '
                      << result.p90_ms << ' ' << result.bandwidth_gbs << '\n';
        }
        if (!options.output_csv.empty()) {
            std::ofstream output(options.output_csv);
            if (!output) throw std::runtime_error("cannot open output CSV");
            output << "active_streams,min_team_threads,max_team_threads,"
                      "stream_bytes,working_set_bytes,passes,median_ms,p10_ms,"
                      "p90_ms,bandwidth_gbs\n";
            for (const Result& result : results) {
                output << result.groups << ',' << result.min_team_threads << ','
                       << result.max_team_threads << ',' << options.stream_bytes
                       << ',' << result.working_set_bytes << ',' << options.passes
                       << ',' << result.median_ms << ',' << result.p10_ms << ','
                       << result.p90_ms << ',' << result.bandwidth_gbs << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
