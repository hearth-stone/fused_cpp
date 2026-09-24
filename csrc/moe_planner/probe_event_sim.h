// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace moe_planner {

// One task kind of the probe-calibrated event model (ProbeEventModel): an expert with a given
// route count on a given team width. Python derives every field from the calibration so the
// native loop evaluates exactly the model's per-event formula.
struct ProbeEventProfile {
  std::vector<double> phase_ns;    // isolated time per phase (phase_table)
  std::vector<char> phase_load;    // 1 for a loading (cold-B) phase
  double overhead_ns = 0.0;        // per-expert overhead run before the phases
  int width = 0;
  double steady_load_cores = 0.0;  // cores counted as loading while in a steady phase
  double steady_steady_cores = 0.0;
  int ll_table = -1;  // curve tables, indexed by round(cores * resolution)
  int ls_table = -1;
  int sl_table = -1;
  int ss_table = -1;
  bool windowed = false;
  double r_cal = 1.0;
  double e_cal = 0.0;
};

struct ProbeEventResult {
  double makespan_ns = 0.0;
  std::vector<double> finish_ns;
};

// C++ port of ProbeEventModel.simulate without background width factors, event logs or
// calibration hooks (the Python model keeps those paths).
class NativeProbeEventSim {
 public:
  NativeProbeEventSim(int resolution, double g0);

  int add_table(std::vector<double> values);
  int add_profile(ProbeEventProfile profile);
  ProbeEventResult simulate(const std::vector<int>& profiles, const std::vector<std::vector<int>>& dependencies) const;
  std::vector<double> makespans(const std::vector<std::vector<int>>& profiles,
                                const std::vector<std::vector<std::vector<int>>>& dependencies, int workers) const;

 private:
  int resolution_;
  double g0_;
  std::vector<std::vector<double>> tables_;
  std::vector<ProbeEventProfile> profiles_;
};

}  // namespace moe_planner
