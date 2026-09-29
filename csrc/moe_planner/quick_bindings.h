// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

// Register the torch-free homogeneous quick planner on a Python extension.
// The same internal-stable binding is shared by _C and the MoE-only _moe_C.
void register_moe_quick_planner(pybind11::module_& m);
