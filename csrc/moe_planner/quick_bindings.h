/*
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

// Register the torch-free homogeneous quick planner on a Python extension.
// The same internal-stable binding is shared by _C and the MoE-only _moe_C.
void register_moe_quick_planner(pybind11::module_& m);
