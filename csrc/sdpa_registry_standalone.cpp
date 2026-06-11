#include "sdpa_common.h"

#if defined(FUSED_CPP_SDPA_STANDALONE)

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct SdpaRegistry {
    std::unordered_map<std::string, SdpaKernelFn> map;
    std::vector<std::string> names;
};

SdpaRegistry& registry() {
    static SdpaRegistry inst;
    return inst;
}

}  // namespace

int sdpa_register_version(const char* name, SdpaKernelFn fn) {
    if (name == nullptr || fn == nullptr) {
        throw std::runtime_error(
            "sdpa_register_version: name and fn must be non-null");
    }
    auto& reg = registry();
    std::string key(name);
    if (reg.map.find(key) != reg.map.end()) {
        throw std::runtime_error(
            "sdpa_register_version: duplicate version name '" + key + "'");
    }
    reg.map.emplace(key, fn);
    reg.names.emplace_back(std::move(key));
    return 0;
}

std::vector<std::string> sdpa_list_versions() {
    return registry().names;
}

SdpaKernelFn sdpa_find_kernel(const std::string& name) {
    auto& reg = registry();
    auto it = reg.map.find(name);
    if (it == reg.map.end()) {
        return nullptr;
    }
    return it->second;
}

void sdpa_dispatch(const std::string& name, const SdpaParams& p) {
    auto fn = sdpa_find_kernel(name);
    if (fn == nullptr) {
        std::string msg = "SDPA version '" + name +
                          "' is not registered; available: [";
        const auto& names = registry().names;
        for (size_t i = 0; i < names.size(); ++i) {
            if (i) msg += ", ";
            msg += "'" + names[i] + "'";
        }
        msg += "]";
        throw std::runtime_error(msg);
    }
    fn(p);
}

#endif  // FUSED_CPP_SDPA_STANDALONE

