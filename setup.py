"""Build script for fused_cpp C++ extension."""

import ctypes
import glob
import os
import platform
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


_SUPPORTED_FIXED_SVE_VECTOR_BITS = (128, 256, 512, 1024, 2048)


def _cxx17_compile_flag(system=None) -> str:
    """Return the explicit C++17 flag for the active host compiler family."""
    system = platform.system() if system is None else system
    return "/std:c++17" if system == "Windows" else "-std=c++17"


def _detect_max_sve_vector_bits_for_build(prctl=None) -> int:
    """Return the largest SVE VL supported by the running Linux host.

    Linux exposes the supported range by rounding ``PR_SVE_SET_VL`` down to
    the largest available vector length. Restore the build thread's original
    configuration before returning so probing does not affect setuptools or
    compiler subprocesses.
    """
    pr_sve_set_vl = 50
    pr_sve_get_vl = 51
    pr_sve_vl_len_mask = 0xFFFF
    max_sve_vector_bytes = max(_SUPPORTED_FIXED_SVE_VECTOR_BITS) // 8

    if prctl is None:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.restype = ctypes.c_int

    original_config = prctl(pr_sve_get_vl, 0, 0, 0, 0)
    if original_config < 0:
        error = ctypes.get_errno()
        raise RuntimeError(
            "failed to query SVE vector length with PR_SVE_GET_VL "
            f"({os.strerror(error)}); set FUSED_CPP_SVE_VECTOR_BITS explicitly"
        )

    selected_config = prctl(pr_sve_set_vl, max_sve_vector_bytes, 0, 0, 0)
    if selected_config < 0:
        error = ctypes.get_errno()
        raise RuntimeError(
            "failed to probe the maximum SVE vector length with PR_SVE_SET_VL "
            f"({os.strerror(error)}); set FUSED_CPP_SVE_VECTOR_BITS explicitly"
        )

    restore_result = prctl(pr_sve_set_vl, original_config, 0, 0, 0)
    if restore_result < 0:
        error = ctypes.get_errno()
        raise RuntimeError(f"failed to restore the build thread's SVE vector length ({os.strerror(error)})")

    bits = (selected_config & pr_sve_vl_len_mask) * 8
    if bits not in _SUPPORTED_FIXED_SVE_VECTOR_BITS:
        raise RuntimeError(
            f"detected maximum SVE vector length {bits}, but fixed-length builds support only "
            f"{_SUPPORTED_FIXED_SVE_VECTOR_BITS}; set FUSED_CPP_SVE_VECTOR_BITS explicitly"
        )
    return bits


def _sve_vector_bits_for_build(detect_host_max: bool = True) -> int:
    raw = os.environ.get("FUSED_CPP_SVE_VECTOR_BITS")
    if raw is None:
        if not detect_host_max:
            return 128
        bits = _detect_max_sve_vector_bits_for_build()
        print(f"Detected maximum host SVE vector length: {bits} bits")
        return bits
    raw = raw.strip()
    try:
        bits = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"FUSED_CPP_SVE_VECTOR_BITS must be an integer, got {raw!r}") from exc
    if bits not in _SUPPORTED_FIXED_SVE_VECTOR_BITS:
        raise RuntimeError(f"FUSED_CPP_SVE_VECTOR_BITS must be one of {_SUPPORTED_FIXED_SVE_VECTOR_BITS}")
    return bits


class _BuildExtensionWithFixup(BuildExtension):
    """macOS 上构建完成后自动修复 dylib install name。

    ACL 的 scons 编译系统会把 libarm_compute.dylib 的 install name
    设置为相对路径（如 ``build/libarm_compute.dylib``），导致链接后的
    扩展模块在运行时找不到该库。此子类在构建完成后，使用
    ``install_name_tool`` 将相对路径替换为 ``@rpath/`` 前缀的路径。
    """

    def build_extensions(self) -> None:
        self._compile_native_sources()
        super().build_extensions()
        if platform.system() != "Darwin":
            return
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            if not os.path.isfile(ext_path):
                continue
            self._fix_dylib_paths(ext_path)

    @staticmethod
    def _fix_dylib_paths(so_path: str) -> None:
        """将 .so 中引用的相对路径 dylib 替换为 @rpath 形式。"""
        result = subprocess.run(
            ["otool", "-L", so_path],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines()[1:]:
            dep = line.strip().split()[0]
            # 跳过系统库和已经使用 @rpath 的依赖
            if dep.startswith(("/usr/lib/", "/System/", "@rpath/", "@loader_path/")):
                continue
            # 跳过绝对路径（已正确设置的依赖）
            if dep.startswith("/"):
                continue
            # 相对路径的 dylib，替换为 @rpath 形式
            basename = os.path.basename(dep)
            subprocess.run(
                ["install_name_tool", "-change", dep, f"@rpath/{basename}", so_path],
                check=True,
            )

    def _compile_native_sources(self) -> None:
        for ext in self.extensions:
            native_sources = native_sources_by_extension.get(ext.name, [])
            if not native_sources:
                continue
            obj_dir = os.path.join(self.build_temp, ext.name.replace(".", "_"), "native")
            os.makedirs(obj_dir, exist_ok=True)
            extra_objects = list(getattr(ext, "extra_objects", []) or [])
            for src, source_args in native_sources:
                relative = os.path.relpath(os.path.abspath(src), os.path.abspath(os.curdir))
                object_name = relative.replace("..", "up").replace(os.sep, "__") + ".o"
                obj = os.path.join(obj_dir, object_name)
                self._compile_one_native_source(src, obj, ext, source_args)
                if obj not in extra_objects:
                    extra_objects.append(obj)
            ext.extra_objects = extra_objects

    def _compile_one_native_source(self, src: str, obj: str, ext, source_args: list[str]) -> None:
        compiler_cmd = getattr(self.compiler, "compiler", None)
        compiler = compiler_cmd[0] if isinstance(compiler_cmd, list) else compiler_cmd
        compiler = os.environ.get("CC") or compiler or "cc"
        source_uses_cxx = os.path.splitext(src)[1].lower() in {
            ".cc",
            ".cpp",
            ".cxx",
            ".c++",
            ".mm",
        }

        extra_compile_args = getattr(ext, "extra_compile_args", []) or []
        if isinstance(extra_compile_args, dict):
            extra_compile_args = extra_compile_args.get("cxx", [])

        native_args = []
        keep_next = False
        for arg in extra_compile_args:
            if keep_next:
                native_args.append(arg)
                keep_next = False
                continue
            if arg == "-Xpreprocessor":
                native_args.append(arg)
                keep_next = True
                continue
            if arg.startswith(("-std=", "/std:")) and not source_uses_cxx:
                continue
            if arg == "-fopenmp" or arg.startswith(("-march=", "-mcpu=", "-O", "-I", "-std=", "/std:")):
                native_args.append(arg)

        native_args.extend(source_args)
        for name, value in getattr(ext, "define_macros", []) or []:
            native_args.append(f"-D{name}" if value is None else f"-D{name}={value}")

        if platform.system() != "Windows" and "-fPIC" not in native_args:
            native_args.append("-fPIC")
        include_args = [f"-I{inc}" for inc in (getattr(ext, "include_dirs", []) or [])]
        cmd = [compiler, "-c", src, "-o", obj, *include_args, *native_args]
        subprocess.run(cmd, check=True)


def _detect_acl():
    """检测 ARM Compute Library 是否可用。

    通过以下方式检测：
    1. 环境变量 ACL_ROOT 指定 ACL 安装路径
    2. 系统默认路径下查找 libarm_compute.so

    Returns:
        tuple: (is_available, include_dirs, library_dirs)
    """
    acl_root = os.environ.get("ACL_ROOT", "")

    if acl_root:
        lib_dir = os.path.join(acl_root, "lib")
        # 也检查 build 目录（ACL 源码编译的情况）
        if not os.path.isdir(lib_dir):
            lib_dir = os.path.join(acl_root, "build")

        # 检查两种目录布局：
        # 1. 源码树布局：ACL_ROOT/arm_compute/core/Types.h
        #    同时需要 ACL_ROOT/include/ 用于 half/half.hpp 等第三方依赖
        # 2. 安装后布局：ACL_ROOT/include/arm_compute/core/Types.h
        if os.path.isdir(os.path.join(acl_root, "arm_compute")):
            inc_dirs = [acl_root]
            third_party_inc = os.path.join(acl_root, "include")
            if os.path.isdir(third_party_inc):
                inc_dirs.append(third_party_inc)
            return True, inc_dirs, [lib_dir]
        include_dir = os.path.join(acl_root, "include")
        if os.path.isdir(os.path.join(include_dir, "arm_compute")):
            return True, [include_dir], [lib_dir]

    # 检查系统默认路径
    default_paths = ["/usr/include", "/usr/local/include"]
    for path in default_paths:
        if os.path.exists(os.path.join(path, "arm_compute")):
            return True, [], []

    return False, [], []


def _detect_openmp():
    """检测 OpenMP 可用性，并返回构建该扩展所需的编译/链接标志。

    跨平台行为：
      * **Linux**（GCC / 系统 Clang）：默认通过 ``-fopenmp`` 同时启用编译
        预处理与运行时链接。这是 GCC / 大多数 Linux Clang 的标准用法。
      * **macOS**（Apple Clang）：Apple Clang **不**直接接受 ``-fopenmp``，
        需要：
            -Xpreprocessor -fopenmp -I<libomp>/include   (compile)
            -L<libomp>/lib -lomp -Wl,-rpath,<libomp>/lib  (link)
        其中 ``<libomp>`` 的查找优先级为：
          1) PyTorch wheel 自带的 ``site-packages/torch/{include,lib}`` —— 与
             PyTorch 共用同一份 libomp.dylib，避免运行时出现
             ``OMP: Error #15: Initializing libomp.dylib, but found
             libomp.dylib already initialized.`` 这类双 OpenMP 运行时冲突。
          2) 用户通过 ``LIBOMP_ROOT`` 显式指定。
          3) Homebrew 默认路径 ``/opt/homebrew/opt/libomp`` 或
             ``/usr/local/opt/libomp``，以及 ``brew --prefix libomp`` 探测。

    用户控制：
      * ``FUSED_CPP_DISABLE_OMP=1``：完全跳过 OpenMP（不开启 ``_OPENMP``
        宏，``omp_info`` 会上报 ``has_openmp=false``）。
      * ``LIBOMP_ROOT=<path>``：在 macOS 上指定自定义 libomp 路径；其内部
        应包含 ``include/omp.h`` 与 ``lib/libomp.{dylib,a}``。

    Returns:
        tuple: (is_available, compile_args, link_args)
            * is_available (bool): 是否成功启用 OpenMP。
            * compile_args (list[str]): 追加到 ``extra_compile_args``。
            * link_args (list[str]): 追加到 ``extra_link_args``。
        当 is_available 为 False 时，两个 list 均为空。
    """
    if os.environ.get("FUSED_CPP_DISABLE_OMP", "0") == "1":
        return False, [], []

    system = platform.system()

    if system == "Linux":
        # GCC / Linux Clang 都支持 -fopenmp 一把梭：编译期定义 _OPENMP，
        # 链接期自动拉入 libgomp / libomp。
        return True, ["-fopenmp"], ["-fopenmp"]

    if system == "Darwin":
        # macOS 上的关键约束：PyTorch wheel 在进程加载时已经把它**自带**的
        # libomp.dylib 引入；如果我们再链接一个独立的 libomp（例如 Homebrew
        # 的 /opt/homebrew/opt/libomp），运行时会出现：
        #   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
        #                   already initialized.
        # 因此 macOS 上的优先级必须是：
        #   1) PyTorch 自带的 ``torch/lib/libomp.dylib`` + ``torch/include/omp.h``
        #      —— 与 PyTorch 共用同一份 OpenMP 运行时，最稳。
        #   2) 用户显式 ``LIBOMP_ROOT``（高级用户已自行处理冲突）。
        #   3) Homebrew / MacPorts（仅在 PyTorch 没附带时回退）。
        candidates = []

        # 1) PyTorch 自带的 libomp。CppExtension 会自动把
        #    ``site-packages/torch/lib`` 加到 -L 与 rpath，因此我们这里仅需
        #    确保头文件和 -lomp 标志被加上。
        try:
            import torch  # noqa: WPS433  setuptools 阶段 torch 必然可用

            torch_root = os.path.dirname(os.path.abspath(torch.__file__))
            candidates.append(("pytorch", torch_root))
        except Exception:  # noqa: BLE001
            pass

        # 2) 用户显式指定
        env_root = os.environ.get("LIBOMP_ROOT", "").strip()
        if env_root:
            candidates.append(("user", env_root))

        # 3) Homebrew 默认路径
        for brew_root in (
            "/opt/homebrew/opt/libomp",  # Apple Silicon
            "/usr/local/opt/libomp",  # Intel mac
        ):
            candidates.append(("brew", brew_root))

        # 4) brew --prefix 探测（用户可能装在非默认 prefix）
        try:
            brew_prefix = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if brew_prefix:
                candidates.append(("brew", brew_prefix))
        except (FileNotFoundError, OSError):
            pass

        for kind, root in candidates:
            if not root:
                continue
            inc = os.path.join(root, "include")
            lib = os.path.join(root, "lib")
            header_ok = os.path.isfile(os.path.join(inc, "omp.h"))
            libomp_dylib = os.path.join(lib, "libomp.dylib")
            libomp_a = os.path.join(lib, "libomp.a")
            lib_ok = os.path.isfile(libomp_dylib) or os.path.isfile(libomp_a)
            if not (header_ok and lib_ok):
                continue

            compile_args = [
                "-Xpreprocessor",
                "-fopenmp",
                f"-I{inc}",
            ]
            if kind == "pytorch" and os.path.isfile(libomp_dylib):
                # 关键：在 macOS 上，PyTorch wheel 自带的 libomp.dylib 的
                # ``LC_ID_DYLIB`` 通常是 ``/opt/llvm-openmp/lib/libomp.dylib``
                # （wheel 打包时残留的安装路径），而 PyTorch 自身的
                # ``libtorch_cpu.dylib`` 也是按这个 install_name 链接的。
                # 如果我们用 ``-L<torch/lib> -lomp`` 让链接器自行解析，
                # 链接器会按 ``@rpath/libomp.dylib`` 形式记录依赖；
                # 运行时 dyld 把 ``@rpath/libomp.dylib`` 与 PyTorch 内部
                # 已加载的 ``/opt/llvm-openmp/lib/libomp.dylib`` 视为不同
                # 模块，从而抛出：
                #   OMP: Error #15: Initializing libomp.dylib, but found
                #                   libomp.dylib already initialized.
                # 解决办法：把 libomp.dylib 的**绝对路径**作为链接输入
                # 文件传给 ld，这样 ld 会把它的 LC_ID_DYLIB（即
                # /opt/llvm-openmp/lib/libomp.dylib）原样写进扩展模块的
                # LC_LOAD_DYLIB 中，与 PyTorch 共享同一份运行时。
                link_args = [libomp_dylib]
            else:
                link_args = [
                    f"-L{lib}",
                    "-lomp",
                    f"-Wl,-rpath,{lib}",
                ]
            return True, compile_args, link_args

        # 没找到 libomp：静默关闭 OpenMP，omp_info 会上报 has_openmp=false。
        # 这是 README 中明确的“macOS 友好”行为，不应让 pip install 失败。
        return False, [], []

    # 其它系统（Windows 等）暂不主动启用，保留扩展性。
    return False, [], []


def _detect_kleidiai():
    """检测 KleidiAI 源码是否可用。

    通过以下方式检测：
    1. 环境变量 KLEIDIAI_ROOT 指定 KleidiAI 仓库根路径
    2. 默认相对路径 ``third_party/kleidiai``
    3. 相对工作区的兄弟目录 ``../kleidiai``

    Returns:
        tuple: (is_available, root_dir_or_empty)
    """
    candidates = []
    env_root = os.environ.get("KLEIDIAI_ROOT", "")
    if env_root:
        candidates.append(env_root)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "third_party", "kleidiai"))
    candidates.append(os.path.join(here, "..", "kleidiai"))

    for root in candidates:
        if not root:
            continue
        # 判定条件：存在 kai/kai_common.h 且存在 kai/ukernels/matmul 子目录
        if os.path.isfile(os.path.join(root, "kai", "kai_common.h")) and os.path.isdir(
            os.path.join(root, "kai", "ukernels", "matmul")
        ):
            return True, os.path.abspath(root)

    return False, ""


def _collect_kleidiai_sources(kai_root):
    """收集本项目所需的 KleidiAI 源文件列表（.c）。

    setuptools 在 editable 模式下要求 sources 必须是相对于 setup.py
    目录的路径（/-separated），不能使用绝对路径。因此这里将绝对路径
    转换为相对路径返回。

    :param kai_root: KleidiAI 仓库根目录（绝对路径）。
    :return: 相对于 setup.py 目录的源文件路径列表。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    rel_files = [
        # LHS online packing (bf16p8x4 from FP32)
        "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.c",
        # RHS offline packing (bf16p12x4biasf32 from FP32)
        "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.c",
        # FP32-output 8x12 BFMMLA microkernel
        # BF16 输出通过 FP32 微内核 + thread-local scratch 再转换实现，
        # 因此无需编译 f16 变体（f16 表示 IEEE half，并非 BF16）。
        "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.c",
    ]
    result_files = []
    for rel in rel_files:
        abs_path = os.path.join(kai_root, rel)
        if not os.path.isfile(abs_path):
            raise RuntimeError(f"KleidiAI source not found: {abs_path}")
        # 转换为相对于 setup.py 目录的相对路径
        rel_path = os.path.relpath(abs_path, here)
        result_files.append(rel_path)
    return result_files


def _host_cpu_has_flag(flag: str) -> bool:
    """Best-effort Linux host CPU flag probe for native optional features."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
    except OSError:
        return False
    return flag.lower() in text.replace("\n", " ").split()


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "off", "no")


def _profiling_enabled_for_build() -> bool:
    value = os.environ.get("FUSED_CPP_ENABLE_PROFILING")
    if value is not None:
        return value.strip().lower() not in ("", "0", "false", "off", "no")
    build_type = os.environ.get("FUSED_CPP_BUILD_TYPE", "").strip().lower()
    is_release = _env_truthy("FUSED_CPP_RELEASE") or build_type == "release"
    return not is_release


all_cpp_sources = sorted(glob.glob("csrc/**/*.cpp", recursive=True))
moe_source_prefix = os.path.join("csrc", "moe") + os.sep
sources = [source for source in all_cpp_sources if not source.startswith(moe_source_prefix)]
moe_sources = [source for source in all_cpp_sources if source.startswith(moe_source_prefix)]
bf16gemm_c_sources = []
bf16gemm_asm_sources = []
i8gemm_c_sources = []
i8gemm_asm_sources = []
moe_native_sources = []
deepseek_sve_gemm_native_sources = []
is_aarch64 = platform.machine() in ("aarch64", "arm64")
is_x86_64 = platform.machine() in ("x86_64", "AMD64")
acl_available, acl_include_dirs, acl_library_dirs = _detect_acl()
use_acl = is_aarch64 and acl_available

if is_aarch64:
    moe_sources = [source for source in moe_sources if os.path.join("moe", "x86") not in source]
    if platform.system() != "Linux":
        moe_sources = [source for source in moe_sources if os.path.join("arm", "sve_bf16") not in source]
else:
    moe_sources = [source for source in moe_sources if os.path.join("moe", "arm") not in source]

omp_available, omp_compile_args, omp_link_args = _detect_openmp()

kai_available, kai_root = _detect_kleidiai()
# Apple clang 会把 CppExtension 中追加的 KleidiAI .c 源按 C++ 编译，
# 导致部分合法 C 代码（如 const void* 到 const float* 的隐式转换）编译失败。
# 因此 KleidiAI 后端改为显式启用；默认保留 kai_gemm.cpp 的 stub 实现，
# 不影响 SDPA 等其它 fused_cpp 扩展构建。
kai_enabled = os.environ.get("FUSED_CPP_ENABLE_KLEIDIAI", "0") == "1"
use_kai = is_aarch64 and kai_available and kai_enabled

extra_compile_args = [_cxx17_compile_flag()]
extra_link_args = []
include_dirs = ["csrc"]
library_dirs = []
define_macros = []

bf16gemm_workspace = os.path.abspath("refs/i8gemm")
bf16gemm_lib = os.path.join(bf16gemm_workspace, "lib")
moe_include_dirs = ["csrc", bf16gemm_workspace, bf16gemm_lib]
moe_compile_args = [*omp_compile_args, "-O2", _cxx17_compile_flag()]
moe_link_args = list(omp_link_args)
moe_define_macros = [
    ("FUSED_CPP_ENABLE_PROFILING", "1" if _profiling_enabled_for_build() else "0"),
    ("FUSED_CPP_STRICT_MODE", "1" if _env_truthy("FUSED_CPP_STRICT_MODE") else "0"),
]
if omp_available:
    moe_define_macros.append(("FUSED_CPP_HAS_OMP", "1"))

if is_x86_64:
    moe_define_macros.append(("FUSED_CPP_MOE_HAS_X86_AVX512_BF16", "1"))
    # BuildExtension may invoke Ninja from its temporary directory, so this
    # external header-only dependency must use an absolute include path.
    xbyak_root = os.path.abspath(os.path.join("3rdparty", "xbyak"))
    # The first generator uses the System V x86-64 ABI. Windows keeps the
    # intrinsic path until its nonvolatile GPR/ZMM save contract is emitted.
    if platform.system() != "Windows" and os.path.isfile(os.path.join(xbyak_root, "xbyak", "xbyak.h")):
        moe_include_dirs.append(xbyak_root)
        moe_define_macros.append(("FUSED_CPP_MOE_HAS_XBYAK", "1"))
    avx512_bf16_source = os.path.join("csrc", "moe", "x86", "avx512_bf16", "kernels.cpp")
    moe_sources = [source for source in moe_sources if source != avx512_bf16_source]
    moe_native_sources.append(
        (
            avx512_bf16_source,
            [
                "-O3",
                "-std=c++17",
                "-mavx512f",
                "-mavx512bw",
                "-mavx512vl",
                "-mavx512bf16",
                "-mfma",
            ],
        )
    )

define_macros.append(("FUSED_CPP_ENABLE_PROFILING", "1" if _profiling_enabled_for_build() else "0"))
define_macros.append(("FUSED_CPP_STRICT_MODE", "1" if _env_truthy("FUSED_CPP_STRICT_MODE") else "0"))

# OpenMP：在 Linux 上默认启用 -fopenmp；在 macOS 上若检测到 Homebrew 安装的
# libomp 则启用，否则静默退化为单线程（omp_info 报告 has_openmp=false）。
if omp_available:
    extra_compile_args.extend(omp_compile_args)
    extra_link_args.extend(omp_link_args)
    define_macros.append(("FUSED_CPP_HAS_OMP", "1"))

# AArch64 公共编译标志：让 sdpa_flash2_neon_cache.cpp 等模块能拿到
# __ARM_FEATURE_BF16 / __ARM_FEATURE_MATMUL_INT8，从而走 BFMMLA / BFMLAL
# 主路径而不是 widen+FMLA 兜底路径。
#
# 平台默认值：
#   * macOS (Apple Silicon)：默认 -mcpu=apple-m2。Apple M2/M3/M4 同时具备
#     BF16 + MATMUL_INT8 能力，配合本仓库 sdpa_flash2_neon_cache.cpp 中
#     针对 Apple clang 的特殊判定（不依赖 __ARM_FEATURE_MATMUL_FP），可正常
#     生成 bfmmla 指令。M1 不支持 BF16，需显式覆盖。
#   * Linux aarch64：默认 -march=armv8.6-a+bf16+i8mm；若宿主
#     /proc/cpuinfo 暴露 sve，则自动升级为 -march=armv8.6-a+sve+bf16+i8mm，
#     让 SDPA softmax exp 走 SVE poly6。
#
# 用户可通过环境变量 FUSED_CPP_TARGET_CPU 覆盖，例如：
#   FUSED_CPP_TARGET_CPU=apple-m1   → 退回 widen+FMLA 路径，兼容 M1
#   FUSED_CPP_TARGET_CPU="armv9-a+sve2+bf16+i8mm" → SVE2 平台
# 取值若以 `apple-` 或 `cortex-` 开头则按 -mcpu= 处理，否则按 -march= 处理。
if is_aarch64:
    include_dirs.extend([bf16gemm_workspace, bf16gemm_lib])
    moe_define_macros.append(("FUSED_CPP_HAS_BF16GEMM", "1"))
    if omp_available:
        bf16gemm_c_sources.append(os.path.join(bf16gemm_lib, "bf16gemm_mt.c"))
        define_macros.append(("FUSED_CPP_HAS_BF16GEMM", "1"))
    bf16gemm_asm_sources.append(os.path.join(bf16gemm_lib, "bf16gemm_k.S"))
    bf16gemm_asm_sources.append(os.path.join(bf16gemm_lib, "bf16gemm_k_bias.S"))
    bf16gemm_asm_sources.append(os.path.abspath(os.path.join("csrc", "moe", "arm", "neon_bf16", "kernels.S")))

    moe_native_sources.extend(
        [
            (os.path.join(bf16gemm_lib, "bf16gemm_k.S"), []),
            (os.path.join(bf16gemm_lib, "bf16gemm_k_bias.S"), []),
            (os.path.abspath(os.path.join("csrc", "moe", "arm", "neon_bf16", "kernels.S")), []),
        ]
    )

    if platform.system() == "Darwin":
        moe_compile_args.append("-mcpu=apple-m2")
    else:
        moe_compile_args.append("-march=armv8.6-a+bf16+i8mm")

    target_cpu = os.environ.get("FUSED_CPP_TARGET_CPU", "").strip()
    if target_cpu:
        if target_cpu.startswith(("apple-", "cortex-", "neoverse-")):
            extra_compile_args.append(f"-mcpu={target_cpu}")
        else:
            extra_compile_args.append(f"-march={target_cpu}")
    elif platform.system() == "Darwin":
        extra_compile_args.append("-mcpu=apple-m2")
    else:
        features = ["bf16", "i8mm"]
        if _host_cpu_has_flag("sve"):
            features.insert(0, "sve")
        extra_compile_args.append("-march=armv8.6-a+" + "+".join(features))
    extra_compile_args.append("-O2")

    target_has_sve = (
        "sve" in target_cpu.lower() if target_cpu else platform.system() != "Darwin" and _host_cpu_has_flag("sve")
    )
    if platform.system() == "Linux":
        sve_vector_bits = _sve_vector_bits_for_build(detect_host_max=target_has_sve)
        moe_define_macros.append(("FUSED_CPP_MOE_HAS_ARM_SVE", "1"))
        moe_define_macros.append(("FUSED_CPP_MOE_SVE_VECTOR_BITS", str(sve_vector_bits)))
        sve_args = [
            "-march=armv8.6-a+sve+bf16+i8mm",
            f"-msve-vector-bits={sve_vector_bits}",
            "-O2",
            "-std=c++17",
        ]
        sve_sources = [
            os.path.join("csrc", "moe", "arm", "sve_bf16", "jit_kernels.cpp"),
            os.path.join("csrc", "moe", "arm", "sve_bf16", "packing.cpp"),
            os.path.join("csrc", "moe", "arm", "sve_bf16", "route_merge.cpp"),
        ]
        moe_sources = [source for source in moe_sources if source not in sve_sources]
        if target_has_sve:
            indexer_sve_source = os.path.join("csrc", "deepseek_v4_indexer_sve.cpp")
            sources = [source for source in sources if source != indexer_sve_source]
            deepseek_sve_gemm_native_sources.append((indexer_sve_source, sve_args))
        xbyak_aarch64_root = os.path.abspath(os.path.join("3rdparty", "xbyak_aarch64"))
        xbyak_aarch64_sources = [
            os.path.join(xbyak_aarch64_root, "src", "xbyak_aarch64_impl.cpp"),
            os.path.join(xbyak_aarch64_root, "src", "util_impl.cpp"),
        ]
        xbyak_aarch64_available = all(os.path.isfile(source) for source in xbyak_aarch64_sources)
        if xbyak_aarch64_available:
            xbyak_aarch64_include_dirs = [
                xbyak_aarch64_root,
                os.path.join(xbyak_aarch64_root, "src"),
                os.path.join(xbyak_aarch64_root, "xbyak_aarch64"),
            ]
            moe_include_dirs.extend(xbyak_aarch64_include_dirs)
            include_dirs.extend(xbyak_aarch64_include_dirs)
            moe_define_macros.append(("FUSED_CPP_MOE_HAS_XBYAK_AARCH64", "1"))
            define_macros.append(("FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM", "1"))
            shared_sve_jit_args = [
                *sve_args,
                "-DFUSED_CPP_MOE_HAS_ARM_SVE=1",
                f"-DFUSED_CPP_MOE_SVE_VECTOR_BITS={sve_vector_bits}",
                "-DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1",
            ]
            deepseek_sve_gemm_native_sources.extend(
                [
                    *[(source, shared_sve_jit_args) for source in sve_sources[:2]],
                    *[(source, ["-O2", "-std=c++17"]) for source in xbyak_aarch64_sources],
                ]
            )
        else:
            moe_define_macros.append(("FUSED_CPP_MOE_HAS_XBYAK_AARCH64", "0"))
        moe_native_sources.extend(
            [
                (os.path.abspath(os.path.join("csrc", "moe", "arm", "sve_bf16", "kernels.S")), sve_args),
                *[(source, sve_args) for source in sve_sources],
                *(
                    [(source, ["-O2", "-std=c++17"]) for source in xbyak_aarch64_sources]
                    if xbyak_aarch64_available
                    else []
                ),
            ]
        )
    i8gemm_backend = "sve" if target_has_sve else "neon"
    i8gemm_required = [
        os.path.join(bf16gemm_lib, "i8gemm.h"),
        os.path.join(bf16gemm_lib, "i8gemm_pack_a_neon.S"),
    ]
    if i8gemm_backend == "sve":
        i8gemm_required.extend(
            [
                os.path.join(bf16gemm_lib, "i8gemm_sve.c"),
                os.path.join(bf16gemm_lib, "i8gemm_sve.S"),
                os.path.join(bf16gemm_lib, "i8gemm_hybrid.S"),
            ]
        )
    else:
        i8gemm_required.extend(
            [
                os.path.join(bf16gemm_lib, "i8gemm_mt.c"),
                os.path.join(bf16gemm_lib, "i8gemm_k.S"),
                os.path.join(bf16gemm_lib, "i8gemm_k_bias.S"),
            ]
        )
    if omp_available and all(os.path.isfile(p) for p in i8gemm_required):
        if i8gemm_backend == "sve":
            i8gemm_c_sources.append(os.path.join(bf16gemm_lib, "i8gemm_sve.c"))
            i8gemm_asm_sources.extend(
                [
                    os.path.join(bf16gemm_lib, "i8gemm_sve.S"),
                    os.path.join(bf16gemm_lib, "i8gemm_hybrid.S"),
                    os.path.join(bf16gemm_lib, "i8gemm_pack_a_neon.S"),
                ]
            )
        else:
            i8gemm_c_sources.append(os.path.join(bf16gemm_lib, "i8gemm_mt.c"))
            i8gemm_asm_sources.extend(
                [
                    os.path.join(bf16gemm_lib, "i8gemm_k.S"),
                    os.path.join(bf16gemm_lib, "i8gemm_k_bias.S"),
                    os.path.join(bf16gemm_lib, "i8gemm_pack_a_neon.S"),
                ]
            )
        define_macros.append(("FUSED_CPP_HAS_I8GEMM", "1"))
        define_macros.append(("FUSED_CPP_I8GEMM_BACKEND", f'"{i8gemm_backend}"'))

if use_acl:
    define_macros.append(("FUSED_CPP_HAS_ACL", "1"))
    include_dirs.extend(acl_include_dirs)
    library_dirs.extend(acl_library_dirs)
    # 链接 ACL 库
    extra_link_args.append("-larm_compute")
    # 检测 libarm_compute_core 是否存在（较新版本 ACL 已合并到 libarm_compute 中）
    has_core_lib = any(
        os.path.exists(os.path.join(d, f"libarm_compute_core{ext}"))
        for d in acl_library_dirs
        for ext in (".so", ".dylib", ".a")
    )
    if has_core_lib:
        extra_link_args.append("-larm_compute_core")
    # 设置 rpath 以便运行时找到动态库
    if platform.system() == "Darwin":
        for lib_dir in acl_library_dirs:
            extra_link_args.append(f"-Wl,-rpath,{lib_dir}")
    else:
        for lib_dir in acl_library_dirs:
            extra_link_args.append(f"-Wl,-rpath,{lib_dir}")
else:
    # 非 ACL 环境下排除 ACL 相关源文件
    sources = [s for s in sources if "acl_" not in os.path.basename(s)]

if use_kai:
    # 把 KleidiAI 头文件根目录加入 include_dirs，支持 #include "kai/..."
    include_dirs.append(kai_root)
    # 把 KleidiAI 的 .c 源文件追加到编译列表（绝对路径，避免相对路径歧义）
    sources.extend(_collect_kleidiai_sources(kai_root))
    define_macros.append(("FUSED_CPP_HAS_KLEIDIAI", "1"))
    # KleidiAI BFMMLA 内核所需的 BF16 + FP16 能力已经由前面的 AArch64 默认
    # -mcpu=apple-m2 / -march=armv8.6-a+bf16+i8mm 提供，这里不再追加 -march=
    # 以免覆盖（多个 -march/-mcpu 时编译器以最后一个为准，会反向降级 BFMMLA
    # 路径）。仅追加 KleidiAI 自身需要的标志。
    extra_compile_args.extend(
        [
            "-O2",
            # KleidiAI 的 .c 源文件中存在 void* -> T* 的隐式转换，
            # 在 C 中合法但 C++ 中不允许。PyTorch CppExtension 统一使用
            # C++ 编译器编译所有源文件，因此需要 -fpermissive 来容忍此类转换。
            "-fpermissive",
        ]
    )
else:
    # 未启用 KleidiAI 时，不追加外部 KleidiAI C 源，保留 csrc/kai_gemm.cpp
    # 编译其 stub 实现，避免 module.cpp 中的 KAI 绑定出现未解析符号。
    pass

native_sources_by_extension = {
    "fused_cpp._C": [
        (source, [])
        for source in [
            *bf16gemm_c_sources,
            *bf16gemm_asm_sources,
            *i8gemm_c_sources,
            *i8gemm_asm_sources,
        ]
    ]
    + deepseek_sve_gemm_native_sources,
    "fused_cpp._moe_C": moe_native_sources,
}

main_extension = CppExtension(
    name="fused_cpp._C",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)
moe_extension = CppExtension(
    name="fused_cpp._moe_C",
    sources=moe_sources,
    include_dirs=moe_include_dirs,
    extra_compile_args=moe_compile_args,
    extra_link_args=moe_link_args,
    define_macros=moe_define_macros,
)
extensions = [moe_extension] if _env_truthy("FUSED_CPP_BUILD_MOE_ONLY") else [main_extension, moe_extension]

setup(
    packages=[
        *find_packages(where="src"),
        *find_packages(where=".", include=("cpu_moe_schedule_optimization", "cpu_moe_schedule_optimization.*")),
    ],
    package_dir={"": "src", "cpu_moe_schedule_optimization": "cpu_moe_schedule_optimization"},
    ext_modules=extensions,
    cmdclass={"build_ext": _BuildExtensionWithFixup},
)
