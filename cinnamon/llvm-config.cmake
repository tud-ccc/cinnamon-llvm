# How LLVM is configured for Cinnamon (https://github.com/tud-ccc/Cinnamon).
#
# Both the prebuilt LLVM that .github/workflows/cinnamon-prebuilt.yml publishes
# and Cinnamon's own source builds of this repository (its build-llvm.sh)
# configure LLVM with `cmake -C cinnamon/llvm-config.cmake`, so an option
# changed here applies to both. Settings that depend on the machine (Python,
# ccache, parallelism) are passed on the command line instead.

set(LLVM_ENABLE_PROJECTS "mlir;llvm" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host;AArch64" CACHE STRING "")
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "SPIRV" CACHE STRING "")

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(BUILD_SHARED_LIBS ON CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_EH ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")

set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "")
# Installs FileCheck, not, count: Cinnamon's tests need them from an installed
# LLVM too.
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

# Every optional dependency enabled here has to be found again by each project
# that uses the installed LLVM. Cinnamon needs none of them but zlib, which
# lets llvm-symbolizer read the compressed debug info below.
set(LLVM_ENABLE_ZLIB FORCE_ON CACHE STRING "")
set(LLVM_ENABLE_ZSTD OFF CACHE STRING "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE STRING "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")

# Line tables only, compressed: crash backtraces get file and line numbers for
# a fraction of the size of full debug info.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -g1 -gz" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g1 -gz" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-gz" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-gz" CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "-gz" CACHE STRING "")
