# MUTLASS 0.1.1

_MUTLASS 0.1.1 - September 2024_

MUTLASS(MUSA Templates for Linear Algebra Subroutines) is a header-only library for implementing high-performance matrix-matrix multiplication (GEMM) within MUSA(**M**eta-computing **U**nified **S**ystem **A**rchitecture). It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement muDNN.

See the [Quick Start Guide](./media/docs/quickstart.md) to get started quickly.

Note: MUTLASS uses the CuTe library, introduced in CUTLASS 3.x, as the backend, and thus is incompatible with most implementations of CUTLASS 2.x.

# What's in MUTLASS 0.1.1

MUTLASS 0.1.1 is an open-release version based on CUTLASS 3.5 providing:

- [MuTe](./include/mute), a core library and backend adapted from CUTLASS CuTe

- Quyuan Features

  - MMA primitives: TensorFloat32, BFloat16, Float16, INT8

- FMA/MMA GEMM Kernels targeting the Quyuan architecture

  - Note: this is a beta release. Further updates to MUTLASS will include performance improvements, feature enablement, and possible breaking changes to the API

- MUTLASS Profiler, Library, and Utilities

- Two examples that demonstrate the usage of the [low-level API](./examples/00_basic_gemm) and the [collective builders](./examples/01_quyuan_gemm_with_collective_builder) to build GEMM kernels


Minimum requirements:

- Architecture: Quyuan

- Compiler: MCC 3.1.0

- MUSA Toolkit version: 3.1.0


# Documentation

- [Quick Start Guide](./media/docs/quickstart.md) - build and run MUTLASS


# Building MUTLASS

MUTLASS is a header-only template library and does not need to be built to be used by other projects. Client applications should target MUTLASS's `include/` directory in their include paths.

MUTLASS unit tests, examples, and utilities can be build with CMake. The minimum version of CMake is given in the [QuickStart guide](./media/docs/quickstart.md).

Create a build directory within the MUTLASS project, then run CMake. By default MUTLASS will build kernels for MUSA architecture version 2.2.
