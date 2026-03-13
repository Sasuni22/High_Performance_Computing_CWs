# High_Performance_Computing_CWs
High-Performance Computing Projects

This repository contains three HPC projects demonstrating parallel computing techniques using OpenMP, MPI, and CUDA. Each project focuses on matrix and vector operations to show performance improvements with parallelization.

Projects Overview
1️⃣ OpenMP – Matrix Addition with Performance Evaluation

Objective: Add two large matrices using both serial and parallel approaches with OpenMP.

Matrix Size: 2048 × 2048

2️⃣ MPI – Distributed Matrix Transpose

Objective: Transpose a large matrix across multiple MPI processes in a distributed memory setting.

Matrix Size: 1024 × 1024 or 2048 × 2048

Implementation Steps:

Divide the matrix row-wise among processes using MPI_Scatter.

Each process computes partial transpose of its segment.

Reconstruct full transposed matrix at root using MPI_Gather or MPI_Alltoall.

Output: Print a small snippet of the transposed matrix to verify correctness.

Purpose: Learn MPI data distribution and collective communication.

3️⃣ CUDA – Parallel Vector Dot Product

Objective: Compute a dot product of two large vectors using CUDA.

Implementation Steps:

Allocate and initialize two large vectors on the host.

Use a CUDA kernel for element-wise multiplication.

Perform shared memory reduction within thread blocks to sum partial products.

Final sum computed on the host (or with another kernel).

Output: Print the resulting dot product to verify correctness.

Purpose: Demonstrate GPU parallelism for vector operations.
