# ROCKET temporal partition benchmark

Machine: Apple M1 Pro; macOS-27.0.1-arm64-arm-64bit-Mach-O.
Median of 9 warmed repetitions; compilation excluded.
Speedup is kernel-only time / combined time at the **same CPU thread count**.
Scratch is the additional allocated partial-array payload, not RSS.

## prepared

| Backend | Case (N,T,K) | Threads | Original ms | 2 chunks ms (speedup) | 4 chunks ms (speedup) | Auto ms (chunks; speedup) | Scratch at 4 chunks KiB |
|---|---|---:|---:|---:|---:|---:|---:|
| cpp | single_short (1,128,1) | 1 | 0.0020 | 0.0021 (0.97×) | 0.0021 (0.97×) | 0.0020 (1; 1.01×) | 0.047 |
| numba | single_short (1,128,1) | 1 | 0.0024 | 0.0034 (0.69×) | 0.0034 (0.69×) | 0.0023 (1; 1.03×) | 0.047 |
| cpp | single_short (1,128,1) | 2 | 0.0020 | 0.0224 (0.09×) | 0.0219 (0.09×) | 0.0020 (1; 1.01×) | 0.047 |
| numba | single_short (1,128,1) | 2 | 0.0167 | 0.0303 (0.55×) | 0.0297 (0.56×) | 0.0165 (1; 1.01×) | 0.047 |
| cpp | single_short (1,128,1) | 4 | 0.0020 | 0.0216 (0.09×) | 0.0348 (0.06×) | 0.0020 (1; 1.00×) | 0.047 |
| numba | single_short (1,128,1) | 4 | 0.0262 | 0.0486 (0.54×) | 0.0488 (0.54×) | 0.0262 (1; 1.00×) | 0.047 |
| cpp | single_medium (1,8193,1) | 1 | 0.0285 | 0.0286 (1.00×) | 0.0286 (1.00×) | 0.0285 (1; 1.00×) | 0.047 |
| numba | single_medium (1,8193,1) | 1 | 0.0248 | 0.0259 (0.96×) | 0.0259 (0.96×) | 0.0248 (1; 1.00×) | 0.047 |
| cpp | single_medium (1,8193,1) | 2 | 0.0294 | 0.0469 (0.63×) | 0.0461 (0.64×) | 0.0291 (1; 1.01×) | 0.047 |
| numba | single_medium (1,8193,1) | 2 | 0.0352 | 0.0534 (0.66×) | 0.0520 (0.68×) | 0.0360 (1; 0.98×) | 0.047 |
| cpp | single_medium (1,8193,1) | 4 | 0.0294 | 0.0489 (0.60×) | 0.0656 (0.45×) | 0.0292 (1; 1.01×) | 0.047 |
| numba | single_medium (1,8193,1) | 4 | 0.0511 | 0.0858 (0.60×) | 0.0781 (0.65×) | 0.0512 (1; 1.00×) | 0.047 |
| cpp | single_long (1,262145,1) | 1 | 0.9890 | 0.9833 (1.01×) | 0.9938 (1.00×) | 0.9867 (1; 1.00×) | 0.047 |
| numba | single_long (1,262145,1) | 1 | 0.8626 | 0.8731 (0.99×) | 0.8611 (1.00×) | 0.8709 (1; 0.99×) | 0.047 |
| cpp | single_long (1,262145,1) | 2 | 0.9857 | 0.6663 (1.48×) | 0.6677 (1.48×) | 0.6580 (2; 1.50×) | 0.047 |
| numba | single_long (1,262145,1) | 2 | 0.8798 | 0.4979 (1.77×) | 0.5065 (1.74×) | 0.5037 (2; 1.75×) | 0.047 |
| cpp | single_long (1,262145,1) | 4 | 0.9895 | 0.6690 (1.48×) | 0.4866 (2.03×) | 0.4871 (4; 2.03×) | 0.047 |
| numba | single_long (1,262145,1) | 4 | 0.8801 | 0.5149 (1.71×) | 0.3074 (2.86×) | 0.3133 (4; 2.81×) | 0.047 |
| cpp | two_long_kernels (1,131073,2) | 1 | 1.0277 | 1.0252 (1.00×) | 1.0231 (1.00×) | 1.0335 (1; 0.99×) | 0.094 |
| numba | two_long_kernels (1,131073,2) | 1 | 1.0632 | 1.0464 (1.02×) | 1.0464 (1.02×) | 1.0574 (1; 1.01×) | 0.094 |
| cpp | two_long_kernels (1,131073,2) | 2 | 1.0356 | 0.7170 (1.44×) | 0.7177 (1.44×) | 0.7172 (2; 1.44×) | 0.094 |
| numba | two_long_kernels (1,131073,2) | 2 | 0.6838 | 0.7066 (0.97×) | 0.7127 (0.96×) | 0.6824 (1; 1.00×) | 0.094 |
| cpp | two_long_kernels (1,131073,2) | 4 | 1.0139 | 0.4499 (2.25×) | 0.4496 (2.25×) | 0.4499 (2; 2.25×) | 0.094 |
| numba | two_long_kernels (1,131073,2) | 4 | 0.6617 | 0.3834 (1.73×) | 0.3839 (1.72×) | 0.3826 (2; 1.73×) | 0.094 |
| cpp | eight_long_kernels (1,65537,8) | 1 | 2.1505 | 2.1658 (0.99×) | 2.1530 (1.00×) | 2.1641 (1; 0.99×) | 0.375 |
| numba | eight_long_kernels (1,65537,8) | 1 | 2.4337 | 2.4437 (1.00×) | 2.4277 (1.00×) | 2.4361 (1; 1.00×) | 0.375 |
| cpp | eight_long_kernels (1,65537,8) | 2 | 2.1808 | 1.2938 (1.69×) | 1.2913 (1.69×) | 1.2944 (2; 1.68×) | 0.375 |
| numba | eight_long_kernels (1,65537,8) | 2 | 1.4421 | 1.4609 (0.99×) | 1.4407 (1.00×) | 1.4218 (1; 1.01×) | 0.375 |
| cpp | eight_long_kernels (1,65537,8) | 4 | 2.1955 | 0.7695 (2.85×) | 0.7653 (2.87×) | 0.7871 (2; 2.79×) | 0.375 |
| numba | eight_long_kernels (1,65537,8) | 4 | 0.8408 | 0.8621 (0.98×) | 0.8586 (0.98×) | 0.8296 (1; 1.01×) | 0.375 |
| cpp | many_kernels (1,4097,64) | 1 | 0.9947 | 1.0002 (0.99×) | 1.0021 (0.99×) | 1.0005 (1; 0.99×) | 3.000 |
| numba | many_kernels (1,4097,64) | 1 | 1.1635 | 1.1613 (1.00×) | 1.1589 (1.00×) | 1.1700 (1; 0.99×) | 3.000 |
| cpp | many_kernels (1,4097,64) | 2 | 0.5415 | 0.5487 (0.99×) | 0.5482 (0.99×) | 0.5423 (1; 1.00×) | 3.000 |
| numba | many_kernels (1,4097,64) | 2 | 0.6273 | 0.6394 (0.98×) | 0.6433 (0.98×) | 0.6285 (1; 1.00×) | 3.000 |
| cpp | many_kernels (1,4097,64) | 4 | 0.3201 | 0.3224 (0.99×) | 0.3252 (0.98×) | 0.3213 (1; 1.00×) | 3.000 |
| numba | many_kernels (1,4097,64) | 4 | 0.3723 | 0.4043 (0.92×) | 0.4046 (0.92×) | 0.3690 (1; 1.01×) | 3.000 |
| cpp | many_samples (64,1025,8) | 1 | 2.0926 | 2.0949 (1.00×) | 2.1029 (1.00×) | 2.0935 (1; 1.00×) | 24.000 |
| numba | many_samples (64,1025,8) | 1 | 2.3272 | 2.3444 (0.99×) | 2.3751 (0.98×) | 2.3282 (1; 1.00×) | 24.000 |
| cpp | many_samples (64,1025,8) | 2 | 1.1342 | 1.2150 (0.93×) | 1.2215 (0.93×) | 1.1342 (1; 1.00×) | 24.000 |
| numba | many_samples (64,1025,8) | 2 | 1.3131 | 1.3363 (0.98×) | 1.3497 (0.97×) | 1.3142 (1; 1.00×) | 24.000 |
| cpp | many_samples (64,1025,8) | 4 | 1.1350 | 0.7278 (1.56×) | 0.7301 (1.55×) | 0.7278 (2; 1.56×) | 24.000 |
| numba | many_samples (64,1025,8) | 4 | 0.7601 | 0.7902 (0.96×) | 0.7966 (0.95×) | 0.7574 (1; 1.00×) | 24.000 |
| cpp | saturated_short (128,257,128) | 1 | 14.8650 | 15.4057 (0.96×) | 15.7188 (0.95×) | 14.9579 (1; 0.99×) | 768.000 |
| numba | saturated_short (128,257,128) | 1 | 18.7523 | 19.4042 (0.97×) | 20.0407 (0.94×) | 18.6723 (1; 1.00×) | 768.000 |
| cpp | saturated_short (128,257,128) | 2 | 7.7662 | 8.3765 (0.93×) | 8.4470 (0.92×) | 7.7263 (1; 1.01×) | 768.000 |
| numba | saturated_short (128,257,128) | 2 | 10.2147 | 10.5927 (0.96×) | 11.1182 (0.92×) | 10.1700 (1; 1.00×) | 768.000 |
| cpp | saturated_short (128,257,128) | 4 | 3.9931 | 4.4134 (0.90×) | 4.4912 (0.89×) | 3.9974 (1; 1.00×) | 768.000 |
| numba | saturated_short (128,257,128) | 4 | 5.3051 | 5.4978 (0.96×) | 5.6988 (0.93×) | 5.3133 (1; 1.00×) | 768.000 |
| cpp | mixed_long (8,16385,16) | 1 | 7.7453 | 7.7716 (1.00×) | 7.7770 (1.00×) | 7.7331 (1; 1.00×) | 6.000 |
| numba | mixed_long (8,16385,16) | 1 | 8.9645 | 8.9467 (1.00×) | 8.9570 (1.00×) | 8.9623 (1; 1.00×) | 6.000 |
| cpp | mixed_long (8,16385,16) | 2 | 4.3785 | 4.3949 (1.00×) | 4.3937 (1.00×) | 4.3807 (1; 1.00×) | 6.000 |
| numba | mixed_long (8,16385,16) | 2 | 4.9528 | 4.9638 (1.00×) | 4.9687 (1.00×) | 4.9511 (1; 1.00×) | 6.000 |
| cpp | mixed_long (8,16385,16) | 4 | 4.3802 | 2.5155 (1.74×) | 2.5163 (1.74×) | 2.5153 (2; 1.74×) | 6.000 |
| numba | mixed_long (8,16385,16) | 4 | 2.7690 | 2.7933 (0.99×) | 2.7948 (0.99×) | 2.7711 (1; 1.00×) | 6.000 |

## features

| Backend | Case (N,T,K) | Threads | Original ms | 2 chunks ms (speedup) | 4 chunks ms (speedup) | Auto ms (chunks; speedup) | Scratch at 4 chunks KiB |
|---|---|---:|---:|---:|---:|---:|---:|
| cpp | single_long (1,262145,1) | 1 | 2.5036 | 2.5126 (1.00×) | 2.5189 (0.99×) | 2.4976 (1; 1.00×) | 0.047 |
| numba | single_long (1,262145,1) | 1 | 2.4147 | 2.4225 (1.00×) | 2.4234 (1.00×) | 2.4206 (1; 1.00×) | 0.047 |
| cpp | single_long (1,262145,1) | 2 | 2.5174 | 2.2059 (1.14×) | 2.2689 (1.11×) | 2.2072 (2; 1.14×) | 0.047 |
| numba | single_long (1,262145,1) | 2 | 2.4558 | 2.1219 (1.16×) | 2.2057 (1.11×) | 2.1823 (2; 1.13×) | 0.047 |
| cpp | single_long (1,262145,1) | 4 | 2.6087 | 2.3834 (1.09×) | 2.2347 (1.17×) | 2.2421 (4; 1.16×) | 0.047 |
| numba | single_long (1,262145,1) | 4 | 2.5109 | 2.2831 (1.10×) | 2.0536 (1.22×) | 2.1020 (4; 1.19×) | 0.047 |
| cpp | saturated_short (128,257,128) | 1 | 155.9868 | 156.3640 (1.00×) | 155.8985 (1.00×) | 156.1573 (1; 1.00×) | 768.000 |
| numba | saturated_short (128,257,128) | 1 | 159.4321 | 159.9080 (1.00×) | 161.2313 (0.99×) | 158.6412 (1; 1.00×) | 768.000 |
| cpp | saturated_short (128,257,128) | 2 | 148.4183 | 149.4494 (0.99×) | 149.4500 (0.99×) | 148.0860 (1; 1.00×) | 768.000 |
| numba | saturated_short (128,257,128) | 2 | 149.4720 | 151.2784 (0.99×) | 152.8761 (0.98×) | 150.4197 (1; 0.99×) | 768.000 |
| cpp | saturated_short (128,257,128) | 4 | 143.6291 | 145.6724 (0.99×) | 143.4585 (1.00×) | 143.2360 (1; 1.00×) | 768.000 |
| numba | saturated_short (128,257,128) | 4 | 143.3064 | 147.1014 (0.97×) | 145.2346 (0.99×) | 144.1362 (1; 0.99×) | 768.000 |

## Interpretation

- `prepared`: identical normalized float32 input and identical pre-generated kernels. Includes allocations and each engine's existing boundary/packing costs. C++ validates native inputs; the internal Numba map assumes validated inputs.
- Prepared strategy selection is outside timing; `features` includes the public auto-policy decision on every call.
- `features`: full public transformation, identical raw data/seeds, application caches disabled. Includes normalization, sktime parameter generation, and feature assembly.
- Within each engine, every strategy must equal its untouched kernel-only output **exactly**, for PPV and MAX, before and after timing.
- Cross-engine PPV agreement/MAX rounding differences are recorded separately in JSON; existing fast-math differences are not attributed to temporal splitting.
- Auto uses the existing kernel/tile job count and minimum output length. It returns 1 when workers are already occupied or chunks would be too short.
- Worker-ID evidence comes from the actual temporal map with one sample and one kernel, not a separate empty parallel region.
- RSS includes imports, JIT and runtime pools; only partial-array payload is attributed exactly to temporal splitting.
