CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Yiding Tian
  *  [LinkedIn](https://linkedin.com/in/ytian1109), [Github](https://github.com/tonytgrt)
* Tested on: Windows 11 24H2, i7-13900H @ 4.1GHz, 32GB RAM @ 4800Mhz, MSI Shadow RTX 5080 @ 3200MHz (OC +400MHz) 16GB VRAM @ 16801MHz (OC +2000MHz) Driver 581.15, Personal Laptop with External Desktop GPU via NVMe connector (PCIe 4.0 x4 Protocol)

## Features

### 1. CPU Scan & Stream Compaction
A simple for loop for the cpu scan. For cpu stream compaction, for loops are used for both scan and scatter. 

### 2. GPU Naive Scan
Implemented a simple `kernNaiveScanIteration` kernel for the naive scan. Following the outline in parallel algorithms class slides, the kernel does simple array manipulation of:
```
  for all k in parallel
    if (k >= 2d-1)
      x[k] = x[k – 2d-1] + x[k];
```
Here `d` is the depth of the naive scan iterations, the for loop of `for d = 1 to log2n` is called from the scan function on the host to ensure proper synchronization across different blocks. 

### 3. GPU Efficient Scan & Stream Compaction
There are two versions of my efficient scan. When array size `n <= blockSize`, a single kernel of `kernEffScan` is called only once from the host. It is an implementation migrated from the GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html) as provided to us in `INSTRUCTION.md`. As the article suggests, `kernEffScan` uses shared memory, and thus can only handle arrays within 1 block. It has the upsweep and downsweep phases combined within itself. When array size `n > blockSize`, upsweep and downsweep are seperated into `kernUpSweep` and `kernDownSweep`. These kernels are called multiple times in a for loop inside the host function, where each iteration reduces the number of active threads being invoked by the kernel to optimize performance. The two kernels uses global memory so that they can handle across blocks.  
For Stream Compaction, two simple kernels of `kernMapToBoolean` and `kernScatter` are implemented along with a wrapper for the host scan function. The algorithm is very similar to its cpu counterpart only with parallel computation differing from the sequential for loops.

### 4. Thrust Invocation
Calling the Thrust library for performance metrics comparison.

### 5. Extra Credits
#### 5.1 Why is My GPU Approach So Slow?
As seen in the Performance Analysis part below, my naive scan on gpu matches the cpu run time in smaller array sizes, and slightly outperforms the cpu in larger array sizes. Both efficient scan and efficient compaction outperforms the CPU, where efficient scan is very close to the Thrust scan in run time. 

The key optimizations implemented:
- **Dynamic thread reduction**: In both `kernUpSweep` and `kernDownSweep`, I calculate the number of active threads at each level and only launch the necessary threads
- **Early thread termination**: Threads that don't have work exit early based on their index
- **Optimized block launching**: The number of blocks launched decreases with each level of the tree traversal
- **Shared memory utilization**: For small arrays (n ≤ blockSize), using shared memory significantly improves cache locality

These optimizations ensure that GPU resources are not wasted on idle threads, leading to the improved performance observed.

#### 5.2 Shared Memory (partly)
Shared memory is used in Efficient scan and Efficient compact when array size `n <= blockSize`.

## Performance Analysis

### 1. Optimal Blocksize
`blockSize = 256` is found to be optimal for the scan and compact algorithms. In fact, the Thrust scan implementation also uses block size of 128, 256, and 384 as profiled in NSight Compute, suggesting that smaller `blockSize` is more suitable for the scan tasks due to better shared memory utilization and occupancy.
![](/img/i-compute-thrust-sum.png)

### 2. Time Comparison

#### 2.1 Scan
Time cost comparison between scan implementations. 
![](/img/i-scan-time.png)
**Key Observations:**
- CPU scan shows linear scaling with array size (appears exponential in chart because the horizontal axis scales exponentially)
- Naive GPU scan initially underperforms CPU for small arrays due to kernel launch overhead
- Efficient scan consistently outperforms naive scan by ~30%
- Thrust scan maintains best performance across all sizes with optimized memory access patterns

#### 2.2 Stream Compaction
Time cost comparison between stream compaction implementations.
![](/img/i-compact-time.png)
**Key Observations:**
- CPU compaction without scan is fastest for small arrays (<= 2^21)
- GPU efficient compaction overtakes CPU at larger array sizes
- The crossover point is around 2^22 elements, where parallelization benefits outweigh kernel overhead

### 3. Algorithm Performance Analysis in NSight Compute
The program was profiled with array size `n = 1 << 26` in NSight Compute with data metrics set to full.

#### 3.1 Naive Scan
In Naive Scan, each kernel launch have the same block size since each time we are processing the entire array. 
![](/img/i-compute-naive-sum.png)

Throughput for naive scan shows a low SM thoughput and a high memory thoughput, indicating that we are bound by memory bandwidth and are under utilizing the compute power.
![](/img/i-compute-naive-detail.png)

A detailed memory chart shows intensive device memory read/writes, while the L1 Cache is only hit 37.5% of the time.
![](/img/i-compute-naive-mem.png)

**Performance Bottleneck:** Memory bandwidth limited, with poor cache utilization due to strided memory access patterns.

#### 3.2 Efficient Scan
In efficient scan there are two mirrored phases of upsweep and downsweep. I will analyze the upsweep performance here. Downsweep follows the same idea. See how Grid Size and Block Size exponentially decreases as we proceed deeper into the scan. This critically improves performance over the naive method. 
![](/img/i-compute-ups-sum.png)

Similarly, efficient scan is still bound by the memory throughput as with naive scan.
![](/img/i-compute-ups-detail.png)

In memory chart, efficient scan hit the L1 Cache much more often than the naive scan with a 66.62% compared to previous 37.5%.
![](/img/i-compute-ups-mem.png)

**Performance Bottleneck:** Still memory bandwidth limited, but improved cache locality through coalesced memory access patterns.

#### 3.3 Thrust
As shown earlier, the thrust implementation of scan only has 3 kernel calls, a sharp contrast with the `log n` kernel calls I implemented. The scan kernel of thrust has a higher compute throughput than both of my implementation as well, which makes sense as it performs the best across all array sizes.
![](/img/i-compute-thrust-sum.png)

Focusing on the details of DeviceScanKernel, thrust also appears to be bound by memory with its high memory throughput and low compute throughput, which alligns with scan algorithm's nature of simple array manipulations that requires low compute power but large memory bandwidth for large arrays.
![](/img/i-compute-thrust-detail.png)

Quite noticably, thrust does not hit the L1 cache at all with a 0% hit rate. Instead, it utilizes shared memory as indicated with the 6.80M Inst in the graph below. This ensures its best performance across all implementations.
![](/img/i-compute-thrust-mem.png)

**Performance Bottleneck:** Optimally memory bandwidth limited with shared memory usage eliminating cache misses.

#### 3.4 Performance Bottleneck Analysis Summary

| Implementation | Primary Bottleneck | Secondary Issue | Optimization Opportunity |
|----------------|-------------------|-----------------|-------------------------|
| CPU Scan | Sequential processing | Cache misses on large arrays | N/A (inherently sequential) |
| Naive GPU | Memory bandwidth | Poor cache utilization (37.5%) | Shared memory, coalescing |
| Efficient GPU | Memory bandwidth | Moderate cache utilization (66.6%) | Shared memory, bank conflicts |
| Thrust | Memory bandwidth (optimal) | N/A | Already optimized |

All GPU implementations are memory I/O bound rather than compute bound, which is expected for scan algorithms that perform simple addition operations. The key differentiator is memory access efficiency:
- **Naive**: Strided, non-coalesced access
- **Efficient**: Partially coalesced with tree-based access
- **Thrust**: Fully optimized with shared memory and bank conflict avoidance

## Output

```
Found 2 CUDA device(s):
Device 0: NVIDIA GeForce RTX 5080
  Compute capability: 12.0
  Total memory: 16302 MB
  SM count: 84
Device 1: NVIDIA GeForce RTX 4070 Laptop GPU
  Compute capability: 8.9
  Total memory: 8187 MB
  SM count: 36

Using device 0, Array Size n = 67108864
===============================================


****************
** SCAN TESTS **
****************
    [  22  14  10   7   2  44   2  30   3  21  23  33  30 ...  24   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 99.1891ms    (std::chrono Measured)
    [   0  22  36  46  53  55  99 101 131 134 155 178 211 ... 1643574074 1643574098 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 97.8035ms    (std::chrono Measured)
    [   0  22  36  46  53  55  99 101 131 134 155 178 211 ... 1643573974 1643573986 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 105.267ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 101.563ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 89.9345ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 88.712ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 85.9899ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 85.3373ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   2   3   0   3   1   1   1   3   1   1   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 118.745ms    (std::chrono Measured)
    [   3   2   3   3   1   1   1   3   1   1   2   3   3 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 116.897ms    (std::chrono Measured)
    [   3   2   3   3   1   1   1   3   1   1   2   3   3 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 291.995ms    (std::chrono Measured)
    [   3   2   3   3   1   1   1   3   1   1   2   3   3 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 99.2683ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 95.2448ms    (CUDA Measured)
    passed


========================================
CSV Performance Data (for graphing)
========================================

Size,CPU_Scan,Naive_Scan,Efficient_Scan,Thrust_Scan,CPU_Compact,Efficient_Compact
2097152,3.610,4.074,3.369,3.362,3.711,4.417
4194304,7.462,6.825,6.066,6.099,7.416,6.826
8388608,15.552,12.475,11.335,11.939,14.838,12.785
16777216,28.930,26.139,22.018,22.296,29.275,24.355
33554432,58.623,52.444,44.357,43.141,58.317,49.582
67108864,118.409,106.432,89.887,85.140,118.276,96.416
134217728,242.728,261.904,175.625,169.533,239.148,189.516
268435456,475.805,456.818,355.820,335.181,478.533,408.221
536870912,1007.188,952.551,704.023,675.127,995.752,817.421

Performance data saved to results.csv
========================================
```

## Anecdotes, Challenges, and Reflections

### 1. Naive Scan 
In a class I had a conversation with Mr. Mohammed about where the for loop of `for d = 1 to log2n` should be - is it called inside the kernel or from the host. Mr. Mohammed suggested it should be inside the kernel. In the same class I went on to implement the naive scan with for loop inside the kernel and the kernel being called only once from host. Then immediately I found it would not work when array size `n > blockSize`. I went on to discuss with Mr. Mohammed after class. He explained to me that since `__syncthreads()` will not work across different blocks, I should refer to the last few slides of the parallel algorithms to implement that method for larger array sizes to work. It took me a long time to implement the method discussed as `Scan on Arrays of Arbitrary Length` for the naive scan. Then when I tested it, the algorithm was surprisingly slow: often 10 times slower than the CPU for larger arrays. I found that this algorithm essentially takes `O(n)` to run as it divides the array into multiple subarrays of `blockSize`. Then I reflected back on my first discussion about the location of for loop, and came up with the current solution where I place the for loop in the host function, contrary to what Mr. Mohammed suggested. For larger arrays, by putting the for loop out and calling the kernel to each time perform a single parallel operation across multiple blocks, I can ensure the proper synchronization across blocks, while only invoking the kernel for `O(log n)` times, making the final run time of my naive scan implementation close to the CPU instead of the previous 10x worse. 

### 2. Efficient Scan
Story for efficient scan was similar. I first implemented a version using shared memory as outlined in the GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html), but can only handle array size for up to `n <= blockSize`. When I tried to find a solution for larger arrays, I first migrated the `Scan on Arrays of Arbitrary Length` again. It also gave me similarly bad performance, and lots of bugs like it would fail for `n > 1 << 24`. Then I went on to divide the `kernEffScan` that includes the upSweep and downSweep in one kernel to seperate kernels of `kernUpSweep` and `kernDownSweep`, and got the implementation we have.

### 3. Memory Access Patterns
One of the key insights from this project was understanding how critical memory access patterns are for GPU performance. The difference between strided access (naive) and coalesced access (efficient) resulted in nearly 2x improvement in L1 cache hit rates. This reinforces the importance of designing algorithms with GPU memory hierarchy in mind.

### 4. Thread Utilization
The challenge of keeping all threads busy throughout the algorithm execution was particularly evident in the tree-based algorithms. The solution of dynamically reducing active threads at each level was crucial for achieving reasonable performance, though Thrust's approach of using persistent thread blocks shows there's still room for improvement.

## Future Work

1. **Complete Radix Sort Implementation**: Implement the bit-wise radix sort using the efficient scan as a building block
2. **Bank Conflict Optimization**: Add padding to shared memory accesses to eliminate bank conflicts
3. **Warp-Level Primitives**: Explore using warp shuffle instructions for intra-warp scans
4. **Multi-GPU Support**: Extend the implementation to work across multiple GPUs for very large arrays
5. **Template Generalization**: Make the algorithms work with arbitrary data types, not just integers