CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Yiding Tian
  *  [LinkedIn](https://linkedin.com/in/ytian1109), [Github](https://github.com/tonytgrt)
* Tested on: Windows 11 24H2, i7-13900H @ 4.1GHz, 32GB RAM @ 4800Mhz, MSI Shadow RTX 5080 @ 3200MHz (OC +400MHz) 16GB VRAM @ 16801MHz (OC +2000MHz) Driver 581.15, Personal Laptop with External Desktop GPU via NVMe connector (PCIe 4.0 x4 Protocol)

## Features

### 1. CPU Scan & Stream Compaction

### 2. GPU Naive Scan

### 3. GPU Efficient Scan & Stream Compaction

### 4. Thrust Invocation

### 5. Extra Credits
#### 5.1 Why is My GPU Approach So Slow?

#### 5.2 Shared Memory (partly)

## Performance Analysis

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

## Challenges and Thoughts