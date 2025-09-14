#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEffScan(int* g_odata, int* g_idata, int n, int* blockSum) {
            extern __shared__ int temp[];
            int thid = threadIdx.x;
            int offset = 1;
            temp[2 * thid] = g_idata[2 * thid];
            temp[2 * thid + 1] = g_idata[2 * thid + 1];

            for (int d = n >> 1; d > 0; d = d >> 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            if (thid == 0) {
                if (blockSum) {
                    blockSum[0] = temp[n - 1];  // Save total sum of this block
                }
                temp[n - 1] = 0;
            }

            for (int d = 1; d < n; d *= 2) {
                offset = offset >> 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }

            __syncthreads();

            if (!blockSum) { 
                // exclusive scan
                g_odata[2 * thid] = temp[2 * thid];
                g_odata[2 * thid + 1] = temp[2 * thid + 1];
            }
            else {
				// inclusive scan
                g_odata[2 * thid] = temp[2 * thid + 1];
                g_odata[2 * thid + 1] = temp[2 * thid + 2];
                if (thid == 0) {
                    g_odata[n - 1] = blockSum[0];
				}
            }
            
        }


        __global__ void kernAddBlockSums(int n, int* odata, int blockSum) {
            int index = threadIdx.x;

            odata[2 * index] += blockSum;
            odata[2 * index + 1] += blockSum;
        }

        __global__ void kernShiftRight(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            odata[index] = (index == 0) ? 0 : idata[index - 1];
        }

        int nextPowerOf2(int n) {
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++;
            return n;
        }

        __global__ void kernUpSweep(int n, int* data, int level) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int stride = 1 << (level + 1);
			int ai = stride * (index + 1) - 1;

            if (ai < n) {
                int bi = ai - (1 << level);
                data[ai] += data[bi];
			}
        }

		__global__ void kernDownSweep(int n, int* data, int level) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			int stride = 1 << (level + 1);
			int ai = stride * (index + 1) - 1;

			if (ai < n) {
				int bi = ai - (1 << level);
				int t = data[bi];
				data[bi] = data[ai];
				data[ai] += t;
			}
		}

        __global__ void kernSetLastZero(int n, int* data) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                data[n - 1] = 0;
            }
		}

#define MUL 1

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int paddedSize = nextPowerOf2(n);

            int* d_idata;
            int* d_odata;
            cudaMalloc((void**)&d_idata, paddedSize * sizeof(int));
            cudaMalloc((void**)&d_odata, paddedSize * sizeof(int));

            cudaMemset(d_idata, 0, paddedSize * sizeof(int));
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
			const int blockSize = 1024;
            int numThreads = paddedSize / 2;
			

            if (paddedSize <= 2048) {
                kernEffScan<<<1, numThreads, n * sizeof(int)>>>(d_odata, d_idata, paddedSize, nullptr);
                cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            } else {
#if MUL
                int* d_data;
                cudaMalloc((void**)&d_data, paddedSize * sizeof(int));
                cudaMemset(d_data, 0, paddedSize * sizeof(int));
                cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

                // For large arrays, use multiple kernel launches
                int depth = ilog2ceil(paddedSize);

                // Up-sweep phase
                for (int level = 0; level < depth; level++) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    kernUpSweep << <numBlocks, blockSize >> > (paddedSize, d_data, level);
                }

                // Set last element to 0 for exclusive scan
                kernSetLastZero << <1, 1 >> > (paddedSize, d_data);

                // Down-sweep phase
                for (int level = depth - 1; level >= 0; level--) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    kernDownSweep << <numBlocks, blockSize >> > (paddedSize, d_data, level);
                }

                // Copy result from device to host
                cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(d_data);

#else
                dim3 numBlocks((paddedSize + 1023) / 1024);
                const int m = paddedSize / 2048;
                int* d_iblockSums;
                cudaMalloc((void**)&d_iblockSums, m * sizeof(int));

                for (int i = 0; i < n; i += 2048) {
					kernEffScan << <1, 1024, 2048 * sizeof(int) >> > (d_odata + i, d_idata + i, 2048, d_iblockSums + i / 2048);
                }

				int* blockSums = new int[m];
				cudaMemcpy(blockSums, d_iblockSums, m * sizeof(int), cudaMemcpyDeviceToHost);
                int a = blockSums[1];

				int* d_oblockSums;
				cudaMalloc((void**)&d_oblockSums, m * sizeof(int));

				kernEffScan << <1, m / 2, m * sizeof(int) >> > (d_oblockSums, d_iblockSums, m, nullptr);

                int* blockSums1 = new int[m];
                cudaMemcpy(blockSums1, d_oblockSums, m * sizeof(int), cudaMemcpyDeviceToHost);
                int a1 = blockSums1[1];
                a1 = blockSums1[2];

                for (int i = 0; i < n; i += 2048) {
					kernAddBlockSums << <1, 1024 >> > (2048, d_odata + i, blockSums1[i / 2048]);
                }
                
                cudaMemcpy(d_idata, d_odata, paddedSize * sizeof(int), cudaMemcpyDeviceToDevice);
				kernShiftRight << <numBlocks, 1024 >> > (paddedSize, d_odata, d_idata);

				cudaFree(d_iblockSums);
				cudaFree(d_oblockSums);

                cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
#endif
            }

			

			cudaFree(d_idata);
			cudaFree(d_odata);

            timer().endGpuTimer();
        }

        // Wrapper for compact, removes timer and takes in device pointers
        void scanCompact(int n, int* odata, const int* idata) {
            // TODO
            int paddedSize = nextPowerOf2(n);

            int* d_idata;
            int* d_odata;
            cudaMalloc((void**)&d_idata, paddedSize * sizeof(int));
            cudaMalloc((void**)&d_odata, paddedSize * sizeof(int));

            cudaMemset(d_idata, 0, paddedSize * sizeof(int));
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();

			const int blockSize = 1024;
            int numThreads = paddedSize / 2;

            if (paddedSize <= 2048) {
                kernEffScan << <1, numThreads, n * sizeof(int) >> > (d_odata, d_idata, paddedSize, nullptr);
                cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            else {
#if MUL
                int* d_data;
                cudaMalloc((void**)&d_data, paddedSize * sizeof(int));
                cudaMemset(d_data, 0, paddedSize * sizeof(int));
                cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

                // For large arrays, use multiple kernel launches
                int depth = ilog2ceil(paddedSize);

                // Up-sweep phase
                for (int level = 0; level < depth; level++) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    kernUpSweep << <numBlocks, blockSize >> > (paddedSize, d_data, level);
                }

                // Set last element to 0
                kernSetLastZero << <1, 1 >> > (paddedSize, d_data);

                // Down-sweep phase
                for (int level = depth - 1; level >= 0; level--) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    kernDownSweep << <numBlocks, blockSize >> > (paddedSize, d_data, level);
                }

                // Copy to output
                cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(d_data);
#else
                dim3 numBlocks((paddedSize + 1023) / 1024);
                const int m = paddedSize / 2048;
                int* d_iblockSums;
                cudaMalloc((void**)&d_iblockSums, m * sizeof(int));

                for (int i = 0; i < n; i += 2048) {
                    kernEffScan << <1, 1024, 2048 * sizeof(int) >> > (d_odata + i, d_idata + i, 2048, d_iblockSums + i / 2048);
                }

                int* blockSums = new int[m];
                cudaMemcpy(blockSums, d_iblockSums, m * sizeof(int), cudaMemcpyDeviceToHost);
                int a = blockSums[1];

                int* d_oblockSums;
                cudaMalloc((void**)&d_oblockSums, m * sizeof(int));

                kernEffScan << <1, m / 2, m * sizeof(int) >> > (d_oblockSums, d_iblockSums, m, nullptr);

                int* blockSums1 = new int[m];
                cudaMemcpy(blockSums1, d_oblockSums, m * sizeof(int), cudaMemcpyDeviceToHost);
                int a1 = blockSums1[1];
                a1 = blockSums1[2];

                for (int i = 0; i < n; i += 2048) {
                    kernAddBlockSums << <1, 1024 >> > (2048, d_odata + i, blockSums1[i / 2048]);
                }

                cudaMemcpy(d_idata, d_odata, paddedSize * sizeof(int), cudaMemcpyDeviceToDevice);
                kernShiftRight << <numBlocks, 1024 >> > (paddedSize, d_odata, d_idata);

                cudaFree(d_iblockSums);
                cudaFree(d_oblockSums);
                cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
#endif
            }

            
            cudaDeviceSynchronize();

            cudaFree(d_idata);
            cudaFree(d_odata);

        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int* d_idata;
			int* d_odata;

			int* d_bools;
			int* d_indices;

			cudaMalloc((void**)&d_idata, n * sizeof(int));
			cudaMalloc((void**)&d_odata, n * sizeof(int));
			cudaMalloc((void**)&d_bools, n * sizeof(int));
			cudaMalloc((void**)&d_indices, n * sizeof(int));

			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

			int blockSize = 1024;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, d_bools, d_idata);
			cudaDeviceSynchronize();

			scanCompact(n, d_indices, d_bools);
			cudaDeviceSynchronize();

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, d_odata, d_idata, d_bools, d_indices);
			cudaDeviceSynchronize();

			int lastBool, lastIndex;
			cudaMemcpy(&lastBool, d_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&lastIndex, d_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

			int count = lastBool + lastIndex;
			cudaMemcpy(odata, d_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			cudaFree(d_idata);
			cudaFree(d_odata);
			cudaFree(d_bools);
			cudaFree(d_indices);

            timer().endGpuTimer();
            return count;
        }
    }
}
