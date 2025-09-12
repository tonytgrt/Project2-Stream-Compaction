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

        __global__ void kernEffScan(int* g_odata, int* g_idata, int n) {
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

            g_odata[2 * thid] = temp[2 * thid];
            g_odata[2 * thid + 1] = temp[2 * thid + 1];
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
            cudaDeviceSynchronize();

            int numThreads = paddedSize / 2;

            if (paddedSize <= 2048) {
                kernEffScan<<<1, numThreads, n * sizeof(int)>>>(d_odata, d_idata, paddedSize);
            }
            else {

            }

			cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

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

            int numThreads = paddedSize / 2;

            if (paddedSize <= 2048) {
                kernEffScan << <1, numThreads, n * sizeof(int) >> > (d_odata, d_idata, paddedSize);
            }
            else {

            }

            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
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
