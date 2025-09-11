#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int lim, int *odata, int *idata, int *blockSum) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            for (int d = 1; d <= lim; d++) {
                if (index >= (1 << (d - 1))) {
                    odata[index] = idata[index - (1 << (d - 1))] + idata[index];

                }
                else {
                    odata[index] = idata[index];
                }
                __syncthreads();
                idata[index] = odata[index];
            }
            
            if (index == n - 1 && blockSum != nullptr) {
                blockSum[0] = odata[index];
			}
		}

        __global__ void kernNaiveScanIteration(int n, int d, int* odata, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            if (index >= (1 << (d - 1))) {
                odata[index] = idata[index - (1 << (d - 1))] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernNaiveScanShift(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            odata[index] = (index == 0) ? 0 : idata[index - 1];
		}

        __global__ void kernAddBlockSums(int n, int* odata, int* blockSums) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
			odata[index] += blockSums[blockIdx.x];
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* d_idata;
			int* d_odata;
			cudaMalloc((void**)&d_idata, n * sizeof(int));
			cudaMalloc((void**)&d_odata, n * sizeof(int));
			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

			int blockSize = 1024;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);  


            if (n <= blockSize) {
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, ilog2ceil(n), d_odata, d_idata, nullptr);
            } else {
// Toggle between Multiple kernal launches and block number kernel launches
#define MUL 0

#if MUL         // Multiple kernel launches, faster
                for (int d = 1; d <= ilog2ceil(n); d++) {
                    kernNaiveScanIteration << <fullBlocksPerGrid, blockSize >> > (n, d, d_odata, d_idata);
                    std::swap(d_idata, d_odata); // Swap pointers
                }


#else           // Block number kernel launches, slower
                const int m = n % blockSize == 0 ? n / blockSize : n / blockSize + 1;
                int* d_iblockSums;
                cudaMalloc((void**)&d_iblockSums, m * sizeof(int));

                for (int i = 0; i < n; i += blockSize) {
                    kernNaiveScan << <1, blockSize >> > (blockSize, ilog2ceil(blockSize), d_odata + i, d_idata + i, d_iblockSums + i / blockSize);
                }

                int* d_oblockSums;
                cudaMalloc((void**)&d_oblockSums, m * sizeof(int));

                kernNaiveScan << <1, blockSize >> > (m, ilog2ceil(m), d_oblockSums, d_iblockSums, nullptr);
                kernNaiveScanShift << <1, blockSize >> > (m, d_oblockSums, d_oblockSums);


                kernAddBlockSums << <fullBlocksPerGrid, blockSize >> > (n, d_odata, d_oblockSums);
                kernAddBlockSums << <fullBlocksPerGrid, blockSize >> > (n, d_idata, d_oblockSums);
                cudaDeviceSynchronize();

                cudaFree(d_iblockSums);
                cudaFree(d_oblockSums);
#endif
            }

            kernNaiveScanShift << <fullBlocksPerGrid, blockSize >> > (n, d_odata, d_idata);

			cudaDeviceSynchronize();
			cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(d_idata);
			cudaFree(d_odata);

            timer().endGpuTimer();
        }
    }
}
