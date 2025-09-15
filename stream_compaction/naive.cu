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

			if (!blockSum) return;
            
            if (index == n - 1) {
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
                for (int d = 1; d <= ilog2ceil(n); d++) {
                    kernNaiveScanIteration << <fullBlocksPerGrid, blockSize >> > (n, d, d_odata, d_idata);
                    std::swap(d_idata, d_odata); // Swap pointers
                }
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
