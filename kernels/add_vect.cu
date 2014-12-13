//	Calling a simple CUDA kernel
 


#include "add_vect.h"


//Vector dot summation:  C = A + B (template)
template <typename ValueType>
__global__ void
vdAdd_kernel(const int N,
			 const ValueType * A, 
			 const ValueType * B,
			 ValueType * C)
{
	
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
	
}



//Add two double vectors on GPU
//Eventually change the type
template <>
void  add_vect(const int N, const double *A, const double *B, double *C)
{
		const int BLOCK_SIZE = 512;
		const int NUM_BLOCKS =  N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0 : 1);
		
		vdAdd_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(N,A,B,C);
		
		
		return;
}
