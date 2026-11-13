#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

__global__ void GenerateInstances(
	float3 *m, float3 *s, float4 *q,
	int number_of_Gaussians,
	uint64_t GAS,
	float *instances
) {
	extern __shared__ float tmp[];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int wid = tid >> 5;
	int number_of_warps = (number_of_Gaussians - 1) >> 5;

	// *********************************************************************************************

	if (wid <= number_of_warps) {
		int index = ((tid < number_of_Gaussians) ? tid : (number_of_Gaussians - 1));

		// *****************************************************************************************

		float3 m_param = m[index];
		float3 s_param = s[index];
		float4 q_param = q[index];

		// *****************************************************************************************

		float aa = q_param.x * q_param.x;
		float bb = q_param.y * q_param.y;
		float cc = q_param.z * q_param.z;
		float dd = q_param.w * q_param.w;
		float s = 2.0f / (aa + bb + cc + dd);

		float bs = q_param.y * s;  float cs = q_param.z * s;  float ds = q_param.w * s;
		float ab = q_param.x * bs; float ac = q_param.x * cs; float ad = q_param.x * ds;
		bb = bb * s;			   float bc = q_param.y * cs; float bd = q_param.y * ds;
		cc = cc * s;			   float cd = q_param.z * ds;       dd = dd * s;

		float Q11 = s_param.x * (1.0f - cc - dd);
		float Q12 = s_param.y * (bc - ad);
		float Q13 = s_param.z * (bd + ac);

		float Q21 = s_param.x * (bc + ad);
		float Q22 = s_param.y * (1.0f - bb - dd);
		float Q23 = s_param.z * (cd - ab);

		float Q31 = s_param.x * (bd - ac);
		float Q32 = s_param.y * (cd + ab);
		float Q33 = s_param.z * (1.0f - bb - cc);

		// *****************************************************************************************

		float *base_address = &tmp[(threadIdx.x * 20) + (threadIdx.x >> 3)];

		// transform
		base_address[0] = Q11;
		base_address[1] = Q12;
		base_address[2] = Q13;
		base_address[3] = m_param.x;

		base_address[4] = Q21;
		base_address[5] = Q22;
		base_address[6] = Q23;
		base_address[7] = m_param.y;

		base_address[8] = Q31;
		base_address[9] = Q32;
		base_address[10] = Q33;
		base_address[11] = m_param.z;

		// instanceId
		base_address[12] = 0.0f;

		// sbtOffset
		base_address[13] = 0.0f;

		// visibilityMask
		base_address[14] = __uint_as_float(255);

		// flags
		base_address[15] = __uint_as_float(0);

		// traversableHandle
		base_address[16] = __uint_as_float(GAS);
		base_address[17] = __uint_as_float(GAS >> 32);

		// pad
		base_address[18] = 0.0f;
		base_address[19] = 0.0f;
	}

	// *********************************************************************************************

	__syncthreads();

	// *********************************************************************************************

	if (wid <= number_of_warps) {
		int lane_id = threadIdx.x & 31;

		float *base_address_1 = &instances[(tid & -32) * 20];
		float *base_address_2 = &tmp[((threadIdx.x & -32) * 20) + ((threadIdx.x & -32) >> 3)];

		base_address_1[lane_id      ] = base_address_2[lane_id      ];
		base_address_1[lane_id + 32 ] = base_address_2[lane_id + 32 ];
		base_address_1[lane_id + 64 ] = base_address_2[lane_id + 64 ];
		base_address_1[lane_id + 96 ] = base_address_2[lane_id + 96 ];
		base_address_1[lane_id + 128] = base_address_2[lane_id + 128];

		base_address_1[lane_id + 160] = base_address_2[lane_id + 160 + 1];
		base_address_1[lane_id + 192] = base_address_2[lane_id + 192 + 1];
		base_address_1[lane_id + 224] = base_address_2[lane_id + 224 + 1];
		base_address_1[lane_id + 256] = base_address_2[lane_id + 256 + 1];
		base_address_1[lane_id + 288] = base_address_2[lane_id + 288 + 1];

		base_address_1[lane_id + 320] = base_address_2[lane_id + 320 + 2];
		base_address_1[lane_id + 352] = base_address_2[lane_id + 352 + 2];
		base_address_1[lane_id + 384] = base_address_2[lane_id + 384 + 2];
		base_address_1[lane_id + 416] = base_address_2[lane_id + 416 + 2];
		base_address_1[lane_id + 448] = base_address_2[lane_id + 448 + 2];

		base_address_1[lane_id + 480] = base_address_2[lane_id + 480 + 3];
		base_address_1[lane_id + 512] = base_address_2[lane_id + 512 + 3];
		base_address_1[lane_id + 544] = base_address_2[lane_id + 544 + 3];
		base_address_1[lane_id + 576] = base_address_2[lane_id + 576 + 3];
		base_address_1[lane_id + 608] = base_address_2[lane_id + 608 + 3];
	}
}

// wrapper - launches kernel
void launchGenerateInstances(
    float3* m, float3* s, float4* q,
    int number_of_Gaussians,
    uint64_t GAS_uint,
    float* instances,
    cudaStream_t stream)
{
    dim3 block(64);
    dim3 grid((number_of_Gaussians + 63) >> 6);
    size_t sharedMem = ((20 * 64) + 7) << 2;

    GenerateInstances<<<grid,block,sharedMem,stream>>>(
        m, s, q,
        number_of_Gaussians,
        GAS_uint,
        instances);

    cudaGetLastError(); // optional
}
}