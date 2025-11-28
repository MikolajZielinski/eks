#include "Header.cuh"

// *** *** *** *** ***

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *** *** *** *** ***

struct SRayPayload {
	int neighbors_num;
	float2 data[HIT_BUFFER_SIZE];
};

// *** *** *** *** ***

extern "C" __global__ void __raygen__() {
	int x = optixGetLaunchIndex().x;
	int number_of_queried_points = optixGetLaunchDimensions().x;

	float3 O = optixLaunchParams.queried_points[x];
	float3 v = make_float3(1.0f, 0.0f, 0.0f);

	// *********************************************************************************************

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *********************************************************************************************

	rp.neighbors_num = 0;

	optixTrace(
		optixLaunchParams.AS,
		O,
		v,
		0.0f,
		2.0f * optixLaunchParams.max_R,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
		0,
		1,
		0,

		rp_addr_lo,
		rp_addr_hi
	);

	// *********************************************************************************************

	for (int i = 0; i < optixLaunchParams.K; ++i) {
		if (i < rp.neighbors_num) {
			float2 tmp = rp.data[i];

			optixLaunchParams.indices[(i * number_of_queried_points) + x] = __float_as_uint(tmp.y);
			optixLaunchParams.distances_squared[(i * number_of_queried_points) + x] = tmp.x;
		} else
			optixLaunchParams.indices[(i * number_of_queried_points) + x] = -1;
	}
}

// *** *** *** *** ***

extern "C" __global__ void __anyhit__() {
	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *********************************************************************************************

	float3 O = optixGetObjectRayOrigin();
	unsigned gauss_ind = optixGetInstanceIndex();
	
	// *********************************************************************************************

	float distance_squared = __fmaf_rn(O.x, O.x, __fmaf_rn(O.y,	O.y, O.z * O.z));
	float t_hit = optixGetRayTmax();

	// *****************************************************************************************

	if (distance_squared <= optixLaunchParams.chi_square_squared_radius) {
		float2 tmp1 = make_float2(distance_squared, __uint_as_float(gauss_ind));
		float2 tmp2;

		for (int i = 0; i < rp->neighbors_num; ++i) {
			tmp2 = rp->data[i];

			if (tmp1.x < tmp2.x) {
				rp->data[i] = tmp1;
				tmp1 = tmp2;
			}
		}

		if (rp->neighbors_num < optixLaunchParams.K) {
			rp->data[rp->neighbors_num++] = tmp1;
			optixIgnoreIntersection();
		} else {
			if (
				__fmaf_rn(
					t_hit,
					t_hit,
					-(optixLaunchParams.four_times_max_R_squared_over_chi_square_squared_radius * tmp2.x)
				) <= 0
			)
				optixIgnoreIntersection();
		}
	} else {
		if (rp->neighbors_num < optixLaunchParams.K)
			optixIgnoreIntersection();
		else {		
			if (
				__fmaf_rn(
					t_hit,
					t_hit,
					-(optixLaunchParams.four_times_max_R_squared_over_chi_square_squared_radius * rp->data[optixLaunchParams.K - 1].x)
				) <= 0
			)
				optixIgnoreIntersection();
		}
	}
}