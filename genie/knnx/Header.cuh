#pragma once

#include "constants.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "optix.h"
#include "optix_host.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"

// *** *** *** *** ***

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *** *** *** *** ***

struct SLaunchParams {
	float3 *queried_points;
	float max_R;
	float four_times_max_R_squared_over_chi_square_squared_radius;
	OptixTraversableHandle AS;
	int *indices;
	float *distances_squared;
	float chi_square_squared_radius;
	int K;
};
