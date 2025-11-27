#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "optix.h"
#include "optix_stubs.h"

#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/reduce.h>

#include <stdio.h>

// *** *** *** *** ***

struct S_CUDA_KNN {
	OptixDeviceContext optixContext;
	OptixModule module;
	OptixProgramGroup raygenPG;
	OptixProgramGroup missPG;
	OptixProgramGroup hitgroupPG;
	OptixPipeline pipeline;
	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;
	float chi_square_squared_radius; // !!! !!! !!!
	float3 *gaussian_as_polygon_vertices;
	int3 *gaussian_as_polygon_indices;
	OptixTraversableHandle GAS;
	void *GASBuffer;
	float4 *means; // !!! !!! !!!
	int number_of_means; // !!! !!! !!!
	void *instancesBuffer;
	OptixTraversableHandle IAS;
	void *IASBuffer;
};

// *** *** *** *** ***

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *** *** *** *** ***

struct SLaunchParams {
	OptixTraversableHandle traversable;
	float4 *means;
	float4 *queried_points;
	float *distances;
	int *indices;
	float chi_square_squared_radius;
	int K;
	float max_R;
};

// *** *** *** *** ***

extern "C" bool CUDA_KNN_Init(float chi_square_squared_radius, S_CUDA_KNN* knn);
extern "C" bool CUDA_KNN_Fit(float4 *means, int number_of_means, S_CUDA_KNN* knn);
extern "C" bool CUDA_KNN_KNeighbors(
	float4 *queried_points,
	int number_of_queried_points,
	int K,
	float *distances,
	int *indices,
	S_CUDA_KNN* knn
);
bool CUDA_KNN_Destroy(S_CUDA_KNN* cknn);