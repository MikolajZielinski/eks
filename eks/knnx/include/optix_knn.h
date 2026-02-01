#pragma once

// *** *** *** *** ***

#include "common.cuh"
#include <string>

// *** *** *** *** ***

class CPyOptiXKNN {
public:
	CPyOptiXKNN(float chi_square_squared_radius, std::string ptx_path);
    ~CPyOptiXKNN();

	void Fit(float* m, float* s, float* q, int number_of_Gaussians);

	void KNeighbors(float* queried_points, int number_of_points, int K, int* indices, float* distances_squared);

private:
	OptixDeviceContext optixContext;

	OptixModule module;

	OptixProgramGroup missPG;
	OptixProgramGroup raygenPG;
	OptixProgramGroup hitgroupPG;

	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;

	void *missRecordsBuffer;
	void *raygenRecordsBuffer;
	void *hitgroupRecordsBuffer;

	float3 *Gaussian_as_icosahedron_vertices;
	int3 *Gaussian_as_icosahedron_indices;

	OptixTraversableHandle GAS;

	void *GASBuffer;
	
	void *instancesBuffer;

	void *IASBuffer;

	void *launchParamsBuffer;

	float chi_square_squared_radius;

	OptixTraversableHandle IAS;

	float max_s;
	float max_R;

	// *** *** *** *** ***

	void Fit_CUDA(
		float3 *m, float3 *s, float4 *q,
		int number_of_Gaussians
	);

	void KNeighbors_CUDA(
		float3 *queried_points,
		int number_of_points,
		int K,
		int *indices, float *distances_squared
	);
};

extern "C" {
    void* CreateKNN(float chi_square_squared_radius, const char* ptx_path);
    void DestroyKNN(void* knn_ptr);
    void Fit(void* knn_ptr, float* m, float* s, float* q, int number_of_Gaussians);
    void KNeighbors(void* knn_ptr, float* queried_points, int number_of_points, int K, int* indices, float* distances_squared);
}
