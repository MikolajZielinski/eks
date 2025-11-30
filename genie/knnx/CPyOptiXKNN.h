#pragma once

// *** *** *** *** ***

#include "Header.cuh"

#ifndef __CUDACC__
	#include <torch/extension.h>
	#include <utility>
#endif

// *** *** *** *** ***

class CPyOptiXKNN {
public:
	CPyOptiXKNN(float chi_square_squared_radius);

	#ifndef __CUDACC__
		void Fit(torch::Tensor &m, torch::Tensor &s, torch::Tensor &q);

		std::tuple<torch::Tensor, torch::Tensor> KNeighbors(torch::Tensor &queried_points, int K);
	#endif

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
