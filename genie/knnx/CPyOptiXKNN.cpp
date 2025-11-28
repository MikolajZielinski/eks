#include "constants.h"
#include "CPyOptiXKNN.h"

// *** *** *** *** ***

void CPyOptiXKNN::Fit(torch::Tensor &m, torch::Tensor &s, torch::Tensor &q) {
	float3 *m_ptr = (float3 *)m.data_ptr();
	float3 *s_ptr = (float3 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();

	c10::IntArrayRef sizes = m.sizes();

	int number_of_Gaussians = sizes[0];

	Fit_CUDA(
		m_ptr, s_ptr, q_ptr,
		number_of_Gaussians
	);
}

// *** *** *** *** ***

std::tuple<torch::Tensor, torch::Tensor> CPyOptiXKNN::KNeighbors(torch::Tensor &queried_points, int K) {
	c10::IntArrayRef sizes = queried_points.sizes();

	int number_of_points = sizes[0];

	const int64_t size[] = {K, number_of_points};
	torch::TensorOptions options;

	options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	torch::Tensor indices = torch::empty(size, options);

	options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor distances_squared = torch::empty(size, options);

	float3 *queried_points_ptr = (float3 *)queried_points.data_ptr();

	int *indices_ptr = (int *)indices.data_ptr();
	float *distances_squared_ptr = (float *)distances_squared.data_ptr();

	KNeighbors_CUDA(
		queried_points_ptr,
		number_of_points,
		K,
		indices_ptr, distances_squared_ptr
	);

	return std::make_tuple(indices, distances_squared);
}