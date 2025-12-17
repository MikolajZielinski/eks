#include "constants.h"
#include "optix_knn.h"

// *** *** *** *** ***

void CPyOptiXKNN::Fit(float* m, float* s, float* q, int number_of_Gaussians) {
float3 *m_ptr = (float3 *)m;
float3 *s_ptr = (float3 *)s;
float4 *q_ptr = (float4 *)q;

Fit_CUDA(
m_ptr, s_ptr, q_ptr,
number_of_Gaussians
);
}

// *** *** *** *** ***

void CPyOptiXKNN::KNeighbors(float* queried_points, int number_of_points, int K, int* indices, float* distances_squared) {
float3 *queried_points_ptr = (float3 *)queried_points;

KNeighbors_CUDA(
queried_points_ptr,
number_of_points,
K,
indices, distances_squared
);
}

CPyOptiXKNN::~CPyOptiXKNN() {
    // TODO: Add cleanup code here (cudaFree, optixDestroy, etc.)
    // For now, we rely on OS cleanup or implement it later if needed for long-running processes
}

// *** C Interface Implementation ***

extern "C" {
    void* CreateKNN(float chi_square_squared_radius, const char* ptx_path) {
        return new CPyOptiXKNN(chi_square_squared_radius, std::string(ptx_path));
    }

    void DestroyKNN(void* knn_ptr) {
        delete (CPyOptiXKNN*)knn_ptr;
    }

    void Fit(void* knn_ptr, float* m, float* s, float* q, int number_of_Gaussians) {
        ((CPyOptiXKNN*)knn_ptr)->Fit(m, s, q, number_of_Gaussians);
    }

    void KNeighbors(void* knn_ptr, float* queried_points, int number_of_points, int K, int* indices, float* distances_squared) {
        ((CPyOptiXKNN*)knn_ptr)->KNeighbors(queried_points, number_of_points, K, indices, distances_squared);
    }
}
