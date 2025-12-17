#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "optix_knn.h"

// !!! !!! !!!
#include "optix_function_table_definition.h" // Included only in one file
// !!! !!! !!!

extern "C" void launchGenerateInstances(
    float3* m, float3* s, float4* q,
    int number_of_Gaussians,
    uint64_t GAS_uint,
    float* instances,
    cudaStream_t stream);

// *** *** *** *** ***

CPyOptiXKNN::CPyOptiXKNN(float chi_square_squared_radius, std::string ptx_path) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	// *********************************************************************************************

	error_CUDA = cudaFree(0);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_CUDA = cudaSetDevice(0);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if (error_CUDA_Driver_API != CUDA_SUCCESS) throw 0;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &optixContext);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	moduleCompileOptions.maxRegisterCount = 40;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 0;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	// *********************************************************************************************

	FILE *f = fopen(ptx_path.c_str(), "rb");
	if (!f) {
		fprintf(stderr, "Failed to open PTX file: %s\n", ptx_path.c_str());
		throw std::runtime_error("Failed to open PTX file");
	}
	fseek(f, 0, SEEK_END);
	int shadersSize = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *shaders = (char *)malloc(sizeof(char) * (shadersSize + 1));
	fread(shaders, 1, shadersSize, f);
	fclose(f);
	shaders[shadersSize] = 0;

	// *********************************************************************************************

	error_OptiX = optixModuleCreate(
		optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		shaders,
		strlen(shaders),
		NULL, NULL,
		&module
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	free(shaders);

	// *********************************************************************************************

	OptixStackSizes oss;
	oss.cssRG = 0;
	oss.cssMS = 0;
	oss.cssCH = 0;
	oss.cssAH = 0;
	oss.cssIS = 0;
	oss.cssCC = 0;
	oss.dssDC = 0;

	// *********************************************************************************************

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1, 
		&pgOptions,
		NULL, NULL,
		&missPG
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(missPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module            = module;           
	pgDesc.raygen.entryFunctionName = "__raygen__";

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1,
		&pgOptions,
		NULL, NULL,
		&raygenPG
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(raygenPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleAH            = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__";

	error_OptiX = optixProgramGroupCreate(
		optixContext,
		&pgDesc,
		1, 
		&pgOptions,
		NULL, NULL,
		&hitgroupPG
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixUtilAccumulateStackSizes(hitgroupPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 1;

	OptixProgramGroup program_groups[] = {missPG, raygenPG, hitgroupPG};

	error_OptiX = optixPipelineCreate(
		optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		3,
		NULL, NULL,
		&pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	unsigned int directCallableStackSizeFromTraversal;
	unsigned int directCallableStackSizeFromState;
	unsigned int continuationStackSize;

	error_OptiX = optixUtilComputeStackSizes(
		&oss,
		1,
		0,
		0,
		&directCallableStackSizeFromTraversal,
		&directCallableStackSizeFromState,
		&continuationStackSize 
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_OptiX = optixPipelineSetStackSize(
		pipeline, 
		directCallableStackSizeFromTraversal,
		directCallableStackSizeFromState,
		continuationStackSize,
		2
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	SbtRecord rec;

	// *********************************************************************************************

	sbt = new OptixShaderBindingTable();
	
	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(missPG, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(missRecordsBuffer, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt->missRecordBase = (CUdeviceptr)missRecordsBuffer;
	sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	sbt->missRecordCount = 1;
	
	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(raygenPG, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(raygenRecordsBuffer, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt->raygenRecord = (CUdeviceptr)raygenRecordsBuffer;

	// *********************************************************************************************

	error_OptiX = optixSbtRecordPackHeader(hitgroupPG, &rec);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaMalloc(&hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(hitgroupRecordsBuffer, &rec, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	sbt->hitgroupRecordBase          = (CUdeviceptr)hitgroupRecordsBuffer;
	sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	float3 *Gaussian_as_icosahedron_vertices_host = (float3 *)malloc(sizeof(float3) * 12);
	int3 *Gaussian_as_icosahedron_indices_host = (int3 *)malloc(sizeof(int3) * 20);

	// *********************************************************************************************

	float phi = (1.0f + sqrt(5.0f)) / 2.0f;
	float scale = sqrt(3.0f * chi_square_squared_radius) / (phi * phi); // !!! !!! !!!

	// *********************************************************************************************

	// Vertices
	Gaussian_as_icosahedron_vertices_host[0]  = make_float3(-1.0f * scale,  phi * scale, 0.0f * scale);
	Gaussian_as_icosahedron_vertices_host[1]  = make_float3( 1.0f * scale,  phi * scale, 0.0f * scale);
	Gaussian_as_icosahedron_vertices_host[2]  = make_float3(-1.0f * scale, -phi * scale, 0.0f * scale);
	Gaussian_as_icosahedron_vertices_host[3]  = make_float3( 1.0f * scale, -phi * scale, 0.0f * scale);

	Gaussian_as_icosahedron_vertices_host[4]  = make_float3(0.0f * scale, -1.0f * scale,  phi * scale);
	Gaussian_as_icosahedron_vertices_host[5]  = make_float3(0.0f * scale,  1.0f * scale,  phi * scale);
	Gaussian_as_icosahedron_vertices_host[6]  = make_float3(0.0f * scale, -1.0f * scale, -phi * scale);
	Gaussian_as_icosahedron_vertices_host[7]  = make_float3(0.0f * scale,  1.0f * scale, -phi * scale);

	Gaussian_as_icosahedron_vertices_host[8]  = make_float3( phi * scale, 0.0f * scale, -1.0f * scale);
	Gaussian_as_icosahedron_vertices_host[9]  = make_float3( phi * scale, 0.0f * scale,  1.0f * scale);
	Gaussian_as_icosahedron_vertices_host[10] = make_float3(-phi * scale, 0.0f * scale, -1.0f * scale);
	Gaussian_as_icosahedron_vertices_host[11] = make_float3(-phi * scale, 0.0f * scale,  1.0f * scale);

	// Indices
	Gaussian_as_icosahedron_indices_host[0] = make_int3(0, 11,  5);
	Gaussian_as_icosahedron_indices_host[1] = make_int3(0,  5,  1);
	Gaussian_as_icosahedron_indices_host[2] = make_int3(0,  1,  7);
	Gaussian_as_icosahedron_indices_host[3] = make_int3(0,  7, 10);
	Gaussian_as_icosahedron_indices_host[4] = make_int3(0, 10, 11);

	Gaussian_as_icosahedron_indices_host[5] = make_int3( 1,  5, 9);
	Gaussian_as_icosahedron_indices_host[6] = make_int3( 5, 11, 4);
	Gaussian_as_icosahedron_indices_host[7] = make_int3(11, 10, 2);
	Gaussian_as_icosahedron_indices_host[8] = make_int3(10,  7, 6);
	Gaussian_as_icosahedron_indices_host[9] = make_int3( 7,  1, 8);

	Gaussian_as_icosahedron_indices_host[10] = make_int3(3, 9, 4);
	Gaussian_as_icosahedron_indices_host[11] = make_int3(3, 4, 2);
	Gaussian_as_icosahedron_indices_host[12] = make_int3(3, 2, 6);
	Gaussian_as_icosahedron_indices_host[13] = make_int3(3, 6, 8);
	Gaussian_as_icosahedron_indices_host[14] = make_int3(3, 8, 9);

	Gaussian_as_icosahedron_indices_host[15] = make_int3(4, 9,  5);
	Gaussian_as_icosahedron_indices_host[16] = make_int3(2, 4, 11);
	Gaussian_as_icosahedron_indices_host[17] = make_int3(6, 2, 10);
	Gaussian_as_icosahedron_indices_host[18] = make_int3(8, 6,  7);
	Gaussian_as_icosahedron_indices_host[19] = make_int3(9, 8,  1);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&Gaussian_as_icosahedron_vertices, sizeof(float3) * 12);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(Gaussian_as_icosahedron_vertices, Gaussian_as_icosahedron_vertices_host, sizeof(float3) * 12, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMalloc(&Gaussian_as_icosahedron_indices, sizeof(int3) * 20);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMemcpy(Gaussian_as_icosahedron_indices, Gaussian_as_icosahedron_indices_host, sizeof(int3) * 20, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	free(Gaussian_as_icosahedron_vertices_host);
	free(Gaussian_as_icosahedron_indices_host);

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	// *********************************************************************************************

	OptixBuildInput mesh_input = {};
	mesh_input.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	mesh_input.triangleArray.vertexBuffers    = (CUdeviceptr *)&Gaussian_as_icosahedron_vertices;
	mesh_input.triangleArray.numVertices      = 12;
	mesh_input.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
	mesh_input.triangleArray.indexBuffer      = (CUdeviceptr)Gaussian_as_icosahedron_indices;
	mesh_input.triangleArray.numIndexTriplets = 20;
	mesh_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

	int mesh_input_flags[1]                = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
	mesh_input.triangleArray.flags         = ((const unsigned int *)mesh_input_flags);
	mesh_input.triangleArray.numSbtRecords = 1;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		optixContext,
		&accel_options,
		&mesh_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;

	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	void *outputBuffer;

	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		optixContext,
		0,
		&accel_options,
		&mesh_input,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&GAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaMalloc(&GASBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) throw 0;

	error_OptiX = optixAccelCompact(
		optixContext,
		0,
		GAS,
		(CUdeviceptr)GASBuffer,
		compactedSize,
		&GAS
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	instancesBuffer = NULL; // !!! !!! !!!
	IASBuffer = NULL; // !!! !!! !!!
	
	// *********************************************************************************************

	error_CUDA = cudaMalloc(&launchParamsBuffer, sizeof(SLaunchParams) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	this->chi_square_squared_radius = chi_square_squared_radius;
}

// *** *** *** *** ***

void CPyOptiXKNN::Fit_CUDA(
	float3 *m, float3 *s, float4 *q,
	int number_of_Gaussians
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// *********************************************************************************************

	if (instancesBuffer != NULL) {
		error_CUDA = cudaFree(instancesBuffer);
		if (error_CUDA != cudaSuccess) throw 0;
	}
	error_CUDA = cudaMalloc(&instancesBuffer, sizeof(OptixInstance) * ((number_of_Gaussians + 31) & -32)); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) throw 0;

    launchGenerateInstances(
        m, s, q,
        number_of_Gaussians,
        GAS,
        (float*)instancesBuffer,
        0   // or stream
    );
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	// *********************************************************************************************

	OptixBuildInput instances_input = {};
	instances_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instances_input.instanceArray.instances    = (CUdeviceptr)instancesBuffer;
	instances_input.instanceArray.numInstances = number_of_Gaussians;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		optixContext,
		&accel_options,
		&instances_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	// *********************************************************************************************

	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) throw 0;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;

	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	void *outputBuffer;

	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		optixContext,
		0,
		&accel_options,
		&instances_input,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&IAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) throw 0;

	if (IASBuffer != NULL) {
		error_CUDA = cudaFree(IASBuffer);
		if (error_CUDA != cudaSuccess) throw 0;
	}
	error_CUDA = cudaMalloc(&IASBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) throw 0;

	error_OptiX = optixAccelCompact(
		optixContext,
		0,
		IAS,
		(CUdeviceptr)IASBuffer,
		compactedSize,
		&IAS
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	try {
		max_s = *thrust::max_element(
			thrust::device_pointer_cast((float *)s),
			thrust::device_pointer_cast((float *)s) + (3 * number_of_Gaussians)
		);
	} catch (...) {
		throw 0;
	}

	max_R = sqrtf(chi_square_squared_radius) * max_s;
}

// *** *** *** *** ***

void CPyOptiXKNN::KNeighbors_CUDA(
	float3 *queried_points,
	int number_of_points,
	int K,
	int *indices, float *distances_squared
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// *********************************************************************************************

	SLaunchParams launchParams;

	launchParams.queried_points = queried_points;
	launchParams.max_s = max_s;
	launchParams.max_R = max_R;
	launchParams.AS = IAS;
	launchParams.indices = indices;
	launchParams.distances_squared = distances_squared;
	launchParams.chi_square_squared_radius = chi_square_squared_radius;
	launchParams.K = K;

	error_CUDA = cudaMemcpy(launchParamsBuffer, &launchParams, sizeof(SLaunchParams) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) throw 0;

	// *********************************************************************************************

	error_OptiX = optixLaunch(
		pipeline,
		0,
		(CUdeviceptr)launchParamsBuffer,
		sizeof(SLaunchParams) * 1,
		sbt,
		number_of_points,
		1,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) throw 0;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) throw 0;
}