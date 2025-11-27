#include "Header.cuh"

// !!! !!! !!!
#include "optix_function_table_definition.h"
// !!! !!! !!!

// *************************************************************************************************
// CUDA_KNN_Init                                                                                   *
// *************************************************************************************************

extern "C" bool CUDA_KNN_Init(float chi_square_squared_radius, S_CUDA_KNN* knn) {

	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	S_CUDA_KNN cknn = *knn;

	error_CUDA = cudaSetDevice(0);
	if (error_CUDA != cudaSuccess) printf("An error occurred... .");

	// *********************************************************************************************

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) return false;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if (error_CUDA_Driver_API != CUDA_SUCCESS) return false;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &cknn.optixContext);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	const char *ptxPath = "genie/knn/build/shaders.ptx";

	/* Determine PTX file size first */
	FILE *f = fopen(ptxPath, "rb");
	if (!f) return false;
	fseek(f, 0, SEEK_END);
	int ptxCodeSize = ftell(f);
	fclose(f);

	/* Allocate buffers: ptxCode will hold the concatenated file contents, buffer is a temporary read buffer */
	char *ptxCode = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	char *buffer = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	if (!ptxCode || !buffer) {
		if (ptxCode) free(ptxCode);
		if (buffer) free(buffer);
		return false;
	}

	/* Initialize ptxCode as empty C-string so strcat works */
	ptxCode[0] = '\0';

	/* Re-open and read the file piecewise into ptxCode using fgets/strcat to emulate the requested behavior */
	f = fopen(ptxPath, "rb");
	if (!f) {
		free(ptxCode);
		free(buffer);
		return false;
	}

	while (fgets(buffer, ptxCodeSize + 1, f) != NULL) {
		strcat(ptxCode, buffer);
	}

	fclose(f);
	free(buffer);

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

	error_OptiX = optixModuleCreate(
		cknn.optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode,
		strlen(ptxCode),
		NULL, NULL,
		&cknn.module
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	free(ptxCode);

	// *********************************************************************************************

	OptixProgramGroupOptions pgOptions = {};

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_raygen = {};
	pgDesc_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc_raygen.raygen.module = cknn.module;           
	pgDesc_raygen.raygen.entryFunctionName = "__raygen__";

	error_OptiX = optixProgramGroupCreate(
		cknn.optixContext,
		&pgDesc_raygen,
		1,
		&pgOptions,
		NULL, NULL,
		&cknn.raygenPG
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_miss = {};
	pgDesc_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	error_OptiX = optixProgramGroupCreate(
		cknn.optixContext,
		&pgDesc_miss,
		1, 
		&pgOptions,
		NULL, NULL,
		&cknn.missPG
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_hitgroup = {};
	pgDesc_hitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	// !!! !!! !!! TRIANGLES !!! !!! !!!
	pgDesc_hitgroup.hitgroup.moduleCH = nullptr;
	pgDesc_hitgroup.hitgroup.entryFunctionNameCH = nullptr;
	pgDesc_hitgroup.hitgroup.moduleAH = cknn.module;
	pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__";
	pgDesc_hitgroup.hitgroup.moduleIS = nullptr;
	pgDesc_hitgroup.hitgroup.entryFunctionNameIS = nullptr;

	error_OptiX = optixProgramGroupCreate(
		cknn.optixContext,
		&pgDesc_hitgroup,
		1, 
		&pgOptions,
		NULL, NULL,
		&cknn.hitgroupPG
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 1;

	OptixProgramGroup program_groups[] = { cknn.raygenPG, cknn.missPG, cknn.hitgroupPG };

	error_OptiX = optixPipelineCreate(
		cknn.optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		3,
		NULL, NULL,
		&cknn.pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixPipelineSetStackSize(
		cknn.pipeline, 
		0,
		0,
		2 * 1024 * 8, // !!! !!! !!! SOME NASTY CONSTANT !!! !!! !!!
		2
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	cknn.sbt = new OptixShaderBindingTable();

	// *********************************************************************************************

	SbtRecord rec_raygen;
	error_OptiX = optixSbtRecordPackHeader(cknn.raygenPG, &rec_raygen);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&cknn.raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.raygenRecordsBuffer, &rec_raygen, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.sbt->raygenRecord = (CUdeviceptr)cknn.raygenRecordsBuffer;

	// *********************************************************************************************

	SbtRecord rec_miss;
	error_OptiX = optixSbtRecordPackHeader(cknn.missPG, &rec_miss);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&cknn.missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.missRecordsBuffer, &rec_miss, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.sbt->missRecordBase = (CUdeviceptr)cknn.missRecordsBuffer;
	cknn.sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	cknn.sbt->missRecordCount = 1;

	// *********************************************************************************************

	SbtRecord rec_hitgroup;
	error_OptiX = optixSbtRecordPackHeader(cknn.hitgroupPG, &rec_hitgroup);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaMalloc(&cknn.hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.hitgroupRecordsBuffer, &rec_hitgroup, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	cknn.sbt->hitgroupRecordBase          = (CUdeviceptr)cknn.hitgroupRecordsBuffer;
	cknn.sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	cknn.sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	// !!! !!! !!!
	cknn.chi_square_squared_radius = chi_square_squared_radius;
	// !!! !!! !!!

	// *********************************************************************************************

	float3 *gaussian_as_polygon_vertices = (float3 *)malloc(sizeof(float3) * 12);
	int3 *gaussian_as_polygon_indices = (int3 *)malloc(sizeof(int3) * 20);

	float phi = (1.0f + sqrt(5.0f)) / 2.0f;
	float scale = sqrt(3.0f * chi_square_squared_radius) / (phi * phi); // !!! !!! !!!
	
	// Vertices
	gaussian_as_polygon_vertices[0] = make_float3(-1.0f * scale, phi * scale, 0.0f * scale);
	gaussian_as_polygon_vertices[1] = make_float3(1.0f * scale, phi * scale, 0.0f * scale);
	gaussian_as_polygon_vertices[2] = make_float3(-1.0f * scale, -phi * scale, 0.0f * scale);
	gaussian_as_polygon_vertices[3] = make_float3(1.0f * scale, -phi * scale, 0.0f * scale);

	gaussian_as_polygon_vertices[4] = make_float3(0.0f * scale, -1.0f * scale, phi * scale);
	gaussian_as_polygon_vertices[5] = make_float3(0.0f * scale, 1.0f * scale, phi * scale);
	gaussian_as_polygon_vertices[6] = make_float3(0.0f * scale, -1.0f * scale, -phi * scale);
	gaussian_as_polygon_vertices[7] = make_float3(0.0f * scale, 1.0f * scale, -phi * scale);

	gaussian_as_polygon_vertices[8] = make_float3(phi * scale, 0.0f * scale, -1.0f * scale);
	gaussian_as_polygon_vertices[9] = make_float3(phi * scale, 0.0f * scale, 1.0f * scale);
	gaussian_as_polygon_vertices[10] = make_float3(-phi * scale, 0.0f * scale, -1.0f * scale);
	gaussian_as_polygon_vertices[11] = make_float3(-phi * scale, 0.0f * scale, 1.0f * scale);

	// Indices
	gaussian_as_polygon_indices[0] = make_int3(0, 11, 5);
	gaussian_as_polygon_indices[1] = make_int3(0, 5, 1);
	gaussian_as_polygon_indices[2] = make_int3(0, 1, 7);
	gaussian_as_polygon_indices[3] = make_int3(0, 7, 10);
	gaussian_as_polygon_indices[4] = make_int3(0, 10, 11);

	gaussian_as_polygon_indices[5] = make_int3(1, 5, 9);
	gaussian_as_polygon_indices[6] = make_int3(5, 11, 4);
	gaussian_as_polygon_indices[7] = make_int3(11, 10, 2);
	gaussian_as_polygon_indices[8] = make_int3(10, 7, 6);
	gaussian_as_polygon_indices[9] = make_int3(7, 1, 8);

	gaussian_as_polygon_indices[10] = make_int3(3, 9, 4);
	gaussian_as_polygon_indices[11] = make_int3(3, 4, 2);
	gaussian_as_polygon_indices[12] = make_int3(3, 2, 6);
	gaussian_as_polygon_indices[13] = make_int3(3, 6, 8);
	gaussian_as_polygon_indices[14] = make_int3(3, 8, 9);

	gaussian_as_polygon_indices[15] = make_int3(4, 9, 5);
	gaussian_as_polygon_indices[16] = make_int3(2, 4, 11);
	gaussian_as_polygon_indices[17] = make_int3(6, 2, 10);
	gaussian_as_polygon_indices[18] = make_int3(8, 6, 7);
	gaussian_as_polygon_indices[19] = make_int3(9, 8, 1);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&cknn.gaussian_as_polygon_vertices, sizeof(float3) * 12);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.gaussian_as_polygon_vertices, gaussian_as_polygon_vertices, sizeof(float3) * 12, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&cknn.gaussian_as_polygon_indices, sizeof(int3) * 20);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(cknn.gaussian_as_polygon_indices, gaussian_as_polygon_indices, sizeof(int3) * 20, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	free(gaussian_as_polygon_vertices);
	free(gaussian_as_polygon_indices);

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput input_tri = {};
	input_tri.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	input_tri.triangleArray.vertexBuffers = (CUdeviceptr *)&cknn.gaussian_as_polygon_vertices;
	input_tri.triangleArray.numVertices = 12;
	input_tri.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	input_tri.triangleArray.indexBuffer = (CUdeviceptr)cknn.gaussian_as_polygon_indices;
	input_tri.triangleArray.numIndexTriplets = 20;
	input_tri.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

	int input_tri_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
	input_tri.triangleArray.flags = (const unsigned int *)input_tri_flags;
	input_tri.triangleArray.numSbtRecords = 1;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		cknn.optixContext,
		&accel_options,
		&input_tri,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************

	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) return false;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;

	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) return false;

	void *outputBuffer;

	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		cknn.optixContext,
		0,
		&accel_options,
		&input_tri,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&cknn.GAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) return false;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMalloc(&cknn.GASBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixAccelCompact(
		cknn.optixContext,
		0,
		cknn.GAS,
		(CUdeviceptr)cknn.GASBuffer,
		compactedSize,
		&cknn.GAS
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	cknn.IASBuffer = NULL; // !!! !!! !!!
	*knn = cknn;

	// *********************************************************************************************

	return true;
}

// *************************************************************************************************
// CUDA_KNN_Fit                                                                                    *
// *************************************************************************************************

__global__ void GenerateInstances(float4 *means, int number_of_means, OptixTraversableHandle GAS, float *instances) {
	extern __shared__ float tmp[];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int wid = tid >> 5;
	int number_of_warps = number_of_means >> 5;

	// *** *** *** *** ***

	if (wid <= number_of_warps) {
		int index = ((tid < number_of_means) ? tid : (number_of_means - 1));
		float4 mean = means[index];

		float *base_address = &tmp[(threadIdx.x * 20) + (threadIdx.x >> 3)];

		// transform
		base_address[0] = mean.w;
		base_address[1] = 0.0f;
		base_address[2] = 0.0f;
		base_address[3] = mean.x;

		base_address[4] = 0.0f;
		base_address[5] = mean.w;
		base_address[6] = 0.0f;
		base_address[7] = mean.y;

		base_address[8] = 0.0f;
		base_address[9] = 0.0f;
		base_address[10] = mean.w;
		base_address[11] = mean.z;

		// instanceId
		base_address[12] = 0.0f;

		// sbtOffset
		base_address[13] = 0.0f;

		// visibilityMask
		base_address[14] = __uint_as_float(255);

		// flags
		base_address[15] = __uint_as_float(OPTIX_INSTANCE_FLAG_NONE);

		// traversableHandle
		base_address[16] = __uint_as_float(GAS);
		base_address[17] = __uint_as_float(GAS >> 32);

		// pad
		base_address[18] = 0.0f;
		base_address[19] = 0.0f;
	}

	// *** *** *** *** ***

	__syncthreads();

	// *** *** *** *** ***

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

// *************************************************************************************************

/*
mean[i].x = X coordinate of the mean
mean[i].y = Y coordinate of the mean
mean[i].z = Z coordinate of the mean
mean[i].w = Scale of the Gaussian. Note that since the covariance matrix is radial, only one scale
            parameter is needed and there's no point using the quaternions parameters to describe
			the rotation of the Gaussian.
*/
extern "C" bool CUDA_KNN_Fit(float4 *means, int number_of_means, S_CUDA_KNN* knn) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	S_CUDA_KNN cknn = *knn;

	// *********************************************************************************************

	// !!! !!! !!!
	cknn.means = means;
	cknn.number_of_means = number_of_means;
	// !!! !!! !!!

	// *********************************************************************************************

	if (cknn.instancesBuffer != NULL) {
		cudaFree(cknn.instancesBuffer);
		cknn.instancesBuffer = NULL;
	}

	cudaMalloc(&cknn.instancesBuffer, sizeof(OptixInstance) * ((number_of_means + 31) & -32)); // !!! !!! !!!
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;
	
	GenerateInstances<<<(number_of_means + 63) >> 6, 64, ((20 * 64) + 7) << 2>>>(
		means,
		number_of_means,
		cknn.GAS,
		(float *)cknn.instancesBuffer
	);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) return false;
	
	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput input_ins = {};
	input_ins.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	input_ins.instanceArray.instances    = (CUdeviceptr)cknn.instancesBuffer;
	input_ins.instanceArray.numInstances = number_of_means;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		cknn.optixContext,
		&accel_options,
		&input_ins,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	// *********************************************************************************************
	
	unsigned long long *compactedSizeBuffer;
	error_CUDA = cudaMalloc(&compactedSizeBuffer, sizeof(unsigned long long) * 1);
	if (error_CUDA != cudaSuccess) return false;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)compactedSizeBuffer;

	void *tempBuffer;
	
	error_CUDA = cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes);
	if (error_CUDA != cudaSuccess) return false;

	void *outputBuffer;
	
	error_CUDA = cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		cknn.optixContext,
		0,
		&accel_options,
		&input_ins,
		1,  
		(CUdeviceptr)tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&cknn.IAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) return false;

	unsigned long long compactedSize;

	error_CUDA = cudaMemcpy(&compactedSize, compactedSizeBuffer, sizeof(unsigned long long) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) return false;

	if (cknn.IASBuffer != NULL) {
		error_CUDA = cudaFree(cknn.IASBuffer);
		if (error_CUDA != cudaSuccess) return false;
	}

	error_CUDA = cudaMalloc(&cknn.IASBuffer, compactedSize);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixAccelCompact(
		cknn.optixContext,
		0,
		cknn.IAS,
		(CUdeviceptr)cknn.IASBuffer,
		compactedSize,
		&cknn.IAS
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(compactedSizeBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(tempBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(outputBuffer);
	if (error_CUDA != cudaSuccess) return false;

	// *********************************************************************************************

	*knn = cknn;

	return true;
}

// *************************************************************************************************
// CUDA_KNN_KNeighbors                                                                             *
// *************************************************************************************************

struct SReductionOperator_float4 {
	__device__ float4 operator()(const float4 &a, const float4 &b) const {
		return make_float4(
			0.0f,
			0.0f,
			0.0f,
			(a.w <= b.w) ? b.w : a.w
		);
	}
};

// *************************************************************************************************

extern "C" bool CUDA_KNN_KNeighbors(
	float4 *queried_points,
	int number_of_queried_points,
	int K,
	float *distances,
	int *indices,
	S_CUDA_KNN* knn
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// *********************************************************************************************
	
	float4 max_R;

	S_CUDA_KNN cknn = *knn;

	try {
		max_R = thrust::reduce(
			thrust::device_pointer_cast(cknn.means),
			thrust::device_pointer_cast(cknn.means) + cknn.number_of_means,
			make_float4(0.0f, 0.0f, 0.0f, -INFINITY),
			SReductionOperator_float4()
		);
	} catch (...) {
		return false;
	}

	// *********************************************************************************************

	SLaunchParams launchParams;

	launchParams.traversable = cknn.IAS;
	launchParams.means = cknn.means;
	launchParams.queried_points = queried_points;
	launchParams.distances = distances;
	launchParams.indices = indices;
	launchParams.chi_square_squared_radius = cknn.chi_square_squared_radius;
	launchParams.K = K;
	launchParams.max_R = max_R.w * sqrtf(cknn.chi_square_squared_radius);

	void *launchParamsBuffer;

	error_CUDA = cudaMalloc(&launchParamsBuffer, sizeof(SLaunchParams) * 1);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaMemcpy(launchParamsBuffer, &launchParams, sizeof(SLaunchParams) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixLaunch(
		cknn.pipeline,
		0,
		(CUdeviceptr)launchParamsBuffer,
		sizeof(SLaunchParams) * 1,
		cknn.sbt,
		number_of_queried_points,
		1,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(launchParamsBuffer);
	if (error_CUDA != cudaSuccess) return false;

	*knn = cknn;

	return true;
}

// *************************************************************************************************
// CUDA_KNN_Destroy                                                                                *
// *************************************************************************************************

bool CUDA_KNN_Destroy(S_CUDA_KNN* knn) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	S_CUDA_KNN cknn = *knn;
	
	delete cknn.sbt;

	error_CUDA = cudaFree(cknn.raygenRecordsBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.missRecordsBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.hitgroupRecordsBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.gaussian_as_polygon_vertices);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.gaussian_as_polygon_indices);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.GASBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.instancesBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(cknn.IASBuffer);
	if (error_CUDA != cudaSuccess) return false;

	error_OptiX = optixPipelineDestroy(cknn.pipeline);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixProgramGroupDestroy(cknn.raygenPG);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixProgramGroupDestroy(cknn.missPG);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixProgramGroupDestroy(cknn.hitgroupPG);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixModuleDestroy(cknn.module);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	error_OptiX = optixDeviceContextDestroy(cknn.optixContext);
	if (error_OptiX != OPTIX_SUCCESS) return false;

	*knn = cknn;

	return true;
}