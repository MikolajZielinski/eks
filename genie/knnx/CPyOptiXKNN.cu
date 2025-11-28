#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "CPyOptiXKNN.h"
#include "Header.cuh"

// !!! !!! !!!
#include "optix_function_table_definition.h" // Included only in one file
// !!! !!! !!!

// *** *** *** *** ***

CPyOptiXKNN::CPyOptiXKNN(float chi_square_squared_radius) {
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

	FILE *f = fopen("/home/mikolaj/Git/genie_3d/genie/knnx/build/shaders.ptx", "rb");
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

__global__ void GenerateInstances(
	float3 *m, float3 *s, float4 *q,
	int number_of_Gaussians,
	OptixTraversableHandle GAS,
	float *instances
) {
	extern __shared__ float tmp[];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int wid = tid >> 5;
	int number_of_warps = number_of_Gaussians >> 5;

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
		base_address[15] = __uint_as_float(OPTIX_INSTANCE_FLAG_NONE);

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

	GenerateInstances<<<(number_of_Gaussians + 63) >> 6, 64, ((20 * 64) + 7) << 2>>>(
		m, s, q,
		number_of_Gaussians,
		GAS,
		(float *)instancesBuffer
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
		max_R = sqrtf(chi_square_squared_radius) * (*thrust::max_element(
			thrust::device_pointer_cast((float *)s),
			thrust::device_pointer_cast((float *)s) + (3 * number_of_Gaussians)
		));
	} catch (...) {
		throw 0;
	}
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
	launchParams.max_R = max_R;
	launchParams.four_times_max_R_squared_over_chi_square_squared_radius = (4.0f * max_R * max_R) / chi_square_squared_radius;
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