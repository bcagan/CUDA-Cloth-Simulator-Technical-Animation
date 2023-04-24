#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include <cmath>
#include "defined.h"
#include <chrono>
#include <Vector>
#include <Vector/Vector.hpp>
#include "device_launch_parameters.h"

#include "Particle.hpp"
#include "Force.hpp"
#include "SpringForce.hpp"

//#define VERBOSE

//https://stackoverflow.com/questions/6061565/setting-up-visual-studio-intellisense-for-cuda-kernel-calls
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ SubParticle::SubParticle(float x, float y, float z) {
	m_Position = Vec3(x, y, z);
	m_ConstructPos = m_Position;
};

extern struct Sphere
{
	float radius = 0.f;
	Vec3 center;
};

//Consts
#define RAND (((rand()%2000)/1000.f)-1.f)
#define DAMP 0.98f
#define EPS 0.001f
#define GRA 9.8f
#define FRICTION 0.5f
#define MAX_FORCE_PART 12

//Cloth and spatial grid parameters
const extern int gridDivisions;

//Global parameters
extern float tclock;
extern bool pin;
extern bool sidePin;
extern float springConstCollision;
extern float springConstSphere;
extern float collisionDist;
extern bool windOn;
extern bool doDrawTriangle;
extern bool tearing;
extern bool stepAhead;
extern bool spheresOn;
extern float sphereRadius;
extern bool randHeight;
extern double dist;
extern double ks;
extern double kd;

extern float totalTime;
extern float totalIntegration;
extern float totalParticles;
extern float totalForces ;
extern int numFrames;

//CUDA Data and related helper functions
static SubParticle* devPVec = NULL;
static Sphere* devSVec = NULL;
static std::pair<int, int>* devFVec = NULL;
static std::pair<int, int>* devFOrderVec = NULL;
static signed char* devTypeVec = NULL;
static Vec3* devFAccumalateVec = NULL;
static bool* devBVec = NULL;

void cudaInit(size_t pVecSize, size_t fVecSize, size_t sVecSize) {
	cudaMalloc(&devPVec, pVecSize * sizeof(SubParticle));
	cudaMemset(&devPVec, 0, pVecSize * sizeof(SubParticle));

	cudaMalloc(&devFOrderVec, fVecSize * sizeof(std::pair<int, int>));
	cudaMemset(&devFOrderVec, 0, fVecSize * sizeof(std::pair<int, int>));

	cudaMalloc(&devFVec, fVecSize * sizeof(std::pair<int, int>));
	cudaMemset(&devFVec, 0, fVecSize * sizeof(std::pair<int, int>));

	cudaMalloc(&devSVec, sVecSize * sizeof(Sphere));
	cudaMemset(&devSVec, 0, sVecSize * sizeof(Sphere));

	cudaMalloc(&devBVec, fVecSize * sizeof(bool));
	cudaMemset(&devBVec, 0, fVecSize * sizeof(bool));

	cudaMalloc(&devFAccumalateVec, pVecSize * MAX_FORCE_PART * sizeof(Vec3));
	cudaMemset(&devFAccumalateVec, 0, pVecSize * MAX_FORCE_PART * sizeof(Vec3));

	cudaMalloc(&devTypeVec, fVecSize * sizeof(signed char));
	cudaMemset(&devTypeVec, 0, fVecSize * sizeof(signed char));
}

void cudaLoad(std::vector<SubParticle> pVec, std::vector<std::pair<int, int>> fVec, std::vector<std::pair<int, int>> fOrderVec, std::vector<Sphere> sVec, 
	std::vector<signed char> typeVec, bool* bVec) {

	cudaMemcpy(devPVec, pVec.data(), pVec.size() * sizeof(SubParticle), cudaMemcpyHostToDevice);
	cudaMemcpy(devFOrderVec, fOrderVec.data(), fVec.size() * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);
	cudaMemcpy(devFVec, fVec.data(), fVec.size() * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);
	cudaMemcpy(devSVec, sVec.data(), sVec.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(devBVec, bVec, fVec.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devTypeVec, typeVec.data(), fVec.size() * sizeof(signed char), cudaMemcpyHostToDevice);
}


void devFree() {
	cudaFree(devFVec);
	cudaFree(devSVec);
	cudaFree(devBVec);
	cudaFree(devPVec);
	cudaFree(devFAccumalateVec);
	cudaFree(devFOrderVec);
	cudaFree(devTypeVec);
}



__device__ float GPUWindMagnitude(Vec3 pos, float tclock) {
	float x = pos.x; float y = pos.y; float z = pos.z;
	return 7.f * (cos(tclock * 10.f) + 1.f) * abs(sin(z + tclock * 5) + cos(y + tclock * 5) / 3.f);
};




__device__ float GPUVecNorm(Vec3 v) {
	sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

//Particle and Force functions for GPU


__device__ void clearForce(SubParticle* p)
{
	p->m_ForceAccumulator = Vec3(0.0f, 0.0f, 0.0f);
}


__device__ void apply_force(SubParticle* m_p1, SubParticle* m_p2, Vec3* retForce1, Vec3* retForce2, float m_ks, float m_kd, float m_dist, float tearFactor, bool* teared, bool testTear, bool copy)
{
	Vec3 p1 = m_p1->m_Position;
	Vec3 p2 = m_p2->m_Position;
	Vec3 p1mp2 = p1 - p2; 
	float pdist = vecNorm(p1mp2);
	Vec3 v1mv2 = m_p1->m_Velocity - m_p2->m_Velocity;
	float firstFactorF = m_ks * (pdist - m_dist); 
	Vec3 f1 =  (p1mp2 / pdist) * -1.f* (firstFactorF + m_kd * (dot(v1mv2, p1mp2)) / pdist);
	Vec3 f2 = f1 * -1.f;
	if (testTear && abs(vecNorm(f1)) > m_ks * tearFactor) *teared = true;
	if (copy) {
		*retForce1 += f1;
		*retForce2 += f2;
	}
	else {
		*retForce1 = f1;
		*retForce2 = f2;

	}
}



__device__ SpringForce::SpringForce(SubParticle* p1, SubParticle* p2, float dist, float ks, float kd, int p_ind1, int p_ind2, float tf) :
	m_p1(p1), m_p2(p2), m_dist(dist), m_ks(ks), m_kd(kd),pind1(p_ind1),pind2(p_ind2), tearFactor(tf) {}

//Cuda Algorithms

__global__ void symplecticRoutine(SubParticle* pVector, size_t pLength, float dt, bool sidePin, bool pin, int diameter) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= pLength) return;



	SubParticle* p = &pVector[index];
	//Pin corners
	if (pin) {
		if (index == 0 || index == diameter - 1 || index == (diameter * diameter) - 1 || index == diameter * (diameter - 1)) {
			p->m_ForceAccumulator = Vec3(0.f);
		}
	}
	else if (sidePin) {
		//PpVin a whole side of the cloth
		if (index < diameter) {
			p->m_ForceAccumulator = Vec3(0.f);
		}
	}


	p->m_Velocity = p->m_Velocity*DAMP + p->m_ForceAccumulator * dt*DAMP;
	p->m_Position += p->m_Velocity*dt;
}

__global__ void sumForceRoutine(Vec3* accVector, SubParticle* pVector, size_t pLength) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= pLength) return;
	for (int i = 0; i < MAX_FORCE_PART; i++) {
		pVector[index].m_ForceAccumulator += accVector[MAX_FORCE_PART * index + i];
	}
}

__global__ void forceRoutine(std::pair<int, int>* fVector, std::pair<int, int>* fOrderVector, Vec3* accVector, bool* bVector, size_t start, size_t fLength, bool tearing, 
	SubParticle* pVector, float ks, float kd, double dist, signed char* typeVec) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= start + fLength || index < start) return;
	if (bVector[index] && tearing) return;
	std::pair<int, int> f = fVector[index];
	float cudaTearFactor = 5.f;
	int accInd1 = MAX_FORCE_PART * f.first + (fOrderVector[index]).first;
	int accInd2 = MAX_FORCE_PART * f.second + (fOrderVector[index]).second;
	float distFactored = dist;
	switch (typeVec[index])
	{
	case 0:
		distFactored *= sqrt(2.0);
		break;
	case 1:
		break;
	default:
		distFactored *= 2;
		break;
	}
	apply_force(&(pVector[f.first]), &(pVector[f.second]), &(accVector[accInd1]), &(accVector[accInd2]),ks, kd, distFactored, cudaTearFactor, &(bVector[index]), tearing && true, false);
}



__global__ void particleRoutine(Sphere* sVector, size_t sLength, SubParticle* pVector, 
	size_t pLength, float tclock, bool windOn, float ks, float springConstSphere, int radius, float dt) {


	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= pLength) return;
	SubParticle* p = &pVector[index];


	if (p->m_Position.y < EPS + GRA * dt) { //EPS used to avoid z-fighting
		p->m_Position = Vec3(p->m_Position.x, EPS + GRA * dt, p->m_Position.z);
	}

	//Wind force information
	Vec3 windDir = Vec3(1.3f, 0.f, 0.f);
	clearForce(p);
	float gOffset = -1.0 * GRA;
	p->m_ForceAccumulator = Vec3(0.f, gOffset, 0.f);
	float gpuMag = GPUWindMagnitude(p->m_Position, tclock);
	if (windOn) p->m_ForceAccumulator = 
		p->m_ForceAccumulator + windDir * gpuMag;


	if (p->m_Position.y <= 2*EPS) {
		//Apply friction force
		Vec3 frictionVelVector =Vec3(-p->m_Velocity.x, 0.f, -p->m_Velocity.z);
		Vec3 frictionVector = vecNormalize(frictionVelVector);
		float frictionVel = vecNorm(frictionVelVector);
		p->m_ForceAccumulator = p->m_ForceAccumulator + frictionVelVector * FRICTION;
	}

	//Sphere collisions via spring forces
	for (size_t sInd = 0; sInd < sLength; sInd++) {
		auto s = sVector[sInd];
		if (vecNorm(p->m_Position - s.center) < s.radius) {	

			SubParticle tempSphereParticle(0.f,0.f,0.f);
			apply_force(p,&tempSphereParticle,&(p->m_ForceAccumulator),&(tempSphereParticle.m_ForceAccumulator), 
				ks != 0.f && ks < 50 ? springConstSphere * (5.8f / (sqrt(ks))) : springConstSphere, 
				10.f, (2.f + radius / 40.f) * s.radius, INFINITY,nullptr,false, true); //Distance const scaled by radii to avoid clipping

		}
	}
}

void CPUPrintVec3 (Vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z << std::endl;
}

void GPU_simulate(static std::vector<Sphere> sVector,
	static std::vector<SubParticle>* pVector,
	static std::vector<std::pair<int, int>>* fVector,
	static std::vector<std::pair<int, int>>* fOrderVector,
	static std::vector<signed char> fTypeVector,
	bool** bVector, const int radius, const int diameter, float dt, bool start, bool drawTriangles) {

	auto start_t = std::chrono::high_resolution_clock::now();



	// 2D block for particle calculations
	const int blockSize1d = 128;
	int numBlocksParticles = (pVector->size() + blockSize1d - 1) / blockSize1d;
	int numBlocksForces = (fVector->size() + blockSize1d - 1) / blockSize1d;

	//Since data is retained on GPU, only need to load once
	if (start) {
		cudaLoad(*pVector, *fVector, *fOrderVector, sVector, fTypeVector, *bVector);
		start = false;
	}

	auto particle_start = std::chrono::high_resolution_clock::now();
	//Clear force accumulators for all particles and then apply gravity and then wind and sphere forces
	particleRoutine CUDA_KERNEL(numBlocksParticles,blockSize1d) (devSVec,sVector.size(), devPVec,pVector->size(),tclock,windOn,ks,springConstSphere,radius, dt);
	auto particle_end = std::chrono::high_resolution_clock::now();

	auto f_start = std::chrono::high_resolution_clock::now();
	//Apply spring forces and tearing if need be
	forceRoutine CUDA_KERNEL(numBlocksForces, blockSize1d) (devFVec, devFOrderVec, devFAccumalateVec, devBVec, 0,fVector->size(), tearing, devPVec, ks, kd, dist,devTypeVec);
	sumForceRoutine CUDA_KERNEL(numBlocksParticles, blockSize1d) (devFAccumalateVec, devPVec, pVector->size());
	auto f_end = std::chrono::high_resolution_clock::now();


	auto inter_start = std::chrono::high_resolution_clock::now();
	//To minimize memory usage, only symplectic is supported on CUDA
	symplecticRoutine CUDA_KERNEL(numBlocksParticles, blockSize1d) (devPVec, pVector->size(), dt, sidePin, pin, diameter);
	//Copy integrated data back to CPU from device
	auto inter_end = std::chrono::high_resolution_clock::now();


	//Copy results
	cudaMemcpy(pVector->data(), devPVec, pVector->size() * sizeof(SubParticle), cudaMemcpyDeviceToHost);
	if (!drawTriangles && tearing) cudaMemcpy(*bVector, devBVec, fVector->size() * sizeof(bool), cudaMemcpyDeviceToHost);



	auto end_t = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE


	std::chrono::duration<double>  dif_t = end_t - start_t;
	std::chrono::duration<double>  particle_dif = particle_end - particle_start;
	std::chrono::duration<double>  f_dif = f_end - f_start;
	std::chrono::duration<double>  inter_dif = inter_end - inter_start;
	std::cout << "Time deltas: \n" << "Particles: " << particle_dif.count() <<
		"\n" << "Forces and Tearing: " << f_dif.count() <<  "\n Integration: " << inter_dif.count() << "\nTotal: " << dif_t.count() << std::endl;
	totalTime += dif_t.count();
	totalParticles += particle_dif.count();
	totalForces += f_dif.count();
	totalIntegration += inter_dif.count();
	numFrames++;
#endif //  VERBOSE


	sVector.clear(); sVector.shrink_to_fit();

}
