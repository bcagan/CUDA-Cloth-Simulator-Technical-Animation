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
extern double ks;
extern double kd;
extern bool tearing;
extern bool stepAhead;
extern bool spheresOn;
extern float sphereRadius;
extern bool randHeight;
extern double dist;



//CUDA Data and related helper functions
static SubParticle* devPVec = NULL;
static Sphere* devSVec = NULL;
static std::pair<int, int>* devFVec = NULL;
static bool* devBVec = NULL;

void clothInit(std::vector<SubParticle> pVec, std::vector<std::pair<int,int>> fVec, std::vector<Sphere> sVec, bool* bVec) {
	cudaMalloc(&devPVec, pVec.size() * sizeof(SubParticle));
	cudaMemset(&devPVec, 0, pVec.size() * sizeof(SubParticle));
	cudaMemcpy(devPVec, pVec.data(), pVec.size() * sizeof(SubParticle), cudaMemcpyHostToDevice);

	cudaMalloc(&devFVec, fVec.size() * sizeof(std::pair<int, int>));
	cudaMemset(&devFVec, 0, fVec.size() * sizeof(std::pair<int, int>));
	cudaMemcpy(devFVec, fVec.data(), fVec.size() * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);

	cudaMalloc(&devSVec, sVec.size() * sizeof(Sphere));
	cudaMemset(&devSVec, 0, sVec.size() * sizeof(Sphere));
	cudaMemcpy(devSVec, sVec.data(), sVec.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

	cudaMalloc(&devBVec, fVec.size() * sizeof(bool));
	cudaMemset(&devBVec, 0, fVec.size() * sizeof(bool));
	cudaMemcpy(&devBVec, bVec, fVec.size() * sizeof(bool), cudaMemcpyHostToDevice);
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


//Something is incorrect here

__device__ void apply_force(SubParticle* m_p1, SubParticle* m_p2, float m_ks, float m_kd, float m_dist, float tearFactor, bool* teared, bool testTear)
{
	Vec3 p1 = m_p1->m_Position;
	Vec3 p2 = m_p2->m_Position;
	Vec3 p1mp2 = p1 - p2; 
	float pdist = vecNorm(p1mp2);
	Vec3 v1mv2 = m_p1->m_Velocity - m_p2->m_Velocity;
	float firstFactorF = m_ks * (pdist - m_dist); 
	Vec3 f1 = Vec3(0.f);
	if (pdist != 0) {
		f1 = Vec3(-1.f) * (Vec3(firstFactorF) + Vec3(m_kd * (dot(v1mv2, p1mp2)) / pdist)) * (p1mp2 / pdist);
	}
	Vec3 f2 = Vec3(-1.f) * f1;
	if (testTear && abs(vecNorm(f1)) > m_ks * tearFactor) *teared = true;
	m_p1->m_ForceAccumulator += f1;
	m_p2->m_ForceAccumulator += f2;
}


__device__ bool willTear(SpringForce* f) {
	return f->teared;
}


__device__ SpringForce::SpringForce(SubParticle* p1, SubParticle* p2, float dist, float ks, float kd, int p_ind1, int p_ind2, float tf) :
	m_p1(p1), m_p2(p2), m_dist(dist), m_ks(ks), m_kd(kd),pind1(p_ind1),pind2(p_ind2), tearFactor(tf) {}

//Cuda Algorithms



__global__ void forceRoutine(std::pair<int, int>* fVector, bool* bVector, size_t start, size_t fLength, bool tearing, SubParticle* pVector, float ks, float kd, double dist) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= start + fLength || index < start) return;
	if (bVector[index]) return;
	std::pair<int, int> f = fVector[index];
	float cudaTearFactor = 4.f;
	apply_force(&(pVector[f.first]), &(pVector[f.second]), ks, kd, dist, cudaTearFactor, &(bVector[index]),true);
	if (!tearing) {
		bVector[index] = false;
	}
}



__global__ void particleRoutine(Sphere* sVector, size_t sLength, SubParticle* pVector, 
	size_t pLength, float tclock, bool windOn, float ks, float springConstSphere, int radius) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= pLength) return;
	SubParticle* p = &pVector[index];

	//Wind force information
	Vec3 windDir = Vec3(1.f, 0.f, 0.f);
	clearForce(p);
	float gOffset = -1.0 * GRA;
	p->m_ForceAccumulator = 
		p->m_ForceAccumulator + Vec3(0.f, gOffset, 0.f);
	float gpuMag = GPUWindMagnitude(
		Vec3(p->m_Position.x * windDir.x, p->m_Position.y * windDir.y, p->m_Position.x * windDir.z), tclock);
	if (windOn) p->m_ForceAccumulator = 
		p->m_ForceAccumulator + Vec3(gpuMag, gpuMag, gpuMag);


	if (p->m_Position.y <= EPS) {
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
			apply_force(p,&tempSphereParticle, ks != 0.f && ks < 50 ? springConstSphere * (5.8f / (sqrt(ks))) : springConstSphere, 
				10.f, (2.f + radius / 40.f) * s.radius, INFINITY,nullptr,false); //Distance const scaled by radii to avoid clipping

		}
	}
}

void CPUPrintVec3 (Vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z << std::endl;
}

void GPU_simulate(static std::vector<Sphere> sVector,
	static std::vector<SubParticle>* pVector,
	static std::vector<std::pair<int,int>>* fVector,
	bool** bVector, const int radius, const int diameter) {

	auto start_t = std::chrono::high_resolution_clock::now();


	auto pvectime = std::chrono::high_resolution_clock::now();

	// 2D block for particle calculations
	const int blockSize1d = 128;
	int numBlocksParticles = (pVector->size() + blockSize1d - 1) / blockSize1d;
	int numBlocksForces = (fVector->size() + blockSize1d - 1) / blockSize1d;

	auto overhead1_start = std::chrono::high_resolution_clock::now();

	clothInit(*pVector, *fVector, sVector,*bVector);

	auto overhead1_end = std::chrono::high_resolution_clock::now();


	for (auto& p : *pVector) {
		CPUPrintVec3(p.m_ForceAccumulator);
	}
	std::cout << std::endl << std::endl << std::endl;

	auto particle_start = std::chrono::high_resolution_clock::now();
	//Clear force accumulators for all particles and then apply gravity and then wind and sphere forces
	particleRoutine CUDA_KERNEL(numBlocksParticles,blockSize1d) (devSVec,sVector.size(), devPVec,pVector->size(),tclock,windOn,ks,springConstSphere,radius);
	auto particle_end = std::chrono::high_resolution_clock::now();



	auto f_start = std::chrono::high_resolution_clock::now();
	//Apply spring forces and tearing if need be
	for (int i = 0; i < fVector->size(); i++) {
		forceRoutine CUDA_KERNEL(numBlocksForces, blockSize1d) (devFVec, devBVec, i, 1, tearing, devPVec, ks, kd, dist);
	}
	//Copy results
	cudaMemcpy(pVector->data(), devPVec, pVector->size() * sizeof(SubParticle), cudaMemcpyDeviceToHost);
	cudaMemcpy(*bVector, devBVec, fVector->size() * sizeof(bool), cudaMemcpyDeviceToHost);
	auto f_end = std::chrono::high_resolution_clock::now();


	for (auto& p : *pVector) {
		CPUPrintVec3(p.m_ForceAccumulator);
	}
	std::cout << std::endl << std::endl << std::endl;

	auto overhead2_start = std::chrono::high_resolution_clock::now();

	cudaFree(devFVec);
	cudaFree(devPVec);
	cudaFree(devSVec);
	cudaFree(devBVec);


	auto overhead2_end = std::chrono::high_resolution_clock::now();


	//Pin corners
	if (pin) {
		(*pVector)[0].m_ForceAccumulator.x = 0.f;
		(*pVector)[0].m_ForceAccumulator.y = 0.f;
		(*pVector)[0].m_ForceAccumulator.z = 0.f;
		(*pVector)[diameter - 1].m_ForceAccumulator.x = 0.f;
		(*pVector)[diameter - 1].m_ForceAccumulator.y = 0.f;
		(*pVector)[diameter - 1].m_ForceAccumulator.z = 0.f;
		(*pVector)[(diameter * diameter) - 1].m_ForceAccumulator.x = 0.f;
		(*pVector)[(diameter * diameter) - 1].m_ForceAccumulator.y = 0.f;
		(*pVector)[(diameter * diameter) - 1].m_ForceAccumulator.z = 0.f;
		(*pVector)[diameter * (diameter - 1)].m_ForceAccumulator.x = 0.f;
		(*pVector)[diameter * (diameter - 1)].m_ForceAccumulator.y = 0.f;
		(*pVector)[diameter * (diameter - 1)].m_ForceAccumulator.z = 0.f;
	}
	else if (sidePin) {
		//Pin a whole side of the cloth
		for (size_t i = 0; i < diameter; i++) {
			(*pVector)[i].m_ForceAccumulator.x = 0.f;
			(*pVector)[i].m_ForceAccumulator.y = 0.f;
			(*pVector)[i].m_ForceAccumulator.z = 0.f;
		}
	}

	auto end_t = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE


	std::chrono::duration<double>  dif_t = end_t - start_t;
	std::chrono::duration<double>  particle_dif = particle_end - particle_start;
	std::chrono::duration<double>  f_dif = f_end - f_start;
	std::chrono::duration<double>  pvecdif = pvectime - start_t;
	std::chrono::duration<double>  overhead1 = overhead1_end - overhead1_start;
	std::chrono::duration<double>  overhead2 = overhead2_end - overhead2_start;
	std::cout << "TIme to change pointers " << pvecdif.count() << std::endl;
	std::cout << "overhead 1 " << overhead1.count() << std::endl; //Issue
	std::cout << "overhead 2 " << overhead2.count() << std::endl;
	std::cout << "Time deltas: \n" << "Particles: " << particle_dif.count() <<
		"\n" << "Forces and Tearing: " << f_dif.count() <<  "\nTotal: " << dif_t.count() << std::endl;
#endif //  VERBOSE


	sVector.clear(); sVector.shrink_to_fit();

}
