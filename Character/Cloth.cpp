
#include "Cloth.hpp"
#include "SpringForce.hpp"
#include <cstdlib>
#include <array>

#include <chrono>

#define CUDA

//#define VERBOSE


//Cloth and spatial grid parameters
int radius = 500;
int diameter = 2 * radius + 1;

int sceneSetting = 1; //Faster way of choosing var. presets

float totalTime = 0.f;
float totalIntegration = 0.f;
float totalParticles = 0.f;
float totalForces = 0.f;
float totalFrameTime = 0.f;
int numFrames = 0;
struct Sphere
{
	float radius = 0.f;
	Vec3 center;
};


struct Triangle {
	Particle a;
	Particle b;
	Particle c;
};

//Object storage
static std::vector<Sphere> sVector;
static std::vector<Particle*> pVector;
static std::vector<SubParticle> cudaPVector;
static std::vector<Force*> fVector;
static std::vector<std::pair<int, int>> cudaFVector;
static std::vector<std::pair<int, int>> cudaFOrderVector;
static std::vector<signed char> fTypeVector;
static bool* bVector;

//Consts
#define RAND (((rand()%2000)/1000.f)-1.f)
#define DAMP 0.98f
#define EPS 0.001f
#define GRA 9.8f
#define FRICTION 0.5f
#define MAX_SMOOTH_RENDER 124
#define SMOOTH_RENDER_FACTOR 149


//Global parameters
float tclock = 0.f;
bool pin = false;
bool sidePin = false;
float springConstCollision = 15;
float springConstSphere = 40;
float collisionDist = 1.5;
bool windOn = false;
bool doDrawTriangle = false;
double ks = 30.f;
double kd = 1.f;
bool tearing = false;
bool stepAhead = false;
bool spheresOn = true;
float sphereRadius = 4.f;
bool randHeight = false;
float normalDist = 7.5;
double dist = 7.5/2.0;
bool renderSave = false;
bool start = true;

//Special variable for setting benchmarking
int benchCount = 450;
bool benchmark = false;
std::chrono::steady_clock::time_point lastFramePoint;


void GPU_simulate(static std::vector<Sphere> sVector,
	static std::vector<SubParticle>* pVector,
	static std::vector<std::pair<int, int>>* fVector,
	static std::vector<std::pair<int, int>>* fOrderVector,
	static std::vector<signed char> fTypeVector,
	bool** bVector, const int radius, const int diameter, float dt, bool drawTriangles, bool tearing);
void cudaInit(size_t pVecSize, size_t fVecSize, size_t sVecSize);
void devFree();

Vector3f Vec3ToVector3f(Vec3 v) {
	return make_vector(v.x, v.y, v.z);
}

void triangleDraw(Vector3f a, Vector3f b, Vector3f c, Vector3f color) {
	Vector3f productVec = product(normalize(cross_product(a-b, c-b)), normalize(make_vector(0.f, -1.f, 1.f)));
	float colorFactor = abs(productVec.x + productVec.y + productVec.z);
	color *= (1.f/3.f+ (1.f-1.f/3.f)*colorFactor);
	glBegin(GL_TRIANGLES);
	glColor3f(color.x,color.y,color.z);
	glVertex3f(a.x,a.y,a.z);
	glColor3f(color.x, color.y, color.z);
	glVertex3f(b.x, b.y, b.z);
	glColor3f(color.x, color.y, color.z);
	glVertex3f(c.x, c.y, c.z);
	glEnd();
}

Cloth::Cloth() {
	lastFramePoint = std::chrono::high_resolution_clock::now();
	//All that follows are variables used through out the program that can be set along with the cloth
	dt = 1.f/60.f;
	float height = 10.f;
	normalDist = 7.5;
	dist = normalDist / radius;
	Vec3 center = Vec3(0.0f, height, 0.0f);
	float offset = dist;
	ks = 100.f;
	kd = ks/100.f/pow(3.f,int(ks) / 500); //Scales kd so it is high enough to
	//have good results, while accounting for the need to lower it for higher ks
	int clothOption = 0;
	spheresOn = true;
	sphereRadius = 4.f;
	pin = true;
	sidePin = false;
	springConstCollision = 25; //Self collision seems to be broken atm
	springConstSphere = 40;
	collisionDist = 15/radius;
	randHeight = false;
	windOn = false;
	doDrawTriangle = true;
	renderSave = true;
	tearing = false; //Visaulizes better without triangles being drawn
	integratorSet = SYMPLECTIC;
	//Forward is the least stable, Symplectic has the best results, 
	//Backwards leads to weak forces and is also extremely slow, 
	//Verlet might have an incorrect implementation. It doesn't explode like forward but it tends to twist quite heavily.



	if (sceneSetting == 0) { //Spheres
		spheresOn = true;
		clothOption = 0;
		pin = true;
		sidePin = false;
		windOn = false;
		tearing = false;
	}
	else if (sceneSetting == 1) { //Banner
		spheresOn = false;
		clothOption = 0;
		pin = false;
		sidePin = true;
		windOn = true;
		tearing = false;
	}
	else if (sceneSetting == 2) { //Pinned folded
		spheresOn = false;
		clothOption = 1;
		pin = true;
		sidePin = false;
		windOn = false;
		tearing = false;
	}
	else if (sceneSetting == 3) { //Unpinned folded
		spheresOn = false;
		clothOption = 1;
		pin = false;
		sidePin = false;
		windOn = false;
		tearing = false;
	}

	if (clothOption == 1) {
		//Creating particles (folded over)
		for (int i = -radius; i <= radius; i++) {
			for (int k = -radius; k <= radius; k++) {
				if (i <= 0) {
					Vec3 offsetVector = Vec3(offset * i, randHeight ? RAND / 2.5f : 0.f, offset * k);
					pVector.push_back(new Particle(center + offsetVector));
					SubParticle subP;
					subP.m_Position = center + offsetVector;
					subP.m_ConstructPos = center + offsetVector;
					cudaPVector.push_back(subP);
				}
				else {
					Vec3 offsetVector = Vec3(-offset * i, 5.0f + (randHeight ? RAND / 2.5f : 0.f), offset * k);
					pVector.push_back(new Particle(center + offsetVector));
					SubParticle subP;
					subP.m_Position = center + offsetVector;
					subP.m_ConstructPos = center + offsetVector;
					cudaPVector.push_back(subP);
				}

			}
		}
	}
	else {
		//Creating particles (Flat)
		for (int i = -radius; i <= radius; i++) {
			for (int k = -radius; k <= radius; k++) {
				Vec3 offsetVector(offset * i, randHeight ? RAND / 2.5f : 0.f, offset * k);
				pVector.push_back(new Particle(center + offsetVector));
				SubParticle subP;
				subP.m_Position = center + offsetVector;
				subP.m_ConstructPos = center + offsetVector;
				cudaPVector.push_back(subP);
			}
		}
	}

	std::vector<int> curForce(cudaPVector.size());

	float stretchTearFactor = 4.f;// 2.f;
	float shearTearFactor = 4.f;// 2.f * sqrt(2);
	float bendTearFactor = 4.f;
	//Stretch and bend Forces
	for (int i = 0; i < diameter; i++) {
		for (int k = 0; k < diameter; k++) {
			//stretch
			if (i != diameter - 1) {
				fVector.push_back(new SpringForce(&(cudaPVector[diameter * i + k]), &(cudaPVector[(diameter * (i + 1) + k)]),
					dist, ks, kd, diameter * i + k, (diameter * (i + 1) + k), i, k, stretchTearFactor));
				cudaFVector.push_back(std::make_pair(diameter * i + k, (diameter * (i + 1) + k)));
				cudaFOrderVector.push_back(std::make_pair(curForce[diameter * i + k], curForce[(diameter * (i + 1) + k)]));
				curForce[diameter * i + k]++; curForce[(diameter * (i + 1) + k)]++;
				fTypeVector.push_back(1);
			}
			if (k != diameter - 1){
				fVector.push_back(new SpringForce(&(cudaPVector[diameter * i + k]), &(cudaPVector[diameter * i + (k + 1)]),
					dist, ks, kd, diameter * i + k, diameter * i + (k + 1), i, k, stretchTearFactor));
				cudaFVector.push_back(std::make_pair(diameter * i + k, diameter * i + (k + 1)));
				cudaFOrderVector.push_back(std::make_pair(curForce[diameter * i + k], curForce[(diameter * i + (k + 1))]));
				curForce[diameter * i + k]++; curForce[diameter * i + (k + 1)]++;
				fTypeVector.push_back(1);
			}
			//bend
			if (i != diameter - 2 && i < diameter - 1) {
				fVector.push_back(new SpringForce(&(cudaPVector[diameter * i + k]), &(cudaPVector[(diameter * (i + 2) + k)]),
					2*dist, ks / 2.f, kd * 2.f, diameter * i + k, diameter * (i + 2) + k, i, k, bendTearFactor));
				cudaFVector.push_back(std::make_pair(diameter * i + k, diameter * (i + 2) + k));
				cudaFOrderVector.push_back(std::make_pair(curForce[diameter * i + k], curForce[(diameter * (i + 2) + k)]));
				curForce[diameter * i + k]++; curForce[(diameter * (i + 2) + k)]++;
				fTypeVector.push_back(2);
			}
			if (k != diameter - 2 && k < diameter - 1) {
				fVector.push_back(new SpringForce(&(cudaPVector[diameter * i + k]), &(cudaPVector[diameter * i + (k + 2)]),
					2*dist, ks / 2.f, kd * 2.f, diameter * i + k, diameter * i + (k + 2), i, k, bendTearFactor));
				cudaFVector.push_back(std::make_pair(diameter * i + k, diameter * i + (k + 2)));
				cudaFOrderVector.push_back(std::make_pair(curForce[diameter * i + k], curForce[(diameter * i + (k + 2))]));
				curForce[diameter * i + k]++; curForce[diameter * i + (k + 2)]++;
				fTypeVector.push_back(2);
			}
		}
	}

	//Shear forces
	//TL->BR from i
	for (int i = 0; i < diameter - 1; i++) {
		for (int offset = 0; offset < diameter - i - 1; offset++) {
			fVector.push_back(new SpringForce(&(cudaPVector[diameter * (i + offset) + offset]), &(cudaPVector[diameter * (i + offset + 1) + offset + 1]), 
				sqrt(2)*dist, ks, kd, diameter * (i + offset) + offset, diameter * (i + offset + 1) + offset + 1, i + offset, offset, shearTearFactor));
			cudaFVector.push_back(std::make_pair(diameter * (i + offset) + offset, diameter * (i + offset + 1) + offset + 1));
			cudaFOrderVector.push_back(std::make_pair(curForce[diameter * (i + offset) + offset], curForce[diameter * (i + offset + 1) + offset + 1]));
			curForce[diameter * (i + offset) + offset]++; curForce[diameter * (i + offset + 1) + offset + 1]++;
			fTypeVector.push_back(0);
		}
	}
	//TL->BR from k
	for (int k = 1; k < diameter - 1; k++) {
		for (int offset = 0; offset < diameter - k - 1; offset++) {
			fVector.push_back(new SpringForce(&(cudaPVector[diameter * (offset)+k + offset]), &(cudaPVector[diameter * (offset + 1) + k + offset + 1]), 
				sqrt(2)* dist, ks, kd, diameter * (offset)+k + offset, diameter * (offset + 1) + k + offset + 1,offset, k + offset, shearTearFactor));
			cudaFVector.push_back(std::make_pair(diameter* (offset)+k + offset, diameter* (offset + 1) + k + offset + 1));
			cudaFOrderVector.push_back(std::make_pair(curForce[diameter * (offset)+k + offset], curForce[diameter * (offset + 1) + k + offset + 1]));
			curForce[diameter * (offset)+k + offset]++; curForce[diameter * (offset + 1) + k + offset + 1]++;
			fTypeVector.push_back(0);
		}
	}
	//TR->BL from i
	for (int i = 0; i < diameter - 1; i++) {
		for (int offset = 0; offset < i; offset++) {
			int koffset = i - offset;
			fVector.push_back(new SpringForce(&(cudaPVector[diameter * (offset)+koffset]), &(cudaPVector[diameter * (offset + 1) + koffset - 1]), 
				sqrt(2)* dist, ks, kd, diameter * (offset)+koffset, diameter * (offset + 1) + koffset - 1,offset, koffset, shearTearFactor));
			cudaFVector.push_back(std::make_pair(diameter* (offset)+koffset, diameter* (offset + 1) + koffset - 1));
			cudaFOrderVector.push_back(std::make_pair(curForce[diameter * (offset)+koffset], curForce[diameter * (offset + 1) + koffset - 1]));
			curForce[diameter * (offset)+koffset]++; curForce[diameter * (offset + 1) + koffset - 1]++;
			fTypeVector.push_back(0);
		}
	}
	//TR->BL from k
	for (int k = diameter - 2; k > 0; k--) {
		for (int offset = 0; offset < diameter - k - 1; offset++) {
			int koffset = k + offset;
			int ioffset = diameter - 1 - offset;
			fVector.push_back(new SpringForce(&(cudaPVector[diameter * (ioffset)+koffset]), &(cudaPVector[diameter * (ioffset - 1) + koffset + 1]), 
				sqrt(2)* dist, ks, kd, diameter* (ioffset)+koffset, diameter* (ioffset - 1) + koffset + 1,ioffset, offset, shearTearFactor));
			cudaFVector.push_back(std::make_pair(diameter* (ioffset)+koffset, diameter* (ioffset - 1) + koffset + 1));
			cudaFOrderVector.push_back(std::make_pair(curForce[diameter * (ioffset)+koffset], curForce[diameter * (ioffset - 1) + koffset + 1]));
			curForce[diameter * (ioffset)+koffset]++; curForce[diameter * (ioffset - 1) + koffset + 1]++;
			fTypeVector.push_back(0);
		}
	}
	


	if (spheresOn) {
		//Adding spheres
		Sphere sphere0;
		sphere0.radius = sphereRadius;
		sphere0.center = Vec3(-4.f, 2.f, -4.f);
		sVector.push_back(sphere0);
		Sphere sphere1;
		sphere1.radius = sphereRadius;
		sphere1.center = Vec3(4.f, 2.f, -4.f);
		sVector.push_back(sphere1);
		Sphere sphere2;
		sphere2.radius = sphereRadius;
		sphere2.center = Vec3(-4.f, 2.f, 4.f);
		sVector.push_back(sphere2);
		Sphere sphere3;
		sphere3.radius = sphereRadius;
		sphere3.center = Vec3(4.f, 2.f, 4.f);
		sVector.push_back(sphere3);
	}
	bVector = (bool*)calloc(fVector.size(), sizeof(bool));

	cudaInit(pVector.size(), fVector.size(), sVector.size());
}

Cloth::~Cloth(){
	cudaPVector.clear();
	sVector.clear();
	cudaFVector.clear();
	pVector.clear();
	fVector.clear();
	devFree();
}

void Cloth::reset(){
	tclock = 0.f;
	int size = pVector.size();
	for(int ii=0; ii<size; ii++){
		pVector[ii]->reset();
		cudaPVector[ii].reset();
	}
}

void Cloth::draw(){
	int renderFactor = 1;
	if (!doDrawTriangle) {
		int size = pVector.size();
		if(renderSave && radius > MAX_SMOOTH_RENDER) renderFactor = (size + (SMOOTH_RENDER_FACTOR * SMOOTH_RENDER_FACTOR))
			/ ((1 + SMOOTH_RENDER_FACTOR) * (1 + SMOOTH_RENDER_FACTOR));
		for (int ii = 0; ii < size; ii = ii + renderFactor) {
			cudaPVector[ii].draw();
		}

		size = fVector.size();
		if (renderSave && radius > MAX_SMOOTH_RENDER)renderFactor = (size + (SMOOTH_RENDER_FACTOR * SMOOTH_RENDER_FACTOR)) 
			/ ((1 + SMOOTH_RENDER_FACTOR) * (1 + SMOOTH_RENDER_FACTOR));
		for (int ii = 0; ii < size; ii = ii + renderFactor) {
#ifdef CUDA
			if(tearing && bVector[ii]) continue;
#endif // CUDA

			fVector[ii]->draw();
		}
	}
	else {
		if (renderSave && radius > MAX_SMOOTH_RENDER) renderFactor = (diameter + SMOOTH_RENDER_FACTOR) / (1 + SMOOTH_RENDER_FACTOR);
		for (int i = 0; i < diameter - 1; i = i + renderFactor) {
			for (int k = 0; k < diameter - 1; k = k + renderFactor) {
				Vector3f color = make_vector(1.f, 0.f, 0.f); //Ensure both have similar noemals by ordering as such
				triangleDraw(Vec3ToVector3f(cudaPVector[i * diameter + k].m_Position), 
							 Vec3ToVector3f(cudaPVector[(i + renderFactor) * diameter + k].m_Position),
							 Vec3ToVector3f(cudaPVector[i * diameter + k + renderFactor].m_Position), color);
				triangleDraw(Vec3ToVector3f(cudaPVector[(i + renderFactor) * diameter + k].m_Position),
					         Vec3ToVector3f(cudaPVector[(i + renderFactor) * diameter + k + renderFactor].m_Position),
					         Vec3ToVector3f(cudaPVector[i * diameter + k + renderFactor].m_Position), color);
			}
		}
	}

}


void Cloth::simulation_step() {


#ifdef CUDA
	GPU_simulate(sVector, &cudaPVector, &cudaFVector, &cudaFOrderVector, fTypeVector, &bVector, radius, diameter, dt, doDrawTriangle, tearing);

#endif // CUDA

#ifndef CUDA
	cpu_simulate();
#endif // !CUDA

	tclock += dt;

	if (start) {
		start = false;
		lastFramePoint = std::chrono::high_resolution_clock::now();
	}

	auto nextFramePoint = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> tempFrameDelta = nextFramePoint - lastFramePoint;
	totalFrameTime += tempFrameDelta.count();
	lastFramePoint = nextFramePoint;

#ifdef VERBOSE
	if (!benchmark || benchCount == numFrames){
		std::cout << "Average total: " << totalTime / (float)numFrames << std::endl;
		std::cout << "Average particles: " << totalParticles / (float)numFrames << std::endl;
		std::cout << "Average forces: " << totalForces / (float)numFrames << std::endl;
		std::cout << "Average integration: " << totalIntegration / (float)numFrames << std::endl;
		std::cout << "Average frametime: " << totalFrameTime / (float) numFrames << std::endl;
	}

#endif // VERBOSE

	

}

void Cloth::euler_step(Integrator integrator){
	//Symplectic Euler
	if (integrator == SYMPLECTIC) {
		for (auto& p : cudaPVector) {
			p.m_Velocity = Vec3(DAMP) * p.m_Velocity + Vec3(DAMP) * p.m_ForceAccumulator * dt;
			p.m_Position += Vec3(dt) * p.m_Velocity;
		}
	}
	else if (integrator == VERLET) {
		for (int i = 0; i < cudaPVector.size(); i++) {
			auto& p = cudaPVector[i];
			auto& pFull = pVector[i];
			Vec3 tempPos = Vec3(p.m_Position.x, p.m_Position.y, p.m_Position.z);
			p.m_Position = Vec3(2) * p.m_Position - pFull->m_LastPosition + Vec3(dt) * Vec3(dt) * p.m_ForceAccumulator * DAMP;
			pFull->m_LastPosition = tempPos;
			
		}
	}
	else if (integrator == BACKWARD) {
		for (auto& p : cudaPVector) {
			stepAhead = true;
			simulation_step();
			p.m_Velocity = Vec3(DAMP) * p.m_Velocity + Vec3(DAMP) * p.m_ForceAccumulator * dt;
			stepAhead = false;
			p.m_Position = p.m_Position + Vec3(dt) * p.m_Velocity;
		}

	}
	else {
		for (auto& p : cudaPVector) {
			p.m_Position += p.m_Velocity * dt;
			p.m_Velocity = p.m_Velocity * DAMP + p.m_ForceAccumulator * dt * DAMP;
		}
	}
}


void CPUPrintVec3C(Vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z << std::endl;
}



void Cloth::cpu_simulate() {

	auto start_t = std::chrono::high_resolution_clock::now();

	//Wind force information
	Vec3 windDir = Vec3(1.3f, 0.f, 0.f);
	auto windMagnitude = [this](Vec3 pos) {
		float x = pos.x; float y = pos.y; float z = pos.z;
		return 7.f * (cos(tclock * 10.f) + 1.f) * abs(sin(z + tclock * 5) + cos(y + tclock * 5) / 3.f);
	};



	auto particle_start = std::chrono::high_resolution_clock::now();
	//Clear force accumulators for all particles and then apply gravity and then wind and sphere forces
	for (auto& pMini : cudaPVector) {
		pMini.clearForce();
		float gOffset = -GRA;
		pMini.m_ForceAccumulator = pMini.m_ForceAccumulator 
			+ Vec3(0, gOffset, 0);
		if (windOn) pMini.m_ForceAccumulator = 
			pMini.m_ForceAccumulator + windDir * windMagnitude(pMini.m_Position);

		//Sphere collisions via spring forces
		for (auto& s : sVector) {
			if (vecNorm(pMini.m_Position - s.center) < s.radius) {
				SubParticle tempSphereParticle(Vec3(0.f));
				SpringForce collideForce(&pMini, &tempSphereParticle, (2.f + radius / 40.f) * s.radius,
					ks != 0.f && ks < 50 ? springConstSphere * (5.8f / (sqrt(ks))) : springConstSphere, //scale sphere ks to allow functioning at lower cloth ks's (10-50)
					(2.f + radius / 40.f) * s.radius, 0, 0, INFINITY); //Cannot find a good values for ks < 10
				collideForce.apply_force(); //Distance const scaled by radii to avoid clipping
			}
		}

		if (pMini.m_Position.y <= 2*EPS) {
			//Apply friction force
			Vec3 frictionVelVector = Vec3(-pMini.m_Velocity.x, 0.f, -pMini.m_Velocity.z);
			Vec3 frictionVector = vecNormalize(frictionVelVector);
			float frictionVel = vecNorm(frictionVelVector);
			pMini.m_ForceAccumulator = pMini.m_ForceAccumulator +  frictionVelVector * FRICTION;
		}
	}
	auto particle_end = std::chrono::high_resolution_clock::now();
	
	auto f_start = std::chrono::high_resolution_clock::now();

	//Apply spring forces and tearing if need be
	std::vector<std::vector<Force*>::iterator> eraseForceList;
	for (auto& f = fVector.end(); f != fVector.begin(); f--) { //Listing backwards avoids indexing errors
		auto& fPred = f - 1;
		(*fPred)->apply_force();
		if ((*fPred)->willTear() && tearing && !stepAhead) {
			eraseForceList.push_back(fPred);
		}
	}
	auto f_end = std::chrono::high_resolution_clock::now();
	if (tearing && !stepAhead) {
		for (auto& f = eraseForceList.begin(); f != eraseForceList.end(); f++) {
			fVector.erase(*f);
		}
	}
	
	


	//Pin corners
	if (pin) {
		cudaPVector[0].m_ForceAccumulator = Vec3(0.f, 0.f, 0.f);
		cudaPVector[diameter - 1].m_ForceAccumulator = Vec3(0.f, 0.f, 0.f);
		cudaPVector[(diameter * diameter) - 1].m_ForceAccumulator = Vec3(0.f, 0.f, 0.f);
		cudaPVector[diameter * (diameter - 1)].m_ForceAccumulator = Vec3(0.f, 0.f, 0.f);
	}
	else if (sidePin) {
		//Pin a whole side of the cloth
		for (size_t i = 0; i < diameter; i++) {
			cudaPVector[i].m_ForceAccumulator = Vec3(0.f, 0.f, 0.f);
		}
	}

	auto start_integration = std::chrono::high_resolution_clock::now();
	if (!stepAhead) {
		//Then, we can move forward
		euler_step(integratorSet);
		//Floor collision
		for (auto& p : cudaPVector) {
			if (p.m_Position.y < EPS) { //EPS used to avoid z-fighting
				p.m_Position = Vec3(p.m_Position.x, EPS, p.m_Position.z);
			}
		}
	}

	auto end_t = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE


	std::chrono::duration<double>  dif_t = end_t - start_t;
	std::chrono::duration<double>  particle_dif = particle_end - particle_start;
	std::chrono::duration<double>  f_dif = f_end - f_start;
	std::chrono::duration<double> i_dif = end_t - start_integration;
	if (!benchmark) {
		std::cout << "Time deltas: \n" << "Particles: " << particle_dif.count() <<
			"\n" << "Forces and Tearing: " << f_dif.count() << "\nTotal: " << dif_t.count() << std::endl;
	}
	totalTime += dif_t.count();
	totalParticles += particle_dif.count();
	totalForces += f_dif.count();
	totalIntegration += i_dif.count();
	numFrames++;

#endif //  VERBOSE
}
