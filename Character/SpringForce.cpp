#include "SpringForce.hpp"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include "defined.h"

SpringForce::SpringForce(SubParticle*p1, SubParticle*p2, float dist, float ks, float kd, int p_ind1, int p_ind2, int i, int k, float tf) :
m_p1(p1), m_p2(p2), m_dist(dist), m_ks(ks), m_kd(kd), pind1(p_ind1), pind2(p_ind2),tearFactor(tf) {}

__device__ void SpringForce::draw()
{
	glBegin( GL_LINES );
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f( m_p1->m_Position.x, m_p1->m_Position.y, m_p1->m_Position.z);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f( m_p2->m_Position.x, m_p2->m_Position.y, m_p2->m_Position.z);
	glEnd();
}


__device__ void SpringForce::apply_force()
{
	auto& dotProduct = [this](Vec3 a, Vec3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	};
	
	Vec3 p1 = m_p1->m_Position;
	Vec3 p2 = m_p2->m_Position;
	Vec3 p1mp2 = p1 - p2;
	float pdist = vecNorm(p1mp2);
	Vec3 v1mv2 = m_p1->m_Velocity - m_p2->m_Velocity;
	float firstFactorF = m_ks * (pdist - m_dist);
	Vec3 f1 = (p1mp2 / pdist) * -1.f* (firstFactorF + m_kd * (dotProduct(v1mv2, p1mp2)) / pdist);
	Vec3 f2 = f1*-1.f;
	if (abs(vecNorm(f1)) > m_ks * tearFactor) teared = true;
	m_p1->m_ForceAccumulator += f1;
	m_p2->m_ForceAccumulator += f2;
}

__device__ bool SpringForce::willTear() {
	return teared;
}



