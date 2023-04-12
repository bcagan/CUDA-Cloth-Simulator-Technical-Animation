#include "Particle.hpp"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>


__device__ Particle::Particle(const Vec3& ConstructPos)
{
	m_ConstructPos = Vec3(ConstructPos.x, ConstructPos.y, ConstructPos.z);
	m_LastPosition = Vec3(0.0f, 0.0f, 0.0f);
	m_Mass = 1.0f;
}


__device__ void Particle::reset()
{
	m_LastPosition = m_ConstructPos;
}

__device__ void SubParticle::draw()
{
	const double h = 0.3;
	glColor3f(1.f, 0.f, 0.f);
	glBegin(GL_QUADS);
	glVertex3f(m_Position.x-h/2.0, m_Position.y-h/2.0, m_Position.z);
	glVertex3f(m_Position.x+h/2.0, m_Position.y-h/2.0, m_Position.z);
	glVertex3f(m_Position.x+h/2.0, m_Position.y+h/2.0, m_Position.z);
	glVertex3f(m_Position.x-h/2.0, m_Position.y+h/2.0, m_Position.z);
	glEnd();
}

void SubParticle::clearForce() {
	m_ForceAccumulator = Vec3(0.f);
}

void SubParticle::reset() {
	m_Position = m_ConstructPos;
	m_Velocity = Vec3(0.0f, 0.0f, 0.0f);
	m_ForceAccumulator = Vec3(0.0f, 0.0f, 0.0f);
}

