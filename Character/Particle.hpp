#ifndef _PARTICLE_HPP_
#define _PARTICLE_HPP_

#include <Vector/Vector.hpp>
#include <Graphics/Graphics.hpp>
#include "defined.h"

class Particle{

public:
	Particle(const Vec3& ConstructPos);
	Particle(float x, float y, float z);
	~Particle(void);

	void reset();

	Vec3 m_ConstructPos;
	Vec3 m_LastPosition;
	float m_Mass;
};

class SubParticle {
public:
	SubParticle() {

	};
	SubParticle(float x, float y, float z);
	SubParticle(Vec3 p) {
		m_Position = p;
		m_ConstructPos = p;
	};
	Vec3 m_ConstructPos = Vec3(0.f);
	Vec3 m_Position = Vec3(0.f);
	Vec3 m_Velocity = Vec3(0.f);
	Vec3 m_ForceAccumulator = Vec3(0.f);
	void clearForce();
	void reset();
	void draw();
};

#endif
