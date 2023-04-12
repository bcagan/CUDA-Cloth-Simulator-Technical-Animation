#pragma once




#ifndef _CLOTH_HPP_
#define _CLOTH_HPP_

#include "Particle.hpp"
#include "Force.hpp"
#include "SpringForce.hpp"

#include <vector>



enum Integrator
{
	SYMPLECTIC, FORWARD, BACKWARD, VERLET
};
class Cloth {
public:
	Cloth();
	~Cloth();
	Integrator integratorSet = VERLET;

public:
	float dt = 0.1f;

	void reset();
	void draw();

	void simulation_step();
	void euler_step(Integrator integrator);

	void cpu_simulate();
};

#endif

