#ifndef _SPRINGFROCE_HPP_
#define _SPRINGFROCE_HPP_

#include "Particle.hpp"
#include "Force.hpp"

class SpringForce : public Force
{

public:
	SpringForce(SubParticle* p1, SubParticle* p2, float dist, float ks, float kd, int p_ind1, int p_ind2, int i, int k, float tearFactorIn);
	SpringForce(SubParticle* p1, SubParticle* p2, float dist, float ks, float kd, int p_ind, int p_ind2, float tearFactorIn);

	virtual void draw();
	virtual void apply_force();
	virtual bool willTear();


	SubParticle* const m_p1; //particle1
	SubParticle* const m_p2; //particle2
	float const m_dist;  //rest length
	float const m_ks, m_kd; //spring strength constants
	float tearFactor;
	bool teared = false;
	int pind1;
	int pind2;

private:
};
#endif
