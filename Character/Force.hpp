#ifndef _FORCE_HPP_
#define _FORCE_HPP_

class Force{

public:

	Force();
	virtual ~Force();

	virtual void draw() = 0;
	virtual void apply_force() = 0;
	virtual bool willTear() = 0;
};

#endif
