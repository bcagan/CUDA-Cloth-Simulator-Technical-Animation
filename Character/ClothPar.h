#ifndef CLOTHPAR
#define CLOTHPAR


struct Sphere
{
	float radius = 0.f;
	Vector3f center;
};


struct Triangle {
	Particle a;
	Particle b;
	Particle c;
};

//Object storage
static std::vector<Sphere> sVector;
static std::vector<Particle*> pVector;
static std::vector<Force*> fVector;

//Consts
#define RAND (((rand()%2000)/1000.f)-1.f)
#define DAMP 0.98f
#define EPS 0.001f
#define GRA 9.8f
#define FRICTION 0.5f

//Cloth and spatial grid parameters
const int radius = 30;
const int diameter = 2 * radius + 1;
const int gridDivisions = diameter / 6;

//Global parameters
float tclock = 0.f;
bool pin = false;
bool sidePin = false;
float springConstCollision = 15;
float springConstSphere = 40;
float collisionDist = 1.5;
bool windOn = false;
bool doDrawTriangle = false;
double ks = 100.f;
double kd = 1.f;
bool tearing = false;
bool stepAhead = false;


#endif // !CLOTHPAR