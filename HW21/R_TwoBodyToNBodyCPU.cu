// Name: Nathan S.
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the terminal you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

#define NUMBER_OF_SPHERES 10 
#define WallDamp 0.89 // To slow it down when hitting the wall

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES];
float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];
float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
float mass[NUMBER_OF_SPHERES];


// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
float r[NUMBER_OF_SPHERES], g[NUMBER_OF_SPHERES], b[NUMBER_OF_SPHERES]; // For colors

// Function prototypes
void set_initial_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initial_conditions()
{
    time_t t;
    srand((unsigned) time(&t));
    
    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        int valid;
		do {
			// Random sphere placement, within the box, hence the division by 2, because it starts from the centre of the box
			// Subtract by diameter so the sphere's edge wouldn't be outside the box
			px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

			valid = 1; // To break the loop, place it once

			// Check overlap with previously placed spheres
			for (int j = 0; j < i; j++) {
				float dx = px[i] - px[j];
				float dy = py[i] - py[j];
				float dz = pz[i] - pz[j];
				float dist = sqrt(dx*dx + dy*dy + dz*dz);

				if (dist < DIAMETER) {
					valid = 0;  // overlap detected
					break;
				}
			}
		} while (!valid);

		// Random speed ranging from [-MAX_VELOCITY, MAX_VELOCITY] hence multiplied by 2
		// Negative speed would move opposite direction (left/down)
        vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
        vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
        vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;

        mass[i] = 1.0;

		// Random color
		r[i] = (float)rand()/RAND_MAX;
		g[i] = (float)rand()/RAND_MAX;
		b[i] = (float)rand()/RAND_MAX;
    }
}


void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
    float radius = DIAMETER * 0.5f;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    Drawwirebox();

    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
		glColor3f(r[i], g[i], b[i]);
        glPushMatrix();
        glTranslatef(px[i], py[i], pz[i]);
        glutSolidSphere(radius, 20, 20);
        glPopMatrix();
    }

    glutSwapBuffers();
}


void keep_in_box()
{
	// Starting from the center, the range is [-halfBox, halfBox]
    float halfBox = (LENGTH_OF_BOX - DIAMETER)/2.0;
	// Reduce speed by 10% after hitting the wall
	float wallDamp = WallDamp; 

	// Because of how reflection work, if you go a certain distance past the wall, then the new position should be twice that
    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
		// For x direction
        if (px[i] > halfBox) {
			px[i] = 2.0*halfBox - px[i];
			vx[i] = -vx[i] * wallDamp;
		}
        else if (px[i] < -halfBox) {
			px[i] = -2.0*halfBox - px[i];
			vx[i] = -vx[i] * wallDamp;
		}

		// For y direction
        if (py[i] > halfBox) {
			py[i] = 2.0*halfBox - py[i];
			vy[i] = -vy[i] * wallDamp;
		}
        else if (py[i] < -halfBox) {
			py[i] = -2.0*halfBox - py[i];
			vy[i] = -vy[i] * wallDamp;
		}

		// For z direction
        if (pz[i] > halfBox) {
			pz[i] = 2.0*halfBox - pz[i];
			vz[i] = -vz[i] * wallDamp;
		}
        else if (pz[i] < -halfBox) { 
			pz[i] = -2.0*halfBox - pz[i];
			vz[i] = -vz[i] * wallDamp;
		}
    }
}


void get_forces()
{
    // Reset forces
    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        fx[i] = fy[i] = fz[i] = 0.0;
    }

    // Pairwise forces
    for (int i = 0; i < NUMBER_OF_SPHERES - 1; i++) {
        for (int j = i + 1; j < NUMBER_OF_SPHERES; j++) {
			// Calculating the distance between sphere i and j
            float dx = px[j] - px[i];
            float dy = py[j] - py[i];
            float dz = pz[j] - pz[i];
            float r2 = dx*dx + dy*dy + dz*dz; // Squared distance
            float r = sqrt(r2); // Actual distance
            
			// If they're basically at the same spot, skip it
            if (r < 1e-6) continue;  // To prevent divide-by-zero

            float forceMag = mass[i]*mass[j]*GRAVITY/r2;

            // Handle collisions
            if (r < DIAMETER) // If the distance is too close, repels the spheres
            {
                float dvx = vx[j] - vx[i];
                float dvy = vy[j] - vy[i];
                float dvz = vz[j] - vz[i];
                float inout = dx*dvx + dy*dvy + dz*dvz;
                if (inout <= 0.0) // If they're moving towards each other, repels them
                    forceMag += SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
                else // If they're moving away, reduce the force
                    forceMag += PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
            }

			// Apply the force in proper x, y, z direction
            float fxij = forceMag*dx/r;
            float fyij = forceMag*dy/r;
            float fzij = forceMag*dz/r;

			// "i" would have force applied by "+f" while "j" would have force by "-f"
            fx[i] += fxij;
            fy[i] += fyij;
            fz[i] += fzij;
            fx[j] -= fxij;
            fy[j] -= fyij;
            fz[j] -= fzij;
        }
    }
}


void move_bodies(float time)
{
    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        float dtFactor = (time == 0.0) ? 0.5*DT : DT; // Time = 0 (at first), half step, later full step

        vx[i] += dtFactor*(fx[i] - DAMP*vx[i])/mass[i];
        vy[i] += dtFactor*(fy[i] - DAMP*vy[i])/mass[i];
        vz[i] += dtFactor*(fz[i] - DAMP*vz[i])/mass[i];

        px[i] += DT*vx[i];
        py[i] += DT*vy[i];
        pz[i] += DT*vz[i];
    }
    keep_in_box();
}


void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initial_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}


