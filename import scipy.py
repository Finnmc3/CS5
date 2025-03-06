import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = scipy.constants.gravitational_constant  # Define G globally for readability

class Planet:
    '''Class for defining a planet in the system'''
    
    def __init__(self, mass, iniVel, iniPos):
        self.mass = mass
        self.iniVel = np.array(iniVel, dtype=float)
        self.iniPos = np.array(iniPos, dtype=float)
        
    def plMass(self):
        return self.mass
    
    def plVel(self):
        return self.iniVel
    
    def plPos(self):
        return self.iniPos

def accelerations(planet1, planet2, timestamps):
    """Computes position, velocity, and acceleration for a two-body system."""
    
    dt = 10  # Set a larger time step for better visualization

    posP1 = np.zeros((timestamps, 2))
    posP2 = np.zeros((timestamps, 2))
    
    velP1 = np.zeros((timestamps, 2))
    velP2 = np.zeros((timestamps, 2))

    posP1[0] = planet1.plPos()
    posP2[0] = planet2.plPos()

    velP1[0] = planet1.plVel()
    velP2[0] = planet2.plVel()
    
    for i in range(timestamps - 1):
        radius_vector = posP2[i] - posP1[i]  
        distance = np.linalg.norm(radius_vector)

        # Compute acceleration due to gravity
        force_magnitude = G / (distance ** 3)
        accP1 = force_magnitude * planet2.plMass() * radius_vector
        accP2 = -force_magnitude * planet1.plMass() * radius_vector  # Newton's Third Law

        # Update velocities
        velP1[i + 1] = velP1[i] + accP1 * dt
        velP2[i + 1] = velP2[i] + accP2 * dt
        
        # Update positions
        posP1[i + 1] = posP1[i] + velP1[i + 1] * dt
        posP2[i + 1] = posP2[i] + velP2[i + 1] * dt

    return posP1[:, 0], posP1[:, 1], posP2[:, 0], posP2[:, 1]

def anim(plan1, plan2, timestamps):
    """Generates an animation of two celestial bodies orbiting."""
    
    x1, y1, x2, y2 = accelerations(plan1, plan2, timestamps)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(min(np.min(x1), np.min(x2)) - 0.1, max(np.max(x1), np.max(x2)) + 0.1)
    ax.set_ylim(min(np.min(y1), np.min(y2)) - 0.1, max(np.max(y1), np.max(y2)) + 0.1)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Orbit Animation")

    planet1_dot, = ax.plot([], [], 'bo', markersize=8, label="Planet 1")
    planet2_dot, = ax.plot([], [], 'ro', markersize=8, label="Planet 2")
    trajectory1, = ax.plot([], [], 'b-', alpha=0.5)
    trajectory2, = ax.plot([], [], 'r-', alpha=0.5)

    ax.legend()

    def init():
        planet1_dot.set_data([], [])
        planet2_dot.set_data([], [])
        trajectory1.set_data([], [])
        trajectory2.set_data([], [])
        return planet1_dot, planet2_dot, trajectory1, trajectory2

    def update(frame):
        if frame < len(x1):
            planet1_dot.set_data(x1[frame], y1[frame])
            planet2_dot.set_data(x2[frame], y2[frame])
            trajectory1.set_data(x1[:frame], y1[:frame])
            trajectory2.set_data(x2[:frame], y2[:frame])
        return planet1_dot, planet2_dot, trajectory1, trajectory2

    ani = animation.FuncAnimation(fig, update, frames=len(x1), init_func=init, interval=30, blit=False)
    plt.show()

def main():
    Mars_mass = 6.4185e23  # kg
    Phobos_mass = 1.06e16  # kg
    Phobos_orbit_radius = 9.3773e6  # meters

    # Initial conditions
    Mars = Planet(mass=Mars_mass, iniVel=[0, 0], iniPos=[0, 0])
    
    # Phobos has an initial velocity perpendicular to its position vector
    orbital_velocity = np.sqrt(G * Mars_mass / Phobos_orbit_radius)  # Correct formula
    
    Phobos = Planet(mass=Phobos_mass, iniVel=[0, orbital_velocity], iniPos=[Phobos_orbit_radius, 0])
    
    anim(Mars, Phobos, timestamps=500)  # Run animation with 500 time steps

main()
