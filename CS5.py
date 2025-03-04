import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Planet:
    '''Class for defining a planet in the system'''
    
    def __init__(self,mass,iniVel,iniPos):
        self.mass = mass
        self.iniVel = np.array(iniVel)
        self.iniPos = np.array(iniPos)
        
    def plMass(self):
        return self.mass
    
    def plVel(self):
        return self.iniVel
    
    def plPos(self):
        return self.iniPos
        
def accelerations(planet1,planet2,timestamps,iniRadius):
    
    timIncr = 0.319/timestamps
    
    posP1 = np.zeros((timestamps, 2), dtype = float)
    posP2 = np.zeros((timestamps, 2), dtype = float)
    
    rad = np.zeros((timestamps, 2), dtype = float)
    
    
    accP1 = np.zeros((timestamps, 2), dtype = float)
    accP2 = np.zeros((timestamps, 2), dtype = float)
    
    velP1 = np.zeros((timestamps, 2), dtype = float)
    velP2 = np.zeros((timestamps, 2), dtype = float)
    
    
    posP1[0] = planet1.plPos()
    posP2[0] = planet2.plPos()

    velP1[0] = planet1.plVel()
    velP2[0] = planet2.plVel()
    
    rad[0] = iniRadius
    
    for i in range(timestamps - 1):
        
        accP1[i] = -scipy.constants.gravitational_constant * (planet2.plMass()/(np.linalg.norm(rad[i]))**3) * rad[i]
        accP2[i] = -scipy.constants.gravitational_constant * (planet1.plMass()/(np.linalg.norm(rad[i]))**3) * rad[i]
        
        if i == timestamps - 2:
            continue
    
        else:
    
            velP1[i + 1] = velP1[i] + (accP1[i] * timIncr)
            velP2[i + 1] = velP2[i] + (accP2[i] * timIncr)
            
            posP1[i + 1] = posP1[i] + (velP1[i + 1] * timIncr)
            posP2[i + 1] = posP2[i] + (velP2[i + 1] * timIncr)
            
            rad[i + 1] = posP1[i+ 1] - posP2[i + 1]
            
    x1, y1 = posP1[:, 0], posP1[:, 1]
    x2, y2 = posP2[:, 0], posP2[:, 1]
        
    return x1, y1, x2, y2


    
def anim(plan1,plan2,timestamps, iniRadius):
        x1,y1,x2,y2 = accelerations(plan1,plan2,timestamps,iniRadius)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(min(np.min(x1), np.min(x2)) - 0.1, max(np.max(x1), np.max(x2)) + 0.1)
        ax.set_ylim(min(np.min(y1), np.min(y2)) - 0.1, max(np.max(y1), np.max(y2)) + 0.1)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Orbit Animation")

        # Plot objects for planets
        planet1_dot, = ax.plot([], [], 'bo', markersize=8, label="Planet 1")  # Blue planet
        planet2_dot, = ax.plot([], [], 'ro', markersize=8, label="Planet 2")  # Red planet
        trajectory1, = ax.plot([], [], 'b-', alpha=0.5)  # Trajectory for planet 1
        trajectory2, = ax.plot([], [], 'r-', alpha=0.5)  # Trajectory for planet 2

        ax.legend()

        # Initialize function
        def init():
            planet1_dot.set_data([], [])
            planet2_dot.set_data([], [])
            trajectory1.set_data([], [])
            trajectory2.set_data([], [])
            return planet1_dot, planet2_dot, trajectory1, trajectory2

        # Update function for animation
        def update(frame):
            planet1_dot.set_data(x1[frame], y1[frame])
            planet2_dot.set_data(x2[frame], y2[frame])
            
            # Update trajectories
            trajectory1.set_data(x1[:frame], y1[:frame])
            trajectory2.set_data(x2[:frame], y2[:frame])
            
            return planet1_dot, planet2_dot, trajectory1, trajectory2

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(x1), init_func=init, interval=30, blit=True)

        plt.show()


def main():
    Mars = Planet(6.4185 * 10 ** 23,[0, 0],[0, 0])
    Phobos = Planet(1.06 * 10 ** 16, [9.3773*10**6, 0],[0, np.sqrt(scipy.constants.gravitational_constant*9.3773*10**6/9.3773*10**6)])
    

    print(anim(Mars,Phobos,100,9.3733 * 10 ** 6))
    
    
main()