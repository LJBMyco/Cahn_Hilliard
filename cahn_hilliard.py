import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

class Cahn_Hilliard(object):

    def __init__(self, shape, sweeps, dt, dx, M, kappa, phi_0):

        self.shape = shape
        self.sweeps = sweeps
        self.dt = dt
        self.dx = dx
        self.M = M
        self.kappa = kappa
        self.phi_0 = phi_0

    def stop(self):

        self.stopsim = True

    """Method used to impliment periodic boundary conditions """
    def pbc(self, i):

        if (i > self.shape-1) or (i < 0):
            i = np.mod(i, self.shape)
            return i
        else:
            return i

    """Create the phi lattice with a given Initial condition and include noise"""
    def create_phi_lattice(self):

        self.phi_lattice = np.zeros((self.shape, self.shape))
        for i in range(self.shape):
            for j in range(self.shape):
                value = np.random.randint(-10,11)/100.0 + self.phi_0
                self.phi_lattice[i][j] = value

    """Calculate the initial mu lattice"""
    def create_mu_lattice(self):

        self.mu_lattice = np.zeros((self.shape, self.shape))
        for i in range(self.shape):
            for j in range(self.shape):
                self.mu_lattice[i][j] = self.calculate_mu(i,j)

    """Method that uses convoluation to update the lattices"""
    def laplacian(self, grid):

        kernal = [[0., 1., 0.,],
                        [1., -4., 1.],
                        [0., 1., 0.]]

        return signal.convolve2d(grid, kernal, boundary='wrap', mode='same')

    """Method that updates the mu lattice"""
    def convolve_mu(self):

        return  -0.1*self.phi_lattice + 0.1*self.phi_lattice**3.0 -  (self.kappa/(self.dx**2.0))*self.laplacian(self.phi_lattice)

    """Method that updates the phi lattice"""
    def convolve_phi(self):

        self.phi_lattice= self.phi_lattice + ( (self.M * self.dt)/(self.dx**2.0))*self.laplacian(self.convolve_mu())

    """The following two methods are not used but demonstrate the understanding of how to do the update"""
    def calculate_mu(self, i, j):

        mu = -0.1*self.phi_lattice[i][j] + 0.1*self.phi_lattice[i][j]**3.0 \
                - (self.kappa/(self.dx**2.0))*(self.phi_lattice[self.pbc(i+1)][j] + self.phi_lattice[self.pbc(i-1)][j] \
                 + self.phi_lattice[i][self.pbc(j+1)] + self.phi_lattice[i][self.pbc(j-1)] - 4*self.phi_lattice[i][j])

        return mu

    def calculate_phi(self, i, j):

        phi =  self.phi_lattice[i][j]  \
                +( (self.M * self.dt)/(self.dx**2.0))*(self.mu_lattice[self.pbc(i+1)][j] + self.mu_lattice[self.pbc(i-1)][j] \
                 + self.mu_lattice[i][self.pbc(j+1)] + self.mu_lattice[i][self.pbc(j-1)] - 4*self.mu_lattice[i][j])

        return phi

    """Method to calculate the free energy of the system"""
    def calculate_free_energy(self):

        grad_x, grad_y = np.gradient(self.phi_lattice)
        f = -(0.1/2.0)*self.phi_lattice**2.0 +(0.1/4.0)*self.phi_lattice**4.0 + (self.kappa/2.0)*(grad_x**2.0 + grad_y**2.0)
        f_sum = 0.0
        for i in range(self.shape):
            for j in range(self.shape):
                f_sum += f[i][j]

        f_sum *= self.dt**2.0
        return f_sum

    """Method that plots free energy as a function of time"""
    def run_model(self):
        self.create_phi_lattice()
        f_list = []
        time = []
        for n in range(self.sweeps):
            self.convolve_phi()
            if n % 10 == 0:
                f_energy = self.calculate_free_energy()
                f_list.append(f_energy)
                time.append(n)

        full_data = []
        for i in range(len(f_list)):
            full_data.append([time[i], f_list[i]])
        np.savetxt("free_energy_output.txt", full_data)


########################Animation###############################

    def run_animation_in_file(self):

        for s in range(10):
            self.convolve_phi()

    def animate(self, i):
        self.run_animation_in_file()
        self.mat.set_data(self.phi_lattice)
        return self.mat,

    def run(self):
        self.create_phi_lattice()
        self.create_mu_lattice()
        fig, ax = plt.subplots()
        self.mat = ax.imshow(self.phi_lattice, interpolation = 'gaussian', cmap='cool')
        fig.colorbar(self.mat)
        ani = FuncAnimation(fig, self.animate, interval = 10, blit = True)

        plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 9:
        print("Incorrect Number of Arguments Presented.")
        print("Usage: " + sys.argv[0] + "lattice Size, sweeps, dt, dx,M, kappa, phi_0, Data/Animate")
        quit()
    else:
        shape = int(sys.argv[1])
        sweeps = int(sys.argv[2])
        dt = float(sys.argv[3])
        dx = float(sys.argv[4])
        M = float(sys.argv[5])
        kappa = float(sys.argv[6])
        phi_0 = float(sys.argv[7])
        type = str(sys.argv[8])


    model = Cahn_Hilliard(shape, sweeps, dt, dx, M, kappa, phi_0)
    if type in ['D', 'd', 'data', 'Data']:
        model.run_model()
    elif type in ['A', 'a', 'animate', 'Animate']:
        model.run()
