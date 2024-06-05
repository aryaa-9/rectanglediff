import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

# Simulation input
L=1
N=2500

# Define the grid
x = np.linspace(-L, L, N)
xx, yy = np.meshgrid(x, x)

# Initialize the screen
E = np.zeros((N, N))

# Parameters for slit
mm = 1e-3
lx = 100 * mm
ly = 50 * mm

# Create the rectangular slit at the center
slit_mask = ((xx > (-lx / 2)) & (xx < (lx / 2)) & 
             (yy > (-ly / 2)) & (yy < (ly / 2)))
E[slit_mask] = 1


# Distance from slit to the screen (mm)
z = 5000

# Wavelength (mm)
λ = 18e-7
k = 2 * np.pi / λ

# Compute the diffraction pattern using Fourier transform
fft_c = fft2(E * np.exp(1j * k / (2 * z) * (xx**2 + yy**2)))
c = fftshift(fft_c)
abs_c = np.abs(c)

# Apply gamma correction to enhance the contrast
gamma = 0.65  # Value less than 1 to increase the contrast
abs_c_gamma = abs_c ** gamma

# Screen size (mm)
dx_screen = z * λ / (2 * L)
x_screen = dx_screen * (np.arange(N) - N // 2)

# Plot the results
plt.imshow(abs_c_gamma, extent=[x_screen[0], x_screen[-1], x_screen[0], x_screen[-1]], 
           cmap='inferno')
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.xlim([-1, 1])
plt.ylim([-1, 1])

"""
# Increase the dimensions when saving as PNG
plt.savefig('Q5Diff1.png', dpi=300)  # Adjust dpi value as needed
plt.show()
"""

fig = plt.figure()
ax2 = fig.add_subplot()  
ax2.plot(x_screen, abs_c[N//2]**2, linewidth = 1)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("I/I0")
ax2.set_xlim([-1,1])

"""
# Increase the dimensions when saving as PNG
plt.savefig('Q5Diff2.png', dpi=300)  # Adjust dpi value as needed
plt.show()
"""
