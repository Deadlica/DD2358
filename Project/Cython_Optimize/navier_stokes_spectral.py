import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import navier_stokes_spectral as NSS
"""
Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""

def main(N = 400, t = 0, tEnd = 1, dt = 0.001, tOut = 0.01, nu = 0.001, plotRealTime = False):
	""" Navier-Stokes Simulation """
	# Simulation parameters
	# N            = Spatial resolution
	# t            = Current time of the simulation
	# tEnd         = Time at which simulation ends
	# dt           = Timestep
	# tOut         = Draw frequency
	# nu           = Viscosity
	# plotRealTime = Switch on for plotting as the simulation goes along
	
	# Domain [0,1] x [0,1]
	L = 1    
	xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
	xlin = xlin[0:N]                  # chop off periodic point
	xx, yy = np.meshgrid(xlin, xlin)
	
	# Intial Condition (vortex)
	vx = -np.sin(2*np.pi*yy)
	vy =  np.sin(2*np.pi*xx*2) 
	
	# Fourier Space Variables
	klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
	kmax = np.max(klin)
	kx, ky = np.meshgrid(klin, klin)
	kx = np.fft.ifftshift(kx)
	ky = np.fft.ifftshift(ky)
	kSq = kx**2 + ky**2
	kSq_inv = 1.0 / kSq
	kSq_inv[kSq==0] = 1
	
	# dealias with the 2/3 rule
	dealias = (np.abs(kx) < (2./3.)*kmax) & (np.abs(ky) < (2./3.)*kmax)
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	#fig = plt.figure(figsize=(4,4), dpi=80)
	outputCount = 1
	
	#Main Loop
	for i in range(Nt):

		# Advection: rhs = -(v.grad)v
		dvx_x, dvx_y = NSS.grad(vx, kx, ky)
		dvy_x, dvy_y = NSS.grad(vy, kx, ky)
		
		rhs_x = -(vx * dvx_x + vy * dvx_y)
		rhs_y = -(vx * dvy_x + vy * dvy_y)
		
		rhs_x = NSS.apply_dealias(rhs_x, dealias)
		rhs_y = NSS.apply_dealias(rhs_y, dealias)

		vx += dt * rhs_x
		vy += dt * rhs_y
		
		# Poisson solve for pressure
		div_rhs = NSS.div(rhs_x, rhs_y, kx, ky)
		P = NSS.poisson_solve( div_rhs, kSq_inv )
		dPx, dPy = NSS.grad(P, kx, ky)
		
		# Correction (to eliminate divergence component of velocity)
		vx += - dt * dPx
		vy += - dt * dPy
		
		# Diffusion solve (implicit)
		vx = NSS.diffusion_solve( vx, dt, nu, kSq )
		vy = NSS.diffusion_solve( vy, dt, nu, kSq )
		
		# vorticity (for plotting)
		wz = NSS.curl(vx, vy, kx, ky)
		
		# update time
		t += dt
		#print(t)
		
		# plot in real time
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			plotThisTurn = True
		if (plotRealTime and plotThisTurn) or (i == Nt-1):
			
			#plt.cla()
			#plt.imshow(wz, cmap = "RdBu")
			#plt.clim(-20,20)
			#ax = plt.gca()
			#ax.invert_yaxis()
			#ax.get_xaxis().set_visible(False)
			#ax.get_yaxis().set_visible(False)	
			#ax.set_aspect("equal")	
			#plt.pause(0.001)
			outputCount += 1
			
			
	# Save figure
	#plt.savefig("navier-stokes-spectral.png",dpi=240)
	#plt.show()
	
	return 0
	


if __name__== "__main__":
  main()
