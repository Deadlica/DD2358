import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cupy as cp
"""
Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


def poisson_solve( rho, kSq_inv ):
	""" solve the Poisson equation, given source field rho """
	V_hat = -(cp.fft.fftn( rho )) * kSq_inv
	V = cp.real(cp.fft.ifftn(V_hat))
	return V

def diffusion_solve( v, dt, nu, kSq ):
	""" solve the diffusion equation over a timestep dt, given viscosity nu """
	v_hat = (cp.fft.fftn( v )) / (1.0+dt*nu*kSq)
	v = cp.real(cp.fft.ifftn(v_hat))
	return v

def grad(v, kx, ky):
	""" return gradient of v """
	v_hat = cp.fft.fftn(v)
	dvx = cp.real(cp.fft.ifftn( 1j*kx * v_hat))
	dvy = cp.real(cp.fft.ifftn( 1j*ky * v_hat))
	return dvx, dvy

def div(vx, vy, kx, ky):
	""" return divergence of (vx,vy) """
	dvx_x = cp.real(cp.fft.ifftn( 1j*kx * cp.fft.fftn(vx)))
	dvy_y = cp.real(cp.fft.ifftn( 1j*ky * cp.fft.fftn(vy)))
	return dvx_x + dvy_y

def curl(vx, vy, kx, ky):
	""" return curl of (vx,vy) """
	dvx_y = cp.real(cp.fft.ifftn( 1j*ky * cp.fft.fftn(vx)))
	dvy_x = cp.real(cp.fft.ifftn( 1j*kx * cp.fft.fftn(vy)))
	return dvy_x - dvx_y

def apply_dealias(f, dealias):
	""" apply 2/3 rule dealias to field f """
	f_hat = dealias * cp.fft.fftn(f)
	return cp.real(cp.fft.ifftn( f_hat ))

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
	xlin = cp.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
	xlin = xlin[0:N]                  # chop off periodic point
	xx, yy = cp.meshgrid(xlin, xlin)
	
	# Intial Condition (vortex)
	vx = -cp.sin(2*cp.pi*yy)
	vy =  cp.sin(2*cp.pi*xx*2) 
	
	# Fourier Space Variables
	klin = 2.0 * cp.pi / L * cp.arange(-N/2, N/2)
	kmax = cp.max(klin)
	kx, ky = cp.meshgrid(klin, klin)
	kx = cp.fft.ifftshift(kx)
	ky = cp.fft.ifftshift(ky)
	kSq = kx**2 + ky**2
	kSq_inv = 1.0 / kSq
	kSq_inv[kSq==0] = 1
	
	# dealias with the 2/3 rule
	dealias = (cp.abs(kx) < (2./3.)*kmax) & (cp.abs(ky) < (2./3.)*kmax)
	
	# number of timesteps
	Nt = int(cp.ceil(tEnd/dt))
	
	# prep figure
	#fig = plt.figure(figsize=(4,4), dpi=80)
	outputCount = 1
	
	#Main Loop
	for i in range(Nt):

		# Advection: rhs = -(v.grad)v
		dvx_x, dvx_y = grad(vx, kx, ky)
		dvy_x, dvy_y = grad(vy, kx, ky)
		
		rhs_x = -(vx * dvx_x + vy * dvx_y)
		rhs_y = -(vx * dvy_x + vy * dvy_y)
		
		rhs_x = apply_dealias(rhs_x, dealias)
		rhs_y = apply_dealias(rhs_y, dealias)

		vx += dt * rhs_x
		vy += dt * rhs_y
		
		# Poisson solve for pressure
		div_rhs = div(rhs_x, rhs_y, kx, ky)
		P = poisson_solve( div_rhs, kSq_inv )
		dPx, dPy = grad(P, kx, ky)
		
		# Correction (to eliminate divergence component of velocity)
		vx += - dt * dPx
		vy += - dt * dPy
		
		# Diffusion solve (implicit)
		vx = diffusion_solve( vx, dt, nu, kSq )
		vy = diffusion_solve( vy, dt, nu, kSq )
		
		# vorticity (for plotting)
		wz = curl(vx, vy, kx, ky)
		
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
