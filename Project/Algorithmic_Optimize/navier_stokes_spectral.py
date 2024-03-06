import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def main(N = 400, t = 0, tEnd = 1, dt = 0.001, tOut = 0.01, nu = 0.001, plotRealTime = False):
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
	
	outputCount = 1
	kx = 1j*kx
	ky = 1j*ky
	r = (1.0+dt*nu*kSq)
	#Main Loop
	for i in range(Nt):
		xx = np.fft.fftn(vx)
		yy = np.fft.fftn(vy)
		rhs_x = vx * np.real(np.fft.ifftn( kx * xx)) + vy * np.real(np.fft.ifftn( ky * xx))
		rhs_y = vx * np.real(np.fft.ifftn( kx * yy)) + vy * np.real(np.fft.ifftn( ky * yy))
		rhs_x = np.real(np.fft.ifftn(dealias * np.fft.fftn(-rhs_x)))
		rhs_y = np.real(np.fft.ifftn(dealias * np.fft.fftn(-rhs_y)))
		div_rhs = np.real(np.fft.ifftn( kx * np.fft.fftn(rhs_x))) + np.real(np.fft.ifftn( ky * np.fft.fftn(rhs_y)))
		P = np.real(np.fft.ifftn(-(np.fft.fftn(div_rhs)) * kSq_inv))
		dPx = np.real(np.fft.ifftn( kx * P))
		dPy = np.real(np.fft.ifftn( ky * P))
		vy += dt * (rhs_y - dPy)
		vx += dt * (rhs_x - dPx)
		xx = np.fft.fftn(vx)
		yy = np.fft.fftn(vy)
		vx = np.real(np.fft.ifftn((xx) / r))
		vy = np.real(np.fft.ifftn((yy) / r))
		wz = np.real(np.fft.ifftn( kx * yy) - np.fft.ifftn( ky * xx))

		# update time
		t += dt
		
		# plot in real time
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			plotThisTurn = True
		if (plotRealTime and plotThisTurn) or (i == Nt-1):
			outputCount += 1
			
	return 0
	


if __name__== "__main__":
  main()
