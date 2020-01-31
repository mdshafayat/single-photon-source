import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import *
from qutip import *

#Defining area normalized Gaussian pulse
def H5_coeff(t, args):
        return  exp(-0.5* square(t/12))/12/sqrt(2*pi)

wc = 5.19  * 2 * pi  # cavity frequency in GHz
wa = 4.68  * 2 * pi  # qubit frequency in GHz
g  = 0.107 * 2 * pi  # coupling strength in GHz

kappa = 0.044* 2 * pi       # cavity dissipation rate in GHz

delta=wc-wa  #detuning
gamma = square(g/delta) * kappa       # qubit dissipation rate in GHz


N = 2            # number of cavity fock states
n_th_a = 0.0        # avg number of thermal bath excitation

tlist = linspace(-24,236,1041) #time interval in ns
dt=tlist[1]-tlist[0]

# intial state
#psi0 = tensor(coherent(N, sqrt(4)), (basis(2,0)+basis(2,0)).unit())
#psi0 = tensor(basis(N,0), basis(2,0))    # start with a ground qubit
psi0 = tensor(basis(N,0), (basis(2,0)+basis(2,0)).unit()) 
#psi0 = tensor(basis(N,0), basis(2,1))    # start with a excited qubit

# operators
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
sz = tensor(qeye(N), sigmaz())
sx = tensor(qeye(N), sigmax())
sy = tensor(qeye(N), sigmay())
I = tensor(qeye(N), qeye(2))

# Hamiltonian

    H0 = (wc-wa) * (a.dag() * a ) + ((wa-wa) / 2.0) * sz + g * (a.dag() * sm + a * sm.dag())

#Formulating collapse operators for Lindblad Master equation
c_ops = []

# cavity relaxation
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(sqrt(rate) * a)

# cavity excitation, if temperature > 0
rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(sqrt(rate) * a.dag())

# qubit relaxation
rate = gamma
if rate > 0.0:
    c_ops.append(sqrt(rate) * sm)
    
##ampl_list= linspace(4*2*pi,8*2*pi,100) #amplitude list
#ampl_list= linspace(0,8*2*pi,200) #amplitude list
ampl_list= linspace(-4*pi,4*pi,36) #amplitude list
rabi_angle=ampl_list*2*pi/4

phase_list= linspace(-4*pi,4*pi,36) #phase list

#initialization for the outputs

sz_a= zeros((phase_list.size, ampl_list.size))
sy_a= zeros((phase_list.size, ampl_list.size))
sx_a= zeros((phase_list.size, ampl_list.size))
voltage= zeros((phase_list.size, ampl_list.size))
current= zeros((phase_list.size, ampl_list.size))


qsz= zeros((ampl_list.size, tlist.size))
qsy= zeros((ampl_list.size, tlist.size))
qsx= zeros((ampl_list.size, tlist.size))
voltage_t= zeros((ampl_list.size, tlist.size)) #(adag+a)
current_t= zeros((ampl_list.size, tlist.size)) #i(adag-a)

index2=0

for phase in phase_list:
        index1=0
        for ampl in ampl_list:
                H5=ampl*sx+phase*sy
                H=[H0,[H5, H5_coeff]] #total time dependent Hamiltonian
                output = mesolve(H, psi0, tlist, c_ops, [ sz, sy, sx (a.dag() + a), 1j*(a.dag() - a)]) #Getting the time evolution by solving the master equation
                
                qsz[index1, :]= output.expect[0]
                qsy[index1, :]= output.expect[1]
                qsx[index1, :]= output.expect[2]
                voltage_t[index1, :]= output.expect[3]
                current_t[index1, :]= output.expect[4]
                
                qsz_at=qsz[index1,145]
                qsy_at=qsy[index1,145]
                qsx_at=qsx[index1,145]

                tlist_at1=tlist[145:1041]
                current_at1= current_t[index, 145:1041]
                voltage_at1= voltage_t[index, 145:1041]

                current[index2,index1]=trapz(current_at1,tlist_at1)
                voltage[index2,index1]=trapz(voltage_at1,tlist_at1)
                
                sz_a[index2,index1]=qsz_at
                sy_a[index2,index1]=qsy_at
                sx_a[index2,index1]=qsx_at
                current[index2,index1]=trapz(current_at1,tlist_at1)
                voltage[index2,index1]=trapz(voltage_at1,tlist_at1)

                index1=index1+1
        index2=index2+1


#Debug for matching the results from the paper using their collection efficiency
#voltage=sx_a*0.12
#current=sy_a*0.12

#Plotting the state tomography
origin = 'lower'

rabi_angle, phase_list= meshgrid(rabi_angle, phase_list)

##fig = plt.figure()
##ax = fig.gca(projection='3d')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sz_a=reshape(sz_a, rabi_angle.shape)
cset = ax.contourf(rabi_angle/2/pi, phase_list/2/pi, sz_a, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsz, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<sigmaz>')
ax.set_xlabel('X angle in 2pi')
ax.set_ylabel('Y angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()

##fig = plt.figure()
##ax = fig.gca(projection='3d')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sy_a=reshape(sy_a, rabi_angle.shape)
cset = ax.contourf(rabi_angle/2/pi, phase_list/2/pi, sy_a, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsy, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<sigmay>')
ax.set_xlabel('X angle in 2pi')
ax.set_ylabel('Y angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
sx_a=reshape(sx_a, rabi_angle.shape)
cset = ax.contourf(rabi_angle/2/pi, phase_list/2/pi, sx_a, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsy, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<sigmax>')
ax.set_xlabel('X angle in 2pi')
ax.set_ylabel('Y angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()

##fig = plt.figure()
##ax = fig.gca(projection='3d')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
current=reshape(current, rabi_angle.shape)
cset = ax.contourf(rabi_angle/2/pi, phase_list/2/pi, current, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsy, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('i<adag-a>')
ax.set_xlabel('X angle in 2pi')
ax.set_ylabel('Y angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
voltage=reshape(voltage, rabi_angle.shape)
cset = ax.contourf(rabi_angle/2/pi, phase_list/2/pi, voltage, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsy, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<adag+a>')
ax.set_xlabel('X angle in 2pi')
ax.set_ylabel('Y angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()
