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

delta=wc-wa #detuning
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
ampl_list= linspace(0,8*2*pi,200) #amplitude list
#ampl_list= linspace(0,8*2*pi,50) #amplitude list
rabi_angle=ampl_list*2*pi/4

#initialization for the outputs
n_c_a= zeros(ampl_list.shape)
sz_a= zeros(ampl_list.shape)
sy_a= zeros(ampl_list.shape)
sx_a= zeros(ampl_list.shape)
voltage_a=zeros(ampl_list.shape) #(adag+a)
current_a=zeros(ampl_list.shape) #i(adag-a)

n_c= zeros((ampl_list.size,tlist.size))
n_a= zeros((ampl_list.size,tlist.size))
voltage= zeros((ampl_list.size,tlist.size))
current= zeros((ampl_list.size,tlist.size))
qsz= zeros((ampl_list.size,tlist.size))
qsy= zeros((ampl_list.size,tlist.size))
qsx= zeros((ampl_list.size,tlist.size))
pulse= zeros((ampl_list.size,tlist.size))

index=0

for ampl in ampl_list:
        H5=ampl*sx
        H=[H0,[H5, H5_coeff]] #total time dependent Hamiltonian
        output = mesolve(H, psi0, tlist, c_ops, [a.dag() * a, sm.dag() * sm, (a.dag() + a), 1j*(a.dag() - a), sz, sy, sx ]) #Getting the time evolution by solving the master equation
        n_c[index, :] = output.expect[0]
        n_a[index, :] = output.expect[1]
        voltage[index, :]= output.expect[2]
        current[index, :]= output.expect[3]
        qsz[index, :]= output.expect[4]
        qsy[index, :]= output.expect[5]
        qsx[index, :]= output.expect[6]
        tlist_at= tlist[300:]
        n_c_at=n_c[index, 300:]
        qsz_at=qsz[index,200]
        qsy_at=qsy[index,200]
        qsx_at=qsx[index,200]
        tlist_at1=tlist[145:1041]
        current_at1= current[index, 145:1041]
        voltage_at1= voltage[index, 145:1041]
        n_c_a[index]=trapz(n_c_at,tlist_at)
        n_c_cum=dt*cumsum(n_c_at)
        n_c_a[index]= n_c_cum[-1]
        #n_c_a[index]=0.38*(qsz_at+1)/2  #for verification with result from the paper, adding the photon collection efficiency 
        current_a[index]=trapz(current_at1,tlist_at1)
        voltage_a[index]=trapz(voltage_at1,tlist_at1)
        #current_a[index]= qsy_at*0.12 # for verification with result from the paper, homodyne voltage collection efficiency
        #voltage_a[index]= qsx_at*0.12
        sz_a[index]=qsz_at
        sy_a[index]=qsy_at
        sx_a[index]=qsx_at
        pulse[index, :]=ampl*H5_coeff(tlist,[])
        index=index+1


#plotting the verification for single photon source
fig, axes1 = plt.subplots(1, 1, figsize=(10,6))

axes1.plot(rabi_angle/2/pi, ampl_list/12/sqrt(2*pi),'b-')

axes1.legend(loc=0)
axes1.set_xlabel('rabi angle in 2pi')
axes1.set_ylabel('control amplitude in V0',color='b')
plt.show()


fig, axes1 = plt.subplots(1, 1, figsize=(10,6))

axes1.plot(rabi_angle/2/pi, ampl_list/12/sqrt(2*pi), label='control amplitude')
axes1.plot(rabi_angle/2/pi, sz_a, label='spontaneous emission')
axes1.legend(loc=0)
axes1.set_xlabel('rabi angle in 2pi')

plt.show()


fig, axes1 = plt.subplots(1, 1, figsize=(10,6))

axes1.plot(rabi_angle/2/pi, n_c_a,'b-')

axes1.legend(loc=0)
axes1.set_xlabel('rabi angle in 2pi')
axes1.set_ylabel('<adag.a>',color='b')

axes1.set_ylim(0, 0.4)
for tl in axes1.get_yticklabels():
    tl.set_color('b')
    
axes2 = axes1.twinx()
axes2.plot(rabi_angle/2/pi, sz_a,'r')
axes2.set_ylabel('<sigmaz>',color='r')
axes2.set_ylim(-1, 1)
for tl in axes2.get_yticklabels():
    tl.set_color('r')
plt.show()


fig, axes1 = plt.subplots(1, 1, figsize=(10,6))

axes1.plot(rabi_angle/2/pi, current_a,'b-')

axes1.legend(loc=0)
axes1.set_xlabel('rabi angle in 2pi')
axes1.set_ylabel('i<adag-a>',color='b')
axes1.set_ylim(-0.2, 0.2)
for tl in axes1.get_yticklabels():
    tl.set_color('b')
    
axes2 = axes1.twinx()
axes2.plot(rabi_angle/2/pi, sy_a,'r')
axes2.set_ylabel('<sigmay>',color='r')
axes2.set_ylim(-1, 1)
for tl in axes2.get_yticklabels():
    tl.set_color('r')
plt.show()

fig, axes1 = plt.subplots(1, 1, figsize=(10,6))

axes1.plot(rabi_angle/2/pi, voltage_a,'b-')

axes1.legend(loc=0)
axes1.set_xlabel('rabi angle in 2pi')
axes1.set_ylabel('<adag+a>',color='b')
axes1.set_ylim(-0.2, 0.2)
for tl in axes1.get_yticklabels():
    tl.set_color('b')
    
axes2 = axes1.twinx()
axes2.plot(rabi_angle/2/pi, sx_a,'r')
axes2.set_ylabel('<sigmax>',color='r')
axes2.set_ylim(-1, 1)
for tl in axes2.get_yticklabels():
    tl.set_color('r')
plt.show()


#Plotting the time evolution
origin = 'lower'

        
##fig = plt.figure()
##ax = fig.gca(projection='3d')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
tlist, rabi_angle= meshgrid(tlist, rabi_angle)
n_c=reshape(n_c, tlist.shape)
#surf = ax.plot_surface(tlist, ampl_list, n_c, cmap=cm.jet, linewidth=0, antialiased=False)

cset = ax.contourf(tlist, rabi_angle/2/pi, n_c, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
ax.set_title('<adag.a>')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
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
n_a=reshape(n_a, tlist.shape)
#surf = ax.plot_surface(tlist, ampl_list, n_a, cmap=cm.jet, linewidth=0, antialiased=False)
cset = ax.contourf(tlist, rabi_angle/2/pi, n_a, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
ax.set_title('<sigma+.sigma->')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
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
qsz=reshape(qsz, tlist.shape)
cset = ax.contourf(tlist, rabi_angle/2/pi, qsz, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsz, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<sigmaz>')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
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
qsy=reshape(qsy, tlist.shape)
cset = ax.contourf(tlist, rabi_angle/2/pi, qsy, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsy, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<sigmay>')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
qsx=reshape(qsx, tlist.shape)
cset = ax.contourf(tlist, rabi_angle/2/pi, qsx, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, qsy, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<sigmax>')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
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
voltage=reshape(voltage, tlist.shape)
cset = ax.contourf(tlist, rabi_angle/2/pi, voltage, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, voltage, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<adag+a>')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
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
current=reshape(current, tlist.shape)
cset = ax.contourf(tlist, rabi_angle/2/pi, current, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
#surf = ax.plot_surface(tlist, ampl_list, current, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_title('<adag-a>')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
pulse=reshape(pulse, tlist.shape)
#surf = ax.plot_surface(tlist, ampl_list, n_a, cmap=cm.jet, linewidth=0, antialiased=False)
cset = ax.contourf(tlist, rabi_angle/2/pi, pulse, 100, cmap=cm.jet, linewidth=.5, origin=origin, extend='both')
ax.set_title('Drive pulse in 2 uV')
ax.set_xlabel('time (ns)')
ax.set_ylabel('rabi angle in 2pi')
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##fig.colorbar(surf, shrink=0.5, aspect=5)
cset.cmap.set_under('blue')
cset.cmap.set_over('red')
plt.colorbar(cset)
plt.show()
