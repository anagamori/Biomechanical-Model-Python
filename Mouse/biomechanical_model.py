# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:14:08 2021

@author: anaga
"""

"""
biomechanical_model.py
Author: Akira Nagamori
Last update: 6/25/21
Descriptions: 
    a biomechanical model of the human elbow joint controlled by a pair of antagonistic muscles 
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import muscle_model_functions as muscle_fns
import parameter_muscle_1 as m1
import parameter_muscle_1 as m2 # using identical muscles here for now
plt.rcParams['pdf.fonttype'] = 42


Fs = 1000000 # sampling frequency 
step = 1/float(Fs) # time step
time_sim = np.arange(0,3,step) # time vector

# Define input to muscle 1 and 2
U_1 = np.zeros(len(time_sim),dtype = "float") # pulse input to muscle 1
U_1[int(0.5*Fs):int(3*Fs)] = 0.1 # amplitude of neural activation to muscle 1 (0-1)
U_2 = np.zeros(len(time_sim),dtype = "float") # pulse input to muscle 2
U_2[int(0.7*Fs):int(1*Fs)] = 0.0 # amplitude of neural activation to muscle 2 (0-1)

# limb segment modeled as a cylinder with the length of 1.2 cm and the diameter of 5 mm
# the proximal segment attaches 5 cm from the end of the distal segment   
l = 0.012 # segment length [m]
r = 0.005/2 # segment radius [m] 
d = 0.003 # distance from the end to the attachment site [m]
d_c = 1/2 - d/l # distance from the center of the distal segment to the joint center
M = 0.01 # segment mass [kg]
I = 1/12*l**2*M + (d_c*l)**2*M + 1/4*M*r**2 # segment inertia
b = 0 # external viscosity
k = 0 # external stiffness
A = np.array([[0.0,1.0],[0.0,0.0]]) # state matrix for the limb
B = np.array([[0.0],[1.0/I]]) # input matrix for the limb

r_m1 = 0.002 # moment arm of muscle 1 [m]
r_m2 = -0.002 # moment arm of muscle 2 [m]

## Initialization of parameters 
# Kinematic model
theta = 0 # joint angle 
theta_dot = 0 # joint angular velocity
x = np.array([[0.0],[0.0]]) # state vector (x[0] = joint angle, x[1] = joint angular velocity)

# Muscle model
# muscle 1
(Lce_1,Lse_1,Lmt_1) =  muscle_fns.InitialLength(m1)
Force_tendon_1 = muscle_fns.F_se_function(Lse_1)*m1.F0; # initial tendon force (N)
U_eff_1 = 0.0; # effective neural activation
f_int_slow_1 = 0.0;
f_eff_slow_1 = 0.0;
f_eff_slow_dot_1 = 0.0;
f_int_fast_1 = 0.0;
f_eff_fast_1 = 0.0;
f_eff_fast_dot_1 = 0.0;
Af_slow_1 = 0.0; # activation-frequency relationship of slow-twitch unit
Af_fast_1 = 0.0; # activation-frequency relationship of fast-twitch unit
Y_1 = 0.0; # yielding 
S_1 = 0.0; # sag 
Vce_1 = 0.0; # normalized muscle velocity
x_1 = np.array([[Lce_1*m1.L0/float(100)],[0.0]]) # state vector for muscle 1 (x_1[0] = muscle length (cm), x_1[1] = muscle velocity(cm/s))

# muscle 2
(Lce_2,Lse_2,Lmt_2) =  muscle_fns.InitialLength(m2)
U_eff_2 = 0.0;
f_int_slow_2 = 0.0;
f_eff_slow_2 = 0.0;
f_eff_slow_dot_2 = 0.0;
f_int_fast_2 = 0.0;
f_eff_fast_2 = 0.0;
f_eff_fast_dot_2 = 0.0;
Af_slow_2 = 0.0;
Af_fast_2 = 0.0;
Y_2 = 0.0;
S_2 = 0.0;    
Vce_2 = 0.0;
Force_tendon_2 = muscle_fns.F_se_function(Lse_2)*m2.F0; # initial tendon force

x_2 = np.array([[Lce_2*m2.L0/float(100)],[0.0]]) # state vector for muscle 2 (x_1[0] = muscle length, x_1[1] = muscle velocity)

# vectors for data storage
theta_vec = np.zeros(len(time_sim),dtype = "float") 
theta_dot_vec = np.zeros(len(time_sim),dtype = "float")
T_vec = np.zeros(len(time_sim),dtype = "float") # total torque from two muscles

U_eff_1_vec = np.zeros(len(time_sim))
Force_tendon_1_vec = np.zeros(len(time_sim))
Lce_1_vec = np.zeros(len(time_sim))
Vce_1_vec = np.zeros(len(time_sim))
Lse_1_vec = np.zeros(len(time_sim))
Lmt_1_vec = np.zeros(len(time_sim))

U_eff_2_vec = np.zeros(len(time_sim))
Force_tendon_2_vec = np.zeros(len(time_sim))
Lce_2_vec = np.zeros(len(time_sim))
Vce_2_vec = np.zeros(len(time_sim))
Lse_2_vec = np.zeros(len(time_sim))
Lmt_2_vec = np.zeros(len(time_sim))

# Start for-loop for simulation
start_time = time.time()
for t in range(len(time_sim)):
    # Muscle 1
    # Effective neural activation
    U_eff_1 = muscle_fns.U_function(U_1[t],U_eff_1,step)      
    
    # Compute weighting on each fiber type 
    (W_slow_1,W_fast_1) = muscle_fns.weighting_function(U_eff_1,m1.U_slow_th,m1.U_fast_th)
    
    # Firing frequency of each unit
    f_env_slow_1 = muscle_fns.U2f_slow_function(U_eff_1,m1.U_slow_th)        
    (f_int_slow_1,f_int_slow_dot_1) = muscle_fns.f_slow_function(f_int_slow_1,f_env_slow_1,f_env_slow_1,f_eff_slow_dot_1,Af_slow_1,Lce_1,step);
    (f_eff_slow_1,f_eff_slow_dot_1) = muscle_fns.f_slow_function(f_eff_slow_1,f_int_slow_1,f_env_slow_1,f_eff_slow_dot_1,Af_slow_1,Lce_1,step);
    
    f_env_fast_1 = muscle_fns.U2f_fast_function(U_eff_1,m1.U_fast_th)
    (f_int_fast_1,f_int_fast_dot_1) = muscle_fns.f_fast_function(f_int_fast_1,f_env_fast_1,f_env_fast_1,f_eff_fast_dot_1,Af_fast_1,Lce_1,step);
    (f_eff_fast_1,f_eff_fast_dot_1) = muscle_fns.f_fast_function(f_eff_fast_1,f_int_fast_1,f_env_fast_1,f_eff_fast_dot_1,Af_fast_1,Lce_1,step);
            
    # Yielding
    Y_1 = muscle_fns.yield_function(Y_1,Vce_1,step)
    # Sag
    S_1 = muscle_fns.sag_function(S_1,f_eff_fast_1,step)    
    
    # Integration to get muscle length (cm) and velocity (cm/s)
    km_1_1 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1,m1)
    km_1_2 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1+km_1_1/2,m1);
    km_1_3 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1+km_1_2/2,m1);
    km_1_4 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1+km_1_3,m1);
    x_1 = x_1 + (km_1_1 + 2*km_1_2 + 2*km_1_3 + km_1_4)/6;
    
    Vce_1 = x_1[1]/(m1.L0/100) # normalized muscle velocity (L0/s)
    Lce_1 = x_1[0]/(m1.L0/100) # normalized muscle length (L0)
    Lse_1 = (Lmt_1 - Lce_1*m1.L0*np.cos(m1.alpha))/(m1.L0T) # normalized length of the series-elastic element
      
    Force_tendon_1 = muscle_fns.F_se_function(Lse_1)*m1.F0; # tendon force
    T_1 = Force_tendon_1*r_m1 # torque from muscle 1
    
    # Muscle 2
    U_eff_2 = muscle_fns.U_function(U_2[t],U_eff_2,step)  
    
    (W_slow_2,W_fast_2) = muscle_fns.weighting_function(U_eff_2,m2.U_slow_th,m2.U_fast_th)
    
    f_env_slow_2 = muscle_fns.U2f_slow_function(U_eff_2,m2.U_slow_th)        
    (f_int_slow_2,f_int_slow_dot_2) = muscle_fns.f_slow_function(f_int_slow_2,f_env_slow_2,f_env_slow_2,f_eff_slow_dot_2,Af_slow_2,Lce_2,step);
    (f_eff_slow_2,f_eff_slow_dot_2) = muscle_fns.f_slow_function(f_eff_slow_2,f_int_slow_2,f_env_slow_2,f_eff_slow_dot_2,Af_slow_2,Lce_2,step);
    
    f_env_fast_2 = muscle_fns.U2f_fast_function(U_eff_2,m2.U_fast_th)
    (f_int_fast_2,f_int_fast_dot_2) = muscle_fns.f_fast_function(f_int_fast_2,f_env_fast_2,f_env_fast_2,f_eff_fast_dot_2,Af_fast_2,Lce_2,step);
    (f_eff_fast_2,f_eff_fast_dot_2) = muscle_fns.f_fast_function(f_eff_fast_2,f_int_fast_2,f_env_fast_2,f_eff_fast_dot_2,Af_fast_2,Lce_2,step);
            
    Y_2 = muscle_fns.yield_function(Y_2,Vce_2,step)
    S_2 = muscle_fns.sag_function(S_2,f_eff_fast_2,step)    
    
    # Integration to get muscle length and velocity
    km_2_1 = step*muscle_fns.contraction_dynamics(U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,Lse_2,x_2,m2)
    km_2_2 = step*muscle_fns.contraction_dynamics(U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,Lse_2,x_2+km_2_1/2,m2);
    km_2_3 = step*muscle_fns.contraction_dynamics(U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,Lse_2,x_2+km_2_2/2,m2);
    km_2_4 = step*muscle_fns.contraction_dynamics(U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,Lse_2,x_2+km_2_3,m2);
    x_2 = x_2 + (km_2_1 + 2*km_2_2 + 2*km_2_3 + km_2_4)/6;
    
    Vce_2 = x_2[1]/(m2.L0/100)
    Lce_2 = x_2[0]/(m2.L0/100)
    Lse_2 = (Lmt_2 - Lce_2*m2.L0*np.cos(m2.alpha))/(m2.L0T)
      
    
    Force_tendon_2 = muscle_fns.F_se_function(Lse_2)*m2.F0;  
    T_2 = Force_tendon_2*r_m2 # torque from muscle 2
    
    # Kinematic model
    T = T_1 + T_2; # total torque acting on the joint
    # Integration of the second order dynamics in Eq. 21 
    k1 = np.dot(A,x) + B*T 
    x1 = x + k1*step/2;
    k2 = np.dot(A,x1) + B*T;
    x = x + k2*step; # state vector of the joint
    
    theta = x[0] # joint angle 
    theta_dot = x[1] # joint angular velocity
    
    # Update the musculotendon length (Eq. 22)
    if t > 0:
        Lmt_1 = Lmt_1 - r_m1*100*(theta-theta_vec[t-1])
        Lmt_2 = Lmt_2 - r_m2*100*(theta-theta_vec[t-1])
    
    # save variables 
    T_vec[t] = T
    theta_vec[t] = theta
    theta_dot_vec[t] = theta_dot
    
    U_eff_1_vec[t] = U_eff_1
    Force_tendon_1_vec[t] = Force_tendon_1;
    Lce_1_vec[t] = Lce_1 
    Vce_1_vec[t] = Vce_1 
    Lmt_1_vec[t] = Lmt_1
    Lse_1_vec[t] = Lse_1
    
    U_eff_2_vec[t] = U_eff_2 
    Force_tendon_2_vec[t] = Force_tendon_2;
    Lce_2_vec[t] = Lce_2  
    Vce_2_vec[t] = Vce_2 
    Lmt_2_vec[t] = Lmt_2        
    Lse_2_vec[t] = Lse_2    

end_time = time.time()
print(end_time - start_time)

# Plot data
fig = plt.figure()
ax1 = plt.subplot(3,1,1)
ax1.plot(time_sim,U_1,time_sim,U_2)
ax1.set_ylabel('Input')
ax1.legend(['m1','m2'])
ax2 = plt.subplot(3,1,2)
ax2.plot(time_sim,T_vec)
ax2.set_xlim([0,np.max(time_sim)])
ax2.set_ylabel('Joint Troque\n(N)')
ax3 = plt.subplot(3,1,3)
ax3.plot(time_sim,np.degrees(theta_vec))
ax3.set_xlim([0,np.max(time_sim)])
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('Joint Angle\n(degrees)')
plt.show()
#fig.savefig("Output.pdf", bbox_inches='tight',transparent=True)

fig = plt.figure()
ax4 = plt.subplot(3,1,1)
ax4.plot(time_sim,Lce_1_vec,time_sim,Lce_2_vec)
ax4.set_xlim([0,np.max(time_sim)])
ax4.legend(['m1','m2'])
ax4.set_xlabel('Time (sec)')
ax4.set_ylabel('Muscle Length \n(L0)')
ax5 = plt.subplot(3,1,2)
ax5.plot(time_sim,Vce_1_vec,time_sim,Vce_2_vec)
ax5.set_xlim([0,np.max(time_sim)])
ax5.legend(['m1','m2'])
ax5.set_xlabel('Time (sec)')
ax5.set_ylabel('Muscle Velocity \n(L0/s)')
ax6 = plt.subplot(3,1,3)
ax6.plot(time_sim,Force_tendon_1_vec,time_sim,Force_tendon_2_vec)
ax6.set_xlim([0,np.max(time_sim)])
ax6.set_xlabel('Time (sec)')
ax6.set_ylabel('Tendon Force \n(N)')

 