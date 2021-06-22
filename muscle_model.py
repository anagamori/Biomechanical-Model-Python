# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 21:42:53 2021

@author: anaga
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import muscle_model_functions as muscle_fns
import parameter_muscle_1 as m1
plt.rcParams['pdf.fonttype'] = 42


Fs = 10000 # sampling frequency 
step = 1/float(Fs) # time step
time_sim = np.arange(0,3,step) # time vector

# Define input to muscle 1 and 2
U_1 = np.zeros(len(time_sim),dtype = "float") # pulse input to muscle 1
U_1[int(0.5*Fs):int(3*Fs)] = 1

# Muscle model
# muscle 1
(Lce_1,Lse_1,Lmt_1) =  muscle_fns.InitialLength(m1)
U_eff_1 = 0.0;
f_int_slow_1 = 0.0;
f_eff_slow_1 = 0.0;
f_eff_slow_dot_1 = 0.0;
f_int_fast_1 = 0.0;
f_eff_fast_1 = 0.0;
f_eff_fast_dot_1 = 0.0;
Af_slow_1 = 0.0;
Af_fast_1 = 0.0;
Y_1 = 0.0;
S_1 = 0.0;    
Vce_1 = 0.0;
Force_tendon_1 = muscle_fns.F_se_function(Lse_1)*m1.F0; # initial tendon force
muscle_length_1 = Lce_1*(m1.L0/float(100)) # non-normalized muscle length 
muscle_velocity_1 = 0 # non-normalized muscle velocity

x_1 = np.array([[Lce_1*m1.L0/float(100)],[0.0]])

# vectors for data storage
U_eff_1_vec = np.zeros(len(time_sim))
Force_tendon_1_vec = np.zeros(len(time_sim))
Lce_1_vec = np.zeros(len(time_sim))
Vce_1_vec = np.zeros(len(time_sim))
Lse_1_vec = np.zeros(len(time_sim))
Lmt_1_vec = np.zeros(len(time_sim))

# Start for-loop for simulation
start_time = time.time()
for t in range(len(time_sim)):
    # Muscle 1
    U_eff_1 = muscle_fns.U_function(U_1[t],U_eff_1,step)      
    
    (W_slow_1,W_fast_1) = muscle_fns.weighting_function(U_eff_1,m1.U_slow_th,m1.U_fast_th)
    
    f_env_slow_1 = muscle_fns.U2f_slow_function(U_eff_1,m1.U_slow_th)        
    (f_int_slow_1,f_int_slow_dot_1) = muscle_fns.f_slow_function(f_int_slow_1,f_env_slow_1,f_env_slow_1,f_eff_slow_dot_1,Af_slow_1,Lce_1,step);
    (f_eff_slow_1,f_eff_slow_dot_1) = muscle_fns.f_slow_function(f_eff_slow_1,f_int_slow_1,f_env_slow_1,f_eff_slow_dot_1,Af_slow_1,Lce_1,step);
    
    f_env_fast_1 = muscle_fns.U2f_fast_function(U_eff_1,m1.U_fast_th)
    (f_int_fast_1,f_int_fast_dot_1) = muscle_fns.f_fast_function(f_int_fast_1,f_env_fast_1,f_env_fast_1,f_eff_fast_dot_1,Af_fast_1,Lce_1,step);
    (f_eff_fast_1,f_eff_fast_dot_1) = muscle_fns.f_fast_function(f_eff_fast_1,f_int_fast_1,f_env_fast_1,f_eff_fast_dot_1,Af_fast_1,Lce_1,step);
            
    Y_1 = muscle_fns.yield_function(Y_1,Vce_1,step)
    S_1 = muscle_fns.sag_function(S_1,f_eff_fast_1,step)    
    
    # Integration to get muscle length and velocity
    km_1_1 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1,m1)
    km_1_2 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1+km_1_1/2,m1);
    km_1_3 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1+km_1_2/2,m1);
    km_1_4 = step*muscle_fns.contraction_dynamics(U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,Lse_1,x_1+km_1_3,m1);
    x_1 = x_1 + (km_1_1 + 2*km_1_2 + 2*km_1_3 + km_1_4)/6;
    
    Vce_1 = x_1[1]/float(m1.L0/100)
    Lce_1 = x_1[0]/float(m1.L0/100)
    Lse_1 = (Lmt_1 - Lce_1*m1.L0*np.cos(m1.alpha))/float(m1.L0T) # Eq. 16
    
    Force_tendon_1 = muscle_fns.F_se_function(Lse_1)*m1.F0;    

    Force_tendon_1_vec[t] = Force_tendon_1;
    Lce_1_vec[t] = Lce_1
    Vce_1_vec[t] = Vce_1
    Lmt_1_vec[t] = Lmt_1  
    Lse_1_vec[t] = Lse_1  

end_time = time.time()
print(end_time - start_time)

# Plot data
fig = plt.figure()
ax1 = plt.subplot(3,1,1)
ax1.plot(time_sim,U_1)
ax1.set_ylabel('Input')
ax2 = plt.subplot(3,1,2)
ax2.plot(time_sim,Force_tendon_1_vec)
ax2.set_ylabel('Muscle Force \n(N)')
plt.show()

fig = plt.figure()
ax4 = plt.subplot(3,1,1)
ax4.plot(time_sim,Lce_1_vec)
ax4.set_xlabel('Time (sec)')
ax4.set_ylabel('Muscle Length \n(L0)')
ax5 = plt.subplot(3,1,2)
ax5.plot(time_sim,Vce_1_vec)
ax5.set_xlabel('Time (sec)')
ax5.set_ylabel('Muscle Velocity \n(L0)')

 