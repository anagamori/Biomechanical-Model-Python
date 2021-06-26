"""
parameter_muscle_1_mouse.py
@author: Akira Nagamori
Last update: 6/22/21
Descriptions: 
    Defines model parameters associated with muscle 1
"""
import numpy as np

L0 = 0.55 # muscle optimal length [cm]
alpha = 21*np.pi/float(180) # pennation angle [rad]
L0T = 0.5*1.05 #*1.05; # optimal length of a series-elastic element [cm]
Lce_initial = L0*1; # initial muscle length [cm]
Lmax = 1.4
    
F0 = 2 # Maximum force output [N]
sigma = 31.8 # muscle specific tension [N/cm^2]
PCSA = F0/(np.cos(alpha)*sigma)  # physiological cross-sectional area [cm^2]
density = 1.06 # muscle densitiy [g/cm^3]
mass = PCSA*density*L0/1000 # muscle mass [kg]

Ur = 0.8 # activation level at which all units are recruited 
F_pcsa_slow = 0.2 # fraction of PCSA of slow-twitch unit
U_slow_th = 0.001 # recruitment threshold of slow-twitch unit 
U_fast_th = Ur*F_pcsa_slow # recruitment threshold of fast-twitch unit
    

