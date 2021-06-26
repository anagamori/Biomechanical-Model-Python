"""
muscle_model_functions.py
@author: Akira Nagamori
Last update: 6/26/21
Descriptions: 
    Defines functions associated with muscle model
"""

import numpy as np

# Determine the initial length of the series elastic element and that of the musculotendon unit
def InitialLength(parameter):
    #Lce_length_initial in cm
    cT = 27.8;
    kT = 0.0047;
    LrT = 0.964;
    c1 = 23;
    k1 = 0.046;
    Lr1 = 1.17;
    
    L0 = parameter.L0; #optimal muscle length 
    L0T = parameter.L0T; #optimal tendon length
    
    PE1_force_initial = c1 * k1 * np.log(np.exp((parameter.Lce_initial/parameter.L0/parameter.Lmax - Lr1)/k1)+1)*np.cos(parameter.alpha); #PE1 force at muscle length = Lce_initial defined by the user
    normalized_SE_Length_initial = kT*np.log(np.exp(PE1_force_initial/cT/kT)-1)+LrT #Normalized tendon length at PE1_force_initial
    Lse_initial = normalized_SE_Length_initial; #initial length of the series-elastic element
    Lmt_initial = parameter.Lce_initial*np.cos(parameter.alpha)+Lse_initial*L0T #initial length of the musculotendon unit
    Lce_initial = parameter.Lce_initial/L0 #normalized initial length of the contractile element

    return (Lce_initial,Lse_initial,Lmt_initial)


# First-order dynamics for neural activation (Eq. 2)
def U_function(U,U_eff,step):
    if U >= U_eff:
        T_U = 0.03;
    else:
        T_U = 0.15;
    U_eff_dot = (U-U_eff)/T_U;
    U_eff = U_eff_dot*step + U_eff;
        
    return U_eff

# Convert effective neural activation to firing frequency of slow-twitch unit
def U2f_slow_function(U_eff,U_th):
    f_half = 8.5;
    fmin = 0.5*f_half;
    fmax = 2*f_half;
    if U_eff >=  U_th:
        f_env = (fmax-fmin)/(1-U_th)*(U_eff-U_th)+fmin;
        f_env = f_env/f_half;
    else:
        f_env = 0;
        
    return f_env

# Convert effective neural activation to firing frequency of fast-twitch unit
def U2f_fast_function(U_eff,U_th):
    f_half = 34;   
    fmin = 0.5*f_half;
    fmax= 2*f_half;

    if U_eff >=  U_th:
        f_env = (fmax-fmin)/(1-U_th)*(U_eff-U_th)+fmin;
        f_env = f_env/f_half;
    else:
        f_env = 0;
        
    return f_env
        
# first-oder dynamics of firing frquency for slow-twithc unit (Eq. 3&4)        
def f_slow_function(f_out,f_in,f_env,f_eff_dot,Af,Lce,step):
    T_f1 = 0.0343;
    T_f2 = 0.0227;
    T_f3 = 0.047;
    T_f4 = 0.0252;
    
    if f_eff_dot >= 0:
        Tf = T_f1*np.power(Lce,2)+T_f2*f_env;
    else:
        Tf = (T_f3 + T_f4*Af)/Lce;
        
    f_out_dot = (f_in - f_out)/Tf;
    f_out = f_out_dot*step + f_out;
    
    return (f_out,f_out_dot)
      
# first-oder dynamics of firing frquency for fast-twithc unit (Eq. 3&4)    
def f_fast_function(f_out,f_in,f_env,f_eff_dot,Af,Lce,step):
    T_f1 = 0.0206;
    T_f2 = 0.0136;
    T_f3 = 0.0282;
    T_f4 = 0.0151;
    
    if f_eff_dot >= 0:
        Tf = T_f1*np.power(Lce,2)+T_f2*f_env;
    else:
        Tf = (T_f3 + T_f4*Af)/Lce;
        
    f_out_dot = (f_in - f_out)/Tf;
    f_out = f_out_dot*step + f_out;
    
    return (f_out,f_out_dot)

# Activation-frequency relationship for slow-twitch unit (Eq. 5)
def Af_slow_function(f_eff,L,Y):
    a_f = 0.56;
    n_f0 = 2.1;
    n_f1 = 5;
    n_f = n_f0 + n_f1*(1/L-1);
    Af = 1 - np.exp(-np.power(Y*f_eff/(a_f*n_f),n_f));
    return Af

# Activation-frequency relationship for slow-twitch unit (Eq. 5)
def Af_fast_function(f_eff,L,S):
    a_f = 0.56;
    n_f0 = 2.1;
    n_f1 = 3.3;
    n_f = n_f0 + n_f1*(1/L-1);
    Af = 1 - np.exp(-np.power(S*f_eff/(a_f*n_f),n_f));
    return Af

# Yield property (Eq. 7) 
def yield_function(Y,V,step):
    c_y = 0.35;
    V_y = 0.1;
    T_y = 0.2;
    Y_dot = (1-c_y*(1-np.exp(-abs(V)/(V_y)))-Y)/T_y;
    Y = Y_dot*step + Y;
    return Y 

# Sag property (Eq. 6)
def sag_function(S,f_eff,step):
    if f_eff < 0.1:
        a_s = 1.76;
    else:
        a_s = 0.96;
    T_s = 0.043;
    S_dot = (a_s - S)/T_s;
    S = S_dot*step + S;
    return S

# Force-length relationship for slow-twitch unit (Eq. 7)
def FL_slow_function(L):
    beta = 2.3;
    omega = 1.12;
    rho = 1.62;
    
    FL = np.exp(-np.power(abs((np.power(L,beta) - 1)/(omega)),rho));
    return FL

# Force-length relationship for fast-twitch unit (Eq. 8)
def FL_fast_function(L):
    beta = 1.55;
    omega = 0.75;
    rho = 2.12;
    
    FL = np.exp(-np.power(abs((np.power(L,beta) - 1)/(omega)),rho));
    return FL

# Concentric (i.e. shortening) force-velocity relationship for slow-twitch unit (Eq. 9)
def FV_con_slow_function(L,V):
    Vmax = -7.88;
    cv0 = 5.88;
    cv1 = 0;
    
    FVcon = (Vmax - V)/(Vmax + (cv0 + cv1*L)*V);
    return FVcon

# Concentric (i.e. shortening) force-velocity relationship for fast-twitch unit (Eq. 9)
def FV_con_fast_function(L,V):
    Vmax = -9.15;
    cv0 = -5.7;
    cv1 = 9.18;
    
    FVcon = (Vmax - V)/(Vmax + (cv0 + cv1*L)*V);
    return FVcon

# Eccentric (i.e. lengthening) force-velocity relationship for slow-twitch unit (Eq. 9)
def FV_ecc_slow_function(L,V):
    av0 = -4.7;
    av1 = 8.41;
    av2 = -5.34;
    bv = 0.35;
    FVecc = (bv - (av0 + av1*L + av2*np.power(L,2))*V)/(bv+V);

    return FVecc

# Eccentric (i.e. lengthening) force-velocity relationship for fast-twitch unit (Eq. 9)
def FV_ecc_fast_function(L,V):
    av0 = -1.53;
    av1 = 0;
    av2 = 0;
    bv = 0.69;
    FVecc = (bv - (av0 + av1*L + av2*np.power(L,2))*V)/(bv+V);

    return FVecc

# computing weighting, W (Eq. 11)
def weighting_function(U_eff,U_slow_th,U_fast_th):
    if U_eff < U_slow_th:
        W1 = 0;
    elif U_eff < U_fast_th:
        W1 = (U_eff - U_slow_th)/(U_eff - U_slow_th);
    else:
        W1 = (U_eff - U_slow_th)/((U_eff - U_slow_th) + (U_eff - U_fast_th));
    if U_eff < U_fast_th:
        W2 = 0;
    else:
        W2 = (U_eff - U_fast_th)/((U_eff - U_slow_th) + (U_eff - U_fast_th));

    return (W1,W2)

# Passive force from passive element 1 (Eq. 15)
def F_pe_1_function(L,V):
    c1_pe1 = 23;
    k1_pe1 = 0.046;
    Lr1_pe1 = 1.17;
    eta = 0.01;
    
    Fpe1 = c1_pe1 * k1_pe1 * np.log(np.exp((L - Lr1_pe1)/(k1_pe1))+1) + eta*V;

    return Fpe1

# Passive force from passive element 2 (Eq. 16)
def F_pe_2_function(Lce):
    c2_pe2 = -0.02;
    k2_pe2 = -21;
    Lr2_pe2 = 0.70;
    
    Fpe2 = c2_pe2*np.exp((k2_pe2*(Lce-Lr2_pe2))-1);
    return Fpe2

# Force-length relationship of series-elastic element (Eq. 17)
def F_se_function(Lse):
    cT_se = 27.8; 
    kT_se = 0.0047;
    LrT_se = 0.964;
    
    Fse = cT_se * kT_se * np.log(np.exp((Lse - LrT_se)/(kT_se))+1);
    return Fse

# Equation for contraction dyamics (Eq. 18)
def contraction_dynamics(U_eff,f_eff_slow,f_eff_fast,W_slow,W_fast,Y,S,Lse,x,parameter):
              
    Af_slow = Af_slow_function(f_eff_slow,x[0]/(parameter.L0/100),Y);
    Af_fast = Af_fast_function(f_eff_fast,x[0]/(parameter.L0/100),S);      
        
    FL_slow = FL_slow_function(x[0]/(parameter.L0/100));
    FL_fast = FL_fast_function(x[0]/(parameter.L0/100));
    
    if x[1]/(parameter.L0/100) <= 0.0:
        FV_slow = FV_con_slow_function(x[0]/(parameter.L0/100),x[1]/(parameter.L0/100));
        FV_fast = FV_con_fast_function(x[0]/(parameter.L0/100),x[1]/(parameter.L0/100));
    else:
        FV_slow = FV_ecc_slow_function(x[0]/(parameter.L0/100),x[1]/(parameter.L0/100));
        FV_fast = FV_ecc_fast_function(x[0]/(parameter.L0/100),x[1]/(parameter.L0/100));
        
    Fpe1 = F_pe_1_function(x[0]/(parameter.L0/100)/(parameter.Lmax),x[1]/(parameter.L0/100));
    Fpe2 = F_pe_2_function(x[0]/(parameter.L0/100));
    if Fpe2 > 0:
        Fpe2 = 0;     
        
    Fce = U_eff*((W_slow*Af_slow*(FL_slow*FV_slow+Fpe2))+(W_fast*Af_fast*(FL_fast*FV_fast+Fpe2)));
    if Fce < 0:
        Fce = 0;
    elif Fce > 1:
        Fce = 1;
        
    Fce = (Fce + Fpe1)*parameter.F0;
    Fse = F_se_function(Lse)*parameter.F0;
    Fd = Fse*np.cos(parameter.alpha) - Fce*np.power(np.cos(parameter.alpha),2);
    
    temp = x[1]*np.power(np.tan(parameter.alpha),2)/x[0]
    A_m = np.array([[0.0,1.0],[0.0,0.0]]);
    A_m[1,1] = temp
    B_m = np.array([[0.0],[1/parameter.mass]])
    dx = np.dot(A_m,x) + B_m*Fd;
    
    return dx





