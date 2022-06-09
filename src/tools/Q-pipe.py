import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math


data = pd.read_csv(r"C:\Users\dzimball\OneDrive - Stichting Deltares\Desktop\WL_zuidgors.csv", delimiter=';')
r = 0.5 #pipe radius
n = 0.018 #Manning number
WL = data.iloc[0:5000, 1]
# x = np.zeros(1500)
# for i in range(0, 1500):
#     x[i] = i
# WL = np.zeros(x.shape)
# for j in range(0, len(x)):
#     WL[j] = 2.5 * math.cos((2*math.pi/745.2)*x[j]+(745.2/4))

BL = 1.9 #bed level at pipe location [mNAP]
WD = WL - BL #water depth
WD[WD < 0] = 0
dWD = np.zeros(WD.shape)
R = np.zeros(WD.shape)
Q = np.zeros(WD.shape)
A = np.zeros(WD.shape)
C = np.zeros(WD.shape)

r2 = 0.25 #m
BL2 = 2 #m
WD2 = WL - BL2
WD2[WD2 < 0] = 0
Q2 = np.zeros(WD2.shape)
dWD2 = np.zeros(WD2.shape)
R2 = np.zeros(WD2.shape)
A2 = np.zeros(WD2.shape)
C2 = np.zeros(WD2.shape)
WDbasin = np.zeros(WD2.shape)
for i in range(0, (len(WD)-1)):
    # dWD[i+1] = WD[i+1]-WD[i] # slope (water level differnce)
    WDbasin[i] = ((sum(Q[0:i+1])+sum(Q2[0:i+1])) * 3 * 60 / 18600)
    WDbasin[WDbasin < 0] = 0
    dWD[i + 1] = WD[i + 1] - WDbasin[i]  # previous discharges*time step divided by basin area
    if dWD[i] == 0:
        R[i] = 0
        A[i] = 0
        Q[i] = 0
    else:
        if WD[i] > r:
            WD[i] = r
        O = 2*np.arccos(((r-WD[i])/r))
        A[i] = ((r**2) * (O-np.sin(O))) / 2
        if O == 0:
            R[i] = A[i]
        else:
            R[i] = A[i]/(r*O)
        # R[i] = A[i]/(r*O)
        C[i] = (1/n)*(R[i]**(1/6))
        if dWD[i] >= 0:
            Q[i] = C[i]*np.sqrt((R[i]*dWD[i]))*A[i]
        else:
            Q[i] = - C[i]*np.sqrt((R[i]*(-dWD[i])))*A[i]

    # dWD[j+1] = WD2[j+1]-WD2[j] # slope (water level differnce)
    dWD2[i+1] = WD2[i+1] - WDbasin[i] # previous discharges*time step divided by basin area
    if dWD2[i] == 0:
        R2[i] = 0
        A2[i] = 0
        Q2[i] = 0
    else:
        if WD2[i] > r2:
            WD2[i] = r2
        O2 = 2*np.arccos(((r2-WD2[i])/r2))

        A2[i] = ((r2**2) * (O2-np.sin(O2))) / 2
        if O2 == 0:
            R2[i] = A2[i]
        else:
            R2[i] = A2[i]/(r2*O2)
        C2[i] = (1/n)*(R2[i]**(1/6))
        if dWD2[i] >= 0:
            Q2[i] = C2[i]*np.sqrt((R2[i]*dWD2[i]))*A2[i]
        else:
            Q2[i] = - C2[i]*np.sqrt((R2[i]*(-dWD2[i])))*A2[i]

Q = np.nan_to_num(Q, nan=0.0)
vel1 = Q/r**2*math.pi
Q2 = np.nan_to_num(Q2, nan=0.0)
vel2 = Q2/r2**2*math.pi

# plt.plot(C)
# plt.show
plt.plot(Q)
plt.xlabel("Time")
plt.ylabel("Discharge [m3/s]")
plt.show

# volume1 = sum(Q[0:500]*3*60)

plt.plot(Q2)
plt.xlabel("Time")
plt.ylabel("Discharge [m3/s]")
plt.show

# volume2 = sum(Q2[0:500]*3*60)
#
# volume_tot = volume1+volume2

# df_Q = pd.DataFrame()
# df_Q.index=data['Time [-]']
# df_Q.loc[:,0]=Q
# df_Q.columns = ['Discharge [m3/s]']
# # plt.plot(df_Q['Time [-]'], df_Q['Discharge [m3/s]'])
# # plt.show
# # df_Q = pd.DataFrame([data['Time [-]'], q2[:,0]], columns=['Time[-]', 'Discharge[m3/s]'])
# df_Q.to_csv(r"C:\Users\dzimball\OneDrive - Stichting Deltares\Desktop\Q_zuidgors_tides.csv", sep= ";")

volume = (sum(Q[0:13])+sum(Q2[0:13]))*3*60