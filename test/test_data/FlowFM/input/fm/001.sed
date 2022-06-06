[SedimentFileInformation]
   FileCreatedBy    = Delft3D FLOW-GUI, Version: 3.56.29165         
   FileCreationDate = Tue Feb 21 2017, 18:35:38         
   FileVersion      = 02.00                        
[SedimentOverall]
   Cref             =  1.6000000e+003      [kg/m3]  CSoil Reference density for hindered settling calculations
   IopSus           = 0                             If Iopsus = 1: susp. sediment size depends on local flow and wave conditions
[Sediment]
   Name             = #Mud#                Name of sediment fraction
   SedTyp           = mud                           Must be "sand", "mud" or "bedload"
   RhoSol           =  2.6500000e+003      [kg/m3]  Specific density
   SalMax           =  0.0000000e+000      [ppt]    Salinity for saline settling velocity
   WS0              =  5.0000000e-004      [m/s]    Settling velocity fresh water
   WSM              =  5.0000000e-004      [m/s]    Settling velocity saline water
   TcrSed           =  1.0000000e+003      [N/m2]   Critical bed shear stress for sedimentation (uniform value or filename)
   TcrEro           =  5.0000000e-001      [N/m2]   Critical bed shear stress for erosion       (uniform value or filename)
   EroPar           =  5.0000000e-005      [kg/m2/s] Erosion parameter                           (uniform value or filename)
   CDryB            =  5.0000000e+002      [kg/m3]  Dry bed density
   IniSedThick      =  5.0000000e+000      [m]      Initial Sediment Thickness (uniform value or filename)       
   FacDSS           =  1.0000000e+000      [-]      FacDss * SedDia = Initial suspended sediment diameter. Range [0.6 - 1.0]
