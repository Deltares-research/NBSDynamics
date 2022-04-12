[SedimentFileInformation]
    FileCreatedBy         = Deltares, FM-Suite DFlowFM Model Version 4.5.0.2430, DFlow FM Version 1.2.130.69672M 
    FileCreationDate      = Fri Apr 08 2022, 11:29:17 
    FileVersion           = 02.00                  
[SedimentOverall]
    Cref                  = 1600                   [kg/m3]   Reference density for hindered settling calculations
[Sediment]
    Name                  = #sand#                           Name of sediment fraction
    SedTyp                = sand                             Must be "sand", "mud" or "bedload"
    IniSedThick           = 5                      [m]       Initial sediment layer thickness at bed
    FacDss                = 1                      [-]       Factor for suspended sediment diameter
    RhoSol                = 2650                   [kg/m3]   Specific density
    TraFrm                = -1                               Integer selecting the transport formula
    CDryB                 = 1600                   [kg/m3]   Dry bed density
    SedDia                = 0.0002                 [m]       Median sediment diameter (D50)
    IopSus                = 0                                Option for determining suspended sediment diameter
    AksFac                = 1                      [-]       Calibration factor for Van Rijn’s reference height
    Rwave                 = 2                      [-]       Calibration factor wave roughness height
    RDC                   = 0.01                   [m]       Current related roughness ks
    RDW                   = 0.02                   [m]       Wave related roughness kw
    IopKCW                = 1                                Option for ks and kw
    EpsPar                = False                            Use Van Rijn's parabolic mixing coefficient
