"""
CoralModel v3 - hydrodynamics

@author: Gijs G. Hendrickx
"""

import numpy as np
import os
from scipy.optimize import fsolve
import bmi.wrapper
import faulthandler
faulthandler.enable()


class Delft3D:
    """Coupling of CoralModel to Delft3D using the BMI wrapper."""
    
    def __init__(self, home_dir, mdu_file, config_file=None):
        self.home = home_dir
        self.mdu = mdu_file
        self.config = config_file
        
        self.environment()
        self.initialize()
        
        self.timestep = None
    
    def __str__(self):
        if self.config:
            incl = f'DFlow- and DWaves-modules'
            files = f'\n\tDFlow file         : {self.mdu}'\
                    f'\n\tConfiguration file : {self.config}'
        else:
            incl = f'DFlow-module'
            files = f'\n\tDFlow file         : {self.mdu}'
        msg = (
            f'Coupling with Delft3D model (incl. {incl}) with the following '
            f'settings:'
            f'\n\tDelft3D home dir.  : {self.home}'
            f'{files}'
        )
        return msg
    
    def __repr__(self):
        msg = (
            f'Delft3D(home_dir={self.home}, mdu_file={self.mdu}, '
            f'config_file={self.config})'
        )
        return msg
        
    @property
    def dflow_dir(self):
        """Directory to DFlow-ddl."""
        return os.path.join(self.home, 'dflowfm', 'bin', 'dflowfm.dll')
    
    @dflow_dir.setter
    def dflow_dir(self, directory):
        """Set directory to DFlow-ddl."""
        if isinstance(directory, str):
            directory = directory.replace('/', '\\').split('\\')
        self.dflow_dir = os.path.join(self.home, *directory)
    
    @property
    def dimr_dir(self):
        """Directory to DIMR-dll."""
        return os.path.join(self.home, 'dimr', 'bin', 'dimr_dll.dll')
    
    @dimr_dir.setter
    def dimr_dir(self, directory):
        """Set directory to DIMR-dll."""
        if isinstance(directory, str):
            directory = directory.replace('/', '\\').split('\\')
        self.dimr_dir = os.path.join(self.home, *directory)
        
    @property
    def model(self):
        """Main model-object."""
        if self.config:
            return self.model_dimr
        return self.model_fm
    
    @property
    def model_fm(self):
        """Deflt3D-FM model-object."""
        return bmi.wrapper.BMIWrapper(
            engine=self.dflow_dir, 
            configfile=self.mdu
        )
        
    @property
    def model_dimr(self):
        """Delft3D DIMR model-object."""
        if not self.config:
            return bmi.wrapper.BMIWrapper(
                engine=self.dimr_dir,
                configfile=self.config
            )
    
    def environment(self):
        """Set Python environment to include Delft3D-code."""
        dirs = [
            os.path.join(self.home, 'share', 'bin'),
            os.path.join(self.home, 'dflowfm', 'bin'),
        ]
        if self.config:
            dirs.extend([
                os.path.join(self.home, 'dimr', 'bin'),
                os.path.join(self.home, 'dwaves', 'bin'),
                os.path.join(self.home, 'esmf', 'scripts'),
                os.path.join(self.home, 'swan', 'scripts'),
            ])
            
        env = ';'.join(dirs)
        os.environ['PATH'] = env
            
        print(f'\nEnvironment \"PATH\":')
        [print(f'\t{path}') for path in dirs]
        
    def initialize(self):
        """Initialize the working model."""
        self.model.initialize()
        
    def update(self, timestep):
        """Update the working model."""
        self.timestep = timestep
        self.reset_counters()
        self.model.update(self.timestep)
    
    def finalize(self):
        """Finalize the working model."""
        self.model.finalize()
        
    def reset_counters(self):
        """Reset properties for next model update."""
        sums = self.model_fm.get_var('is_sumvalsnd')
        sums.fill(0.)
        self.model_fm.set_var('is_sumvalsnd', sums)
        
        maxs = self.model_fm.get_var('is_maxvalsnd')
        maxs.fill(0.)
        self.model_fm.set_var('is_maxvalsnd', maxs)
    
    def get_var(self, variable):
        """Get variable from DFlow-model."""
        return self.model_fm.get_var(variable)
    
    def set_var(self, variable):
        """Set variable to DFlow-model."""
        self.model_fm.set_var(variable)
    
    @property
    def space(self):
        """Number of non-boundary boxes; i.e. within-domain boxes."""
        return self.model_fm.get_var('ndxi')
    
    @property
    def x(self):
        """Center of gravity's x-coordinates as part of `space`."""
        return self.model_fm.get_var('xzw')[range(self.space)]
    
    @property
    def y(self):
        """Center of gravity's y-coodinates as part of `space`."""
        return self.model_fm.get_var('yzw')[range(self.space)]
    
    @property
    def vel_wave(self):
        """Wave orbital velocity [ms-1] as part of `space`."""
        return self.model_fm.get_var('Uorb')[range(self.space)]
    
    @property
    def vel_curr_mn(self):
        """Mean current velocity [ms-1] as part of `space`."""      
        vel_sum = self.model_fm.get_var('is_sumvalsnd')[range(self.space), 1]
        return vel_sum / self.timestep
    
    @property
    def vel_curr_mx(self):
        """Maximum current velocity [ms-1] as part of `space`."""
        return self.model_fm.get_var('is_maxvalsnd')[range(self.space), 1]
    
    @property
    def per_wave(self):
        """Peak wave period [s] as part of `space`."""
        return self.model_fm.get_var('twav')[range(self.space)]
    
    @property
    def depth(self):
        """Water depth [m] as part of `space`"""
        dep_sum = self.model_fm.get_var('is_sumvalsnd')[range(self.space), 2]
        return dep_sum / self.timestep
        
    @property
    def can_dia(self):
        """Representative diameter of the canopy [m] as part of `space`."""
        return self.model_fm.get_var('diaveg')[range(self.space)]
    
    @can_dia.setter
    def can_dia(self, canopy_diameter):
        self.model_fm.set_var('diaveg', canopy_diameter)
    
    @property
    def can_height(self):
        """Height of the canopy [m] as part of `space`."""
        return self.model_fm.get_var('stemheight')[range(self.space)]
    
    @can_height.setter
    def can_height(self, canopy_height):
        self.model_fm.set_var('stemheight', canopy_height)
    
    @property
    def can_den(self):
        """Density of the canopy [pcs m-2] as part of `space`."""
        return self.model_fm.get_var('rnveg')[range(self.space)]
    
    @can_den.setter
    def can_den(self, canopy_density):
        self.model_fm.set_var('rnveg', canopy_density)
        
        
class Reef1D:
    """Simplified one-dimensional hydrodynamic model over a (coral) reef."""
    
    def __init__(self, bathymetry, wave_height, wave_period, dx=1):
        """
        Initialisation of `Reef1D`.

        Parameters
        ----------
        bathymetry : numeric
            Bathymetric cross-shore data with means sea level as reference [m]
            and x=0 at the offshore boundary.
        wave_height : numeric
            Significant wave height [m].
        wave_period : numeric
            Peak wave period [s].
        dx : numeric
            Spatial step between bathymetric data points [m].
        """
        self.bath = bathymetry
        self.Hs = wave_height
        self.Tp = wave_period
        self.dx = dx
        
        self.z = np.zeros(self.space)
        
        self._diameter = None
        self._height = None
        self._density = None
    
    def __str__(self):
        msg = (
            f'One-dimensional simple hydrodynamic model to simulate the '
            f'hydrodynamics on a (coral) reef with the following settings:'
            f'\n\tBathymetric cross-shore data : {type(self.bath).__name__}'
            f'\n\t\trange [m]  : {min(self.bath)}-{max(self.bath)}'
            f'\n\t\tlength [m] : {self.space * self.dx}'
            f'\n\tSignificant wave height [m]  : {self.Hs}'
            f'\n\tPeak wave period [s]         : {self.Tp}'
        )
        return msg
    
    def __repr__(self):
        msg = (
            f'Reef1D(bathymetry={self.bath}, wave_height={self.Hs}, '
            f'wave_period={self.Tp})'
        )
        return msg
    
    def update(self, timestep):
        pass
    
    def finalize(self):
        pass
    
    @property
    def space(self):
        return len(self.bath)
    
    @property
    def x(self):
        return np.arange(0, self.space, self.dx)
    
    @property
    def y(self):
        return 0
    
    @property
    def vel_wave(self):
        pass
    
    @property
    def vel_curr_mn(self):
        return 0
    
    @property
    def vel_curr_mx(self):
        return 0
    
    @property
    def per_wav(self):
        return self.Tp
    
    @property
    def depth(self):
        return self.bath + self.z
    
    @property
    def can_dia(self):
        return self._diameter
    
    @can_dia.setter
    def can_dia(self, canopy_diameter):
        self._diameter = canopy_diameter
    
    @property
    def can_height(self):
        return self._height
    
    @can_height.setter
    def can_height(self, canopy_height):
        self._height = canopy_height
    
    @property
    def can_den(self):
        return self._density
    
    @can_den.setter
    def can_den(self, canopy_density):
        self._density = canopy_density
        
    def dispersion(self, wave_length, wave_period, depth, grav_acc):
        """Dispersion relation to determine the wave length based on the
        wave period.
        """
        func = wave_length - ((grav_acc * wave_period ** 2) / (2 * np.pi)) * \
            np.tanh(2 * np.pi * depth / wave_length)
        return func
    
    @property
    def wave_length(self):
        """Solve the dispersion relation to retrive the wave length."""
        L0 = 9.81 * self.per_wav ** 2
        L = np.zeros(len(self.depth))
        for i, h in enumerate(self.depth):
            if h > 0:
                L[i] = fsolve(self.dispersion, L0, args=(self.per_wav, h, 9.81))
        return L
    
    @property
    def wave_frequnecy(self):
        return 2 * np.pi / self.per_wav
    
    @property
    def wave_number(self):
        k = np.zeros(len(self.wave_length))
        k[self.wave_length > 0] = 2 * np.pi / self.wave_length[
            self.wave_length > 0]
        return k
    
    @property
    def wave_celerity(self):
        return self.wave_length / self.per_wav
    
    @property
    def group_celerity(self):
        n = .5 * (1 + (2 * self.wave_number * self.depth) /
                  (np.sinh(self.wave_number * self.depth)))
        return n * self.wave_celerity
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = Reef1D(np.linspace(20, 2), 1, 4)
    plt.plot(model.x, model.z)
    plt.plot(model.x, -model.depth)
    plt.plot(model.x, model.wave_celerity)
    plt.plot(model.x, model.group_celerity)
    