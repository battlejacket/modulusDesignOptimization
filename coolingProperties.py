import numpy as np
#from three_fin_geometry import *
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer

from globalProperties import *

from parameterRangeContainer import parameterRangeContainer


paramRanges = parameterRangeContainer(paramRang)

# boundary params
cooling_inlet_velocity =  quantity(20, "m/s")
noslip_velocity = quantity(0.0, "m/s")
cooling_outlet_p = quantity(0.0, "pascal")
cooling_inlet_temp = quantity(300, "K") #not used, specify in channelProperties


################
# Non dim params
################
velocity_scale = cooling_inlet_velocity
density_scale = fluid_density
length_scale = quantity(0.01, "m")

time_scale=length_scale / velocity_scale
mass_scale=density_scale * (length_scale ** 3)
# temp_scale = quantity(700, "K")
temp_scale = cooling_inlet_temp

coolingNonDim = NonDimensionalizer(
    length_scale=length_scale,
    time_scale=time_scale,
    mass_scale=mass_scale,
    temperature_scale=temp_scale,
)

##############################
# Nondimensionalization Params
##############################

# fluid params
nd_fluid_dynamic_viscosity = coolingNonDim.ndim(fluid_dynamic_viscosity)
nd_fluid_kinematic_viscosity = coolingNonDim.ndim(fluid_kinematic_viscosity)
nd_fluid_density = coolingNonDim.ndim(fluid_density)
nd_fluid_specific_heat = coolingNonDim.ndim(fluid_specific_heat)
nd_fluid_conductivity = coolingNonDim.ndim(fluid_conductivity)
nd_fluid_diffusivity = coolingNonDim.ndim(fluid_diffusivity)

# boundary params
nd_cooling_inlet_velocity = coolingNonDim.ndim(cooling_inlet_velocity)
nd_cooling_outlet_p = coolingNonDim.ndim(cooling_outlet_p)
nd_noslip_velocity =coolingNonDim.ndim(noslip_velocity)

paramRanges.nondim(coolingNonDim)
