import numpy as np
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer

from globalProperties import *
from parameterRangeContainer import parameterRangeContainer



#############
# Real Params
#############

paramRanges_ch = parameterRangeContainer(paramRang)

# boundary params
channel_inlet_velocity = quantity(5, "m/s")
# cooling_inlet_velocity = quantity(1.0, "m/s")
noslip_velocity = quantity(0.0, "m/s")
channel_outlet_p = quantity(0.0, "pascal")
# cooling_outlet_p = quantity(1.0, "pa")
channel_inlet_temp = quantity(700, "K")
cooling_inlet_temp = quantity(300, "K")
channel_HTC = quantity(80, "W/((m^2)*K)")

################
# Non dim params
################
velocity_scale_ch = channel_inlet_velocity*2
density_scale_ch = fluid_density
length_scale_ch = quantity(0.13, "m")
time_scale_ch=length_scale_ch / velocity_scale_ch
mass_scale_ch=density_scale_ch * (length_scale_ch ** 3)
temp_scale_ch = channel_inlet_temp

channelNonDim = NonDimensionalizer(
    length_scale=length_scale_ch,
    time_scale=time_scale_ch,
    mass_scale=mass_scale_ch,
    temperature_scale=temp_scale_ch,
)

##############################
# Nondimensionalization Params
##############################

# fluid params
nd_fluid_dynamic_viscosity_ch = channelNonDim.ndim(fluid_dynamic_viscosity)
nd_fluid_kinematic_viscosity_ch = channelNonDim.ndim(fluid_kinematic_viscosity)
nd_fluid_density_ch = channelNonDim.ndim(fluid_density)
nd_fluid_specific_heat_ch = channelNonDim.ndim(fluid_specific_heat)
nd_fluid_conductivity_ch = channelNonDim.ndim(fluid_conductivity)
nd_fluid_diffusivity_ch = channelNonDim.ndim(fluid_diffusivity)

print("nd_fluid_diffusivity_ch ", nd_fluid_diffusivity_ch)

# solid params
nd_solid_density = channelNonDim.ndim(solid_density)
nd_solid_specific_heat = channelNonDim.ndim(solid_specific_heat)
nd_solid_conductivity = channelNonDim.ndim(solid_conductivity)
nd_solid_diffusivity = channelNonDim.ndim(solid_diffusivity)

print("nd_solid_diffusivity ", nd_solid_diffusivity)

# boundary params
nd_channel_inlet_velocity = channelNonDim.ndim(channel_inlet_velocity)
nd_channel_inlet_temp = channelNonDim.ndim(channel_inlet_temp)
nd_channel_outlet_p = channelNonDim.ndim(channel_outlet_p)
nd_channel_inlet_temp = channelNonDim.ndim(channel_inlet_temp)
nd_channel_HTC = channelNonDim.ndim(channel_HTC)

nd_cooling_inlet_temp = channelNonDim.ndim(cooling_inlet_temp)

nd_noslip_velocity_ch =channelNonDim.ndim(noslip_velocity)

paramRanges_ch.nondim(channelNonDim)
