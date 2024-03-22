from modulus.sym import quantity
from sympy import Symbol


# fluid params Air @ 25 C
fluid_density  = quantity(1.185, "kg/m^3") #rho
fluid_dynamic_viscosity = quantity(1.831e-05, "kg/(m*s)") #dynamic viscosity
fluid_kinematic_viscosity = fluid_dynamic_viscosity/fluid_density #m^2/s
fluid_specific_heat = quantity(1004.4, "J/(kg*K)")  # J/(kg K) 
fluid_conductivity = quantity(0.0261, "W/(m*K)")  # W/(m K)
fluid_diffusivity = fluid_conductivity / (fluid_specific_heat*fluid_density)



# solid params
solid_density = quantity(100, "kg/m^3")  # kg/m3 
solid_specific_heat = quantity(500, "J/(kg*K)")  # J/(kg K) 
solid_conductivity = quantity(1, "W/(m*K)")  # W/(m K)
# solid_conductivity = quantity(0.1, "W/(m*K)")  # W/(m K)
solid_diffusivity = solid_conductivity / (solid_specific_heat*solid_density)

print(fluid_diffusivity)
print(solid_diffusivity)

# define sympy varaibles to parametize domain curves
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
y_pos = Symbol("y_pos")
x_pos = Symbol("x_pos")

# parametric variation
holeRadius1 = Symbol("holeRadius1")
holeRadius2 = Symbol("holeRadius2")
inletVelocity1 = Symbol("inletVelocity1")
inletVelocity2 = Symbol("inletVelocity2")
holePosZ1 = Symbol("holePosZ1")
holePosX1= Symbol("holePosX1")
holePosZ2 = Symbol("holePosZ2")
holePosX2= Symbol("holePosX2")

paramRang = {
    holeRadius1: ((3, 5),"mm"),
    holeRadius2: ((3, 5),"mm"),
    inletVelocity1: ((11, 20),"m/s"),
    inletVelocity2: ((11, 20),"m/s"),
    holePosX1: ((10, 22),"mm"),
    holePosX2: ((-22, -10),"mm"),
    holePosZ1: ((-8, 8),"mm"),
    holePosZ2: ((-8, 8),"mm")
}

# paramRang = {
#     holeRadius1: (5,"mm"),
#     holeRadius2: (3,"mm"),
#     inletVelocity1: (11,"m/s"),
#     inletVelocity2: (11,"m/s"),
#     holePosX1: (10,"mm"),
#     holePosX2: (-22,"mm"),
#     holePosZ1: (-8,"mm"),
#     holePosZ2: (-8,"mm")
# }