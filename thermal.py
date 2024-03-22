import torch
from sympy import Symbol, And, LessThan, sqrt, StrictLessThan, GreaterThan, StrictGreaterThan, Not
import numpy as np
import os

import modulus
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.hydra import to_absolute_path
from csv_rw_skiprows import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.eq.non_dim import Scaler
from modulus.sym.eq.pdes.basic import GradNormal
from basicS import NormalDotVec
from modulus.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from advection_diffusion_s import AdvectionDiffusionS
from diffusion_s import DiffusionInterfaceHTC

from channelProperties import *
from coolingProperties import *
from geometry import dLightGeo
from ansysValidator import ansysValidator

def modHeat(designs=[]):

    @modulus.sym.main(config_path="conf", config_name="conf_thermal")
    def run(cfg: ModulusConfig) -> None:

        # --------------------------------------------------------------------Thermal Equations--------------------------------------------------------------------
        ad = AdvectionDiffusion(T="theta_f", rho=nd_fluid_density_ch, D=nd_fluid_diffusivity_ch, dim=3, time=False)
        ad_co = AdvectionDiffusionS(domain="co", T="theta_fco", rho=nd_fluid_density_ch, D=nd_fluid_diffusivity_ch, dim=3, time=False)
        ad_co2 = AdvectionDiffusionS(domain="co2", T="theta_fco2", rho=nd_fluid_density_ch, D=nd_fluid_diffusivity_ch, dim=3, time=False)
        dif = Diffusion(T="theta_s", D=nd_solid_diffusivity, dim=3, time=False)
        dif_inteface = DiffusionInterface("theta_f", "theta_s", nd_fluid_diffusivity_ch, nd_solid_diffusivity, dim=3, time=False)
        dif_intefaceHTC = DiffusionInterfaceHTC("theta_s", nd_channel_inlet_temp, nd_solid_diffusivity, nd_channel_HTC, dim=3, time=False)
        dif_inteface_co = DiffusionInterface("theta_fco", "theta_s", nd_fluid_diffusivity_ch, nd_solid_diffusivity, dim=3, time=False)
        dif_inteface_co2 = DiffusionInterface("theta_fco2", "theta_s", nd_fluid_diffusivity_ch, nd_solid_diffusivity, dim=3, time=False)
        f_grad = GradNormal("theta_f", dim=3, time=False)
        f_grad_co = GradNormal("theta_fco", dim=3, time=False)
        f_grad_co2 = GradNormal("theta_fco2", dim=3, time=False)
        s_grad = GradNormal("theta_s", dim=3, time=False)

        # --------------------------------------------------------------------Flow Equations--------------------------------------------------------------------
        normal_dot_vel_scaled_co1 = NormalDotVec(["u_co_scaled", "v_co_scaled", "w_co_scaled"], domainName="_scaled_co1")
        normal_dot_vel_scaled_co2 = NormalDotVec(["u_co2_scaled", "v_co2_scaled", "w_co2_scaled"], domainName="_scaled_co2")


        # --------------------------------------------------------------------Channel: Networks and Nodes--------------------------------------------------------------------
        # Channel Flow
        # Net
        input_keys = [Key("x"), Key("y"), Key("z")]
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

        flow_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=False,
        )

        # Scaling Nodes
        # Output
        equation_nodes = Scaler(
            ["u", "v", "w", "p"],
            ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
            ["m/s", "m/s", "m/s", "pascal"],
            channelNonDim).make_node()
        
        # Channel Thermal
        # Net
        input_keys = [Key("x"), Key("y"), Key("z"), Key("holeRadius1_s"), Key("inletVelocity1_s"), Key("holePosX1_s"), Key("holePosZ1_s"), Key("holeRadius2_s"), Key("inletVelocity2_s"), Key("holePosX2_s"), Key("holePosZ2_s")]

        thermal_f_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=[Key("theta_f")],
            frequencies=("axis", [i/2 for i in range(32)]),
            frequencies_params=("axis", [i/2 for i in range(32)]),
            adaptive_activations=True,
            layer_size=512,
            nr_layers=6
        )

        # Scaling Nodes
        # Output
        equation_nodes += Scaler(
            ["theta_f"],
            ["theta_f_scaled"],
            ["K"],
            channelNonDim).make_node()


        # --------------------------------------------------------------------Cooling Holes: Networks and Nodes--------------------------------------------------------------------
        # Scales for Normalization
        channelToCoolingLengthScale = length_scale_ch.magnitude/length_scale.magnitude
        coolingToChannelVelocityScale = velocity_scale.magnitude/velocity_scale_ch.magnitude

        # Cooling Hole 1
        # Cooling Hole 1 Flow
        # Net
        input_keys_co = [Key("x_s"), Key("y_s"), Key("holeRadius1_s"), Key("inletVelocity1_s")]
        output_keys_co = [Key("u_s_co"), Key("v_s_co"), Key("p_s_co")]

        flow_net_co = FourierNetArch(
            input_keys=input_keys_co,
            output_keys=output_keys_co,
        )

        # Scaling Nodes
        # Input
        equation_nodes += [
                 Node.from_sympy(y*channelToCoolingLengthScale, "x_s") #rotated to xyPlane
            ] + [Node.from_sympy(sqrt(((x-holePosX1)*channelToCoolingLengthScale)**2 + ((z-holePosZ1)*channelToCoolingLengthScale)**2), "y_s")
            ] + [Node.from_sympy(holeRadius1*channelToCoolingLengthScale, "holeRadius1_s")
            ] + [Node.from_sympy(inletVelocity1/coolingToChannelVelocityScale, "inletVelocity1_s")
            ] + [Node.from_sympy(holePosX1*channelToCoolingLengthScale, "holePosX1_s")
            ] + [Node.from_sympy(holePosZ1*channelToCoolingLengthScale, "holePosZ1_s")]
        
        # Output
        equation_nodes += [Node.from_sympy(Symbol("u_s_co")*coolingToChannelVelocityScale, "v_co")
            ] + [Node.from_sympy(Symbol("v_s_co")*coolingToChannelVelocityScale, "u_co")
            ] + [Node.from_sympy(Symbol("v_s_co")*coolingToChannelVelocityScale, "w_co")
            ] + [Node.from_sympy(Symbol("p_s_co")*(density_scale.magnitude*velocity_scale.magnitude**2)/(density_scale_ch.magnitude*velocity_scale_ch.magnitude**2), "p_co")]
        
        equation_nodes += Scaler(
            ["u_co", "v_co", "w_co", "p_co"],
            ["u_co_scaled", "v_co_scaled", "w_co_scaled", "p_co_scaled"],
            ["m/s", "m/s", "m/s", "pascal"],
            channelNonDim).make_node()
        
        equation_nodes += Scaler(
            ["area"],
            ["area_scaled"],
            ["m"],
            coolingNonDim).make_node()
        
        # Cooling Hole 1 Thermal
        # Net
        input_keys = [Key("x_st"), Key("y"), Key("z_st"), Key("holeRadius1_s"), Key("inletVelocity1_s"), Key("holePosX1_s"), Key("holePosZ1_s"), Key("holeRadius2_s"), Key("inletVelocity2_s"), Key("holePosX2_s"), Key("holePosZ2_s")]
        
        
        thermal_f_net_co = FourierNetArch(
        # thermal_f_net_co = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=[Key("theta_fco")],
            frequencies=("axis", [i/2 for i in range(16)]),
            frequencies_params=("axis", [i/2 for i in range(16)]),
            adaptive_activations=True,
            layer_size=256,
            nr_layers=4
        )

        # Scaling Nodes
        # Input
        equation_nodes += [
                 Node.from_sympy((x-holePosX1)/(2*holeRadius1), "x_st")
            ] + [Node.from_sympy((z-holePosZ1)/(2*holeRadius1), "z_st")]

        #Output
        equation_nodes += Scaler(
            ["theta_fco", "theta_s"],
            ["theta_fco_scaled", "theta_s_scaled"],
            ["K", "K"],
            channelNonDim).make_node()

        # Cooling Hole 2
        # Cooling Hole 2 Flow
        # Net
        input_keys_co2 = [Key("x_s"), Key("y_s2"), Key("holeRadius2_s"), Key("inletVelocity2_s")]
        output_keys_co2 = [Key("u_s_co2"), Key("v_s_co2"), Key("p_s_co2")]

        flow_net_co2 = FourierNetArch(
            input_keys=input_keys_co2,
            output_keys=output_keys_co2
        )

        # Scaling Nodes
        # Input
        equation_nodes += [
                 Node.from_sympy(sqrt(((x-holePosX2)*channelToCoolingLengthScale)**2 + ((z-holePosZ2)*channelToCoolingLengthScale)**2), "y_s2")
            ] + [Node.from_sympy(holeRadius2*channelToCoolingLengthScale, "holeRadius2_s")
            ] + [Node.from_sympy(inletVelocity2/coolingToChannelVelocityScale, "inletVelocity2_s")
            ] + [Node.from_sympy(holePosX2*channelToCoolingLengthScale, "holePosX2_s")
            ] + [Node.from_sympy(holePosZ2*channelToCoolingLengthScale, "holePosZ2_s")]
        
        # Output
        equation_nodes += [Node.from_sympy(Symbol("u_s_co2")*coolingToChannelVelocityScale, "v_co2")
            ] + [Node.from_sympy(Symbol("v_s_co2")*coolingToChannelVelocityScale, "u_co2")
            ] + [Node.from_sympy(Symbol("v_s_co2")*coolingToChannelVelocityScale, "w_co2")
            ] + [Node.from_sympy(Symbol("p_s_co2")*(density_scale.magnitude*velocity_scale.magnitude**2)/(density_scale_ch.magnitude*velocity_scale_ch.magnitude**2), "p_co2")]
        
        equation_nodes += Scaler(
            ["u_co2", "v_co2", "w_co2", "p_co2"],
            ["u_co2_scaled", "v_co2_scaled", "w_co2_scaled", "p_co2_scaled"],
            ["m/s", "m/s", "m/s", "pascal"],
            channelNonDim).make_node()

        # Cooling Hole 2 Thermal
        # Net
        input_keys = [Key("x_st2"), Key("y"), Key("z_st2"), Key("holeRadius2_s"), Key("inletVelocity2_s"), Key("holePosX2_s"), Key("holePosZ2_s"), Key("holeRadius1_s"), Key("inletVelocity1_s"), Key("holePosX1_s"), Key("holePosZ1_s")]
        
        thermal_f_net_co2 = FourierNetArch(
        # thermal_f_net_co2 = FullyConnectedArch(
            input_keys=input_keys, 
            output_keys=[Key("theta_fco2")],
            frequencies=("axis", [i/2 for i in range(16)]),
            frequencies_params=("axis", [i/2 for i in range(16)]),
            adaptive_activations=True,
            layer_size=256,
            nr_layers=4
        )

        #Scaling Nodes
        # Input    
        equation_nodes += [
                Node.from_sympy((x-holePosX2)/(2*holeRadius2), "x_st2")
           ] + [Node.from_sympy((z-holePosZ2)/(2*holeRadius2), "z_st2")]
        # Output
        equation_nodes += Scaler(
            ["theta_fco2"],
            ["theta_fco2_scaled"],
            ["K"],
            channelNonDim).make_node()
        
        # --------------------------------------------------------------------Solid: Networks and Nodes--------------------------------------------------------------------
        
        # Solid Thermal
        # Net
        input_keys = [Key("x"), Key("y"), Key("z"), Key("holeRadius1_s"), Key("inletVelocity1_s"), Key("holePosX1_s"), Key("holePosZ1_s"), Key("holeRadius2_s"), Key("inletVelocity2_s"), Key("holePosX2_s"), Key("holePosZ2_s")] 
        
        thermal_s_net = FourierNetArch(
            input_keys=input_keys, 
            output_keys=[Key("theta_s")],
            frequencies=("axis", [i/2 for i in range(16)]),
            frequencies_params=("axis", [i/2 for i in range(16)]),
            adaptive_activations=True,
            layer_size=512,
            nr_layers=6
        )


        # --------------------------------------------------------------------Full List of Nodes--------------------------------------------------------------------
        solidNodes = (
            [thermal_s_net.make_node(name="thermal_s_network", optimize=cfg.custom.optimizeTs)]
            + s_grad.make_nodes()
            + [thermal_s_net.make_node(name="thermal_s_network", optimize=cfg.custom.optimizeTs)]
            + dif.make_nodes())
        if cfg.custom.useHTC:
            solidNodes += dif_intefaceHTC.make_nodes()
        
        channelNodes = (
            ad.make_nodes()
            + f_grad.make_nodes()
            + [thermal_f_net.make_node(name="thermal_f_network", optimize=cfg.custom.optimizeTf)]
            + [flow_net.make_node(name="flow_network", optimize=False)])
        if not cfg.custom.useHTC:
            channelNodes += dif_inteface.make_nodes()
        
        coolingHole1Nodes = (
            ad_co.make_nodes()
            + dif_inteface_co.make_nodes()
            + f_grad_co.make_nodes()
            + normal_dot_vel_scaled_co1.make_nodes()
            + [flow_net_co.make_node(name="flow_network_co", optimize=False)]
            + [thermal_f_net_co.make_node(name="thermal_f_network_co", optimize=cfg.custom.optimizeTfco1)])

        coolingHole2Nodes = (
            ad_co2.make_nodes()
            + dif_inteface_co2.make_nodes()
            + f_grad_co2.make_nodes()
            + normal_dot_vel_scaled_co2.make_nodes()
            + [flow_net_co2.make_node(name="flow_network_co2", optimize=False)]
            + [thermal_f_net_co2.make_node(name="thermal_f_network_co2", optimize=cfg.custom.optimizeTfco2)])
            
        thermal_nodes = (
            equation_nodes
            + solidNodes
            + channelNodes
            + coolingHole1Nodes
            + coolingHole2Nodes
        )
        

        # --------------------------------------------------------------------Domain--------------------------------------------------------------------
        thermal_domain = Domain()

        # --------------------------------------------------------------------Geometry and Criterias for Constraints--------------------------------------------------------------------
        geo = dLightGeo(paramRanges_ch)

        def finInteriorCriteria(invar, params):
                sdf = geo.finInteriorCriteriaGeo.sdf(invar, params)
                return np.greater(sdf["sdf"], 0)
        
        def finInteriorCriteria2(invar, params):
                sdf = geo.finInteriorCriteriaGeo2.sdf(invar, params)
                return np.greater(sdf["sdf"], 0)
        
        def solidInteriorCriteria(invar, params):
                sdf = geo.solidInteriorCsg.sdf(invar, params)
                return np.greater(sdf["sdf"], 0)
        
        def solidExteriorWallCriteria(invar, params):
                sdf = geo.solidExteriorWallCriteriaGeo.sdf(invar, params)
                return np.less(sdf["sdf"], 0)
        
        def solidCriteria(invar, params):
                sdf = geo.finExteriorCriteriaGeo.sdf(invar, params)
                return np.less(sdf["sdf"], 0)

        
        # --------------------------------------------------------------------Data Contstraints--------------------------------------------------------------------    
        if cfg.run_mode == "train" and (cfg.custom.includeData or cfg.custom.dataOnly):
            def ansysDataConstraint(file_path, ansysVarNames, modulusVarNames, scales, batches, criteria={}):
                if os.path.exists(to_absolute_path(file_path)):
                    mapping = {}
                    for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
                        mapping[ansVarName] = modulusVarName

                    openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping, skiprows=6)

                    if file_path.split("/")[5] == "co1":
                        print("co1")
                        openfoam_var["z"] = -openfoam_var["z"]

                    if file_path.split("/")[5] == "co2":
                        print("co2")
                        openfoam_var["z"] = -openfoam_var["z"]
                        
                    for key, scale in zip(modulusVarNames, scales):
                        openfoam_var[key] /= scale
                    
                   

                    parameters = file_path.split("_")[1].split(".")[0].split("-")
                    dataParameterRanges = {
                        holeRadius1: (float(parameters[1]),"mm"),  
                        holeRadius2: (float(parameters[0]), "mm"),
                        inletVelocity1: (float(parameters[6]), "m/s"),
                        inletVelocity2: (float(parameters[7]), "m/s"),
                        holePosX1: (float(parameters[4]), "mm"),
                        holePosX2: (-float(parameters[5]), "mm"),
                        holePosZ1: (float(parameters[2])-20, "mm"),
                        holePosZ2: (float(parameters[3])-20, "mm"),
                    }
                    specificParas = parameterRangeContainer(dataParameterRanges)
                    specificParas.nondim(channelNonDim)
                    parameterRanges = specificParas.maxval()
                    
                    openfoam_var.update({"holeRadius1": np.full_like(openfoam_var["x"], parameterRanges[holeRadius1])})
                    openfoam_var.update({"holeRadius2": np.full_like(openfoam_var["x"], parameterRanges[holeRadius2])})
                    openfoam_var.update({"inletVelocity1": np.full_like(openfoam_var["x"], parameterRanges[inletVelocity1])})
                    openfoam_var.update({"inletVelocity2": np.full_like(openfoam_var["x"], parameterRanges[inletVelocity2])})
                    openfoam_var.update({"holePosX1": np.full_like(openfoam_var["x"], parameterRanges[holePosX1])})
                    openfoam_var.update({"holePosX2": np.full_like(openfoam_var["x"], parameterRanges[holePosX2])})
                    openfoam_var.update({"holePosZ1": np.full_like(openfoam_var["x"], parameterRanges[holePosZ1])})
                    openfoam_var.update({"holePosZ2": np.full_like(openfoam_var["x"], parameterRanges[holePosZ2])})

                    criteria = {"x":(-0.4,  0.4)}
                    invarKeys = ["x", "y", "z", "holeRadius1", "inletVelocity1", "holeRadius2", "inletVelocity2", "holePosX1", "holePosX2", "holePosZ1", "holePosZ2"]
                    outvarKeys = modulusVarNames[:-3]

                    openfoam_invar_numpy = {
                        key: value
                        for key, value in openfoam_var.items()
                        if key in invarKeys

                    } 

                    openfoam_outvar_numpy = {
                        key: value for key, value in openfoam_var.items() if key in outvarKeys
                    }

                    for critKey in criteria.keys():
                        for outvKey in outvarKeys:
                                openfoam_outvar_numpy[outvKey] = openfoam_outvar_numpy[outvKey][(openfoam_invar_numpy[critKey]>criteria[critKey][0]) & (openfoam_invar_numpy[critKey]<criteria[critKey][1])][np.newaxis].T
                        for invKey in invarKeys:
                            if invKey != critKey:
                                openfoam_invar_numpy[invKey] = openfoam_invar_numpy[invKey][(openfoam_invar_numpy[critKey]>criteria[critKey][0]) & (openfoam_invar_numpy[critKey]<criteria[critKey][1])][np.newaxis].T
                        
                        openfoam_invar_numpy[critKey] = openfoam_invar_numpy[critKey][(openfoam_invar_numpy[critKey]>criteria[critKey][0]) & (openfoam_invar_numpy[critKey]<criteria[critKey][1])][np.newaxis].T


                    # print(openfoam_var['x'].size)
                    dataConstraint = PointwiseConstraint.from_numpy(
                        nodes=thermal_nodes, 
                        invar=openfoam_invar_numpy, 
                        outvar=openfoam_outvar_numpy, 
                        batch_size=int(openfoam_invar_numpy['x'].size/batches),
                        # lambda_weighting={"u": 0.1, "v": 0.1, "w": 0.1}
                        )
                    return dataConstraint
                else:
                    print("Missing Data: ", file_path)
            
            # ansysVarNames = ("Temperature [ K ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
            # scales = (temp_scale_ch.magnitude * 10, length_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude)
            # modulusVarNames = ("theta_f_data", "x", "y", "z")
            # batches = 1000

            # for root, dirs, files in os.walk(to_absolute_path("./ansys/constraints")):
            #     for name in files:
            #         print(os.path.join(root, name))
            #         file_path = str(os.path.join(root, name))
            #         thermal_domain.add_constraint(ansysDataConstraint(file_path, ansysVarNames, modulusVarNames, scales, batches), name)

            # ansysVarNames = ("Temperature [ K ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
            # scales = (temp_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude)
            # modulusVarNames = ("theta_fco", "x", "y", "z")
            # batches = 10

            # for root, dirs, files in os.walk(to_absolute_path("./ansys/constraints/co1")):
            #     for name in files:
            #         print(os.path.join(root, name))
            #         file_path = str(os.path.join(root, name))
            #         thermal_domain.add_constraint(ansysDataConstraint(file_path, ansysVarNames, modulusVarNames, scales, batches), name)

            # ansysVarNames = ("Temperature [ K ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
            # scales = (temp_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude)
            # modulusVarNames = ("theta_fco2", "x", "y", "z")
            # batches = 10

            # for root, dirs, files in os.walk(to_absolute_path("./ansys/constraints/co2")):
            #     for name in files:
            #         print(os.path.join(root, name))
            #         file_path = str(os.path.join(root, name))
            #         thermal_domain.add_con4 layers of size 256  straint(ansysDataConstraint(file_path, ansysVarNames, modulusVarNames, scales, batches), name)

        # --------------------------------------------------------------------Cooling Specific Thermal Contstraints--------------------------------------------------------------------    
        if cfg.run_mode == "train" and not cfg.custom.dataOnly:
            
            if cfg.custom.optimizeTfco1: 
                coolingInlet = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.coolingInlet,
                    outvar={"theta_fco": nd_cooling_inlet_temp},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.inlet),
                    criteria=finInteriorCriteria,
                    parameterization=geo.pr,
                    lambda_weighting= {"theta_fco": 100},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(coolingInlet, "coolingInlet")

                # cooling outlet 1
                coolingOutlet = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.coolingOutlet,
                    outvar={"normal_gradient_theta_fco": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.outlet),
                    lambda_weighting={"normal_gradient_theta_fco": 1}, 
                    criteria=finInteriorCriteria,
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(coolingOutlet, "coolingOutlet")

                # cooling inlet walls insulating 1
                coolingInletWall = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finInteriorCriteriaGeo,
                    outvar={"normal_gradient_theta_fco": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.solidWall/2),
                    criteria=And(StrictLessThan(y, geo.nd_channel_origin[1]), StrictGreaterThan(y, geo.coolingInteriorBounds[1][0])),
                    lambda_weighting={"normal_gradient_theta_fco": 1},
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(coolingInletWall, "coolingInletWall")

                # Cooling Hole Interior 1
                #lambdaW = 0.5*tanh(100 * Symbol("sdf"))
                fin_interior = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finInterior,
                    outvar={"advection_diffusion_theta_fco": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.holeInteriorHR),
                    lambda_weighting={"advection_diffusion_theta_fco": cfg.custom.adDifCo},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(fin_interior, "finInterior")

            if cfg.custom.optimizeTfco2:
                # Cooling Hole 2
                # cooling inlet 2
                coolingInlet2 = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.coolingInlet2,
                    outvar={"theta_fco2": nd_cooling_inlet_temp},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.inlet),
                    criteria=finInteriorCriteria2,
                    parameterization=geo.pr,
                    lambda_weighting= {"theta_fco2": 100},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(coolingInlet2, "coolingInlet2")

                # cooling outlet 2
                coolingOutlet2 = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.coolingOutlet2,
                    outvar={"normal_gradient_theta_fco2": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.outlet),
                    lambda_weighting={"normal_gradient_theta_fco2": 1},  # weight zero on edges
                    criteria=finInteriorCriteria2,
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(coolingOutlet2, "coolingOutlet2")

                # Cooling inlet walls insulating 2
                coolingInletWall2 = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finInteriorCriteriaGeo2,
                    outvar={"normal_gradient_theta_fco2": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.solidWall/2),
                    criteria=And(StrictLessThan(y, geo.nd_channel_origin[1]), StrictGreaterThan(y, geo.coolingInteriorBounds[1][0])),
                    lambda_weighting={"normal_gradient_theta_fco2": 1},
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(coolingInletWall2, "coolingInletWall2")

                # Cooling Hole Interior 2
                fin_interior2 = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finInterior2,
                    outvar={"advection_diffusion_theta_fco2": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.holeInteriorHR),
                    lambda_weighting={"advection_diffusion_theta_fco2": cfg.custom.adDifCo},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(fin_interior2, "finInterior2")
            
            # --------------------------------------------------------------------Cooling Specific Thermal Contstraints--------------------------------------------------------------------    

            #-------------------------------------------------------------------- Channel Specific Constraints--------------------------------------------------------------------
            if cfg.custom.optimizeTf:
                # channel inlet
                constraint_inlet = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelInlet,
                    outvar={"theta_f": nd_channel_inlet_temp},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.inlet),
                    lambda_weighting={"theta_f": 100},  # weight zero on edges
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(constraint_inlet, "inlet")

                # channel outlet
                constraint_outlet = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelOutlet,
                    outvar={"normal_gradient_theta_f": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.outlet),
                    lambda_weighting={"normal_gradient_theta_f": 1},  
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(constraint_outlet, "outlet")

                # flow interior low res upstream
                lr_flow_interior = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelInterior,
                    outvar={"advection_diffusion_theta_f": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.lr_interior_f*1/3),
                    criteria=And(LessThan(x, geo.solidInteriorBounds[0][0]), Not(geo.hrBoundsLarge)),
                    lambda_weighting={"advection_diffusion_theta_f": cfg.custom.adDifCh * 2 * cfg.custom.globalLambdaTf},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(lr_flow_interior, "lr_flow_interior")

                 # flow interior low res downstream
                lr_flow_interior = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelInterior,
                    outvar={"advection_diffusion_theta_f": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.lr_interior_f*2/3),
                    criteria=And(Not(geo.hrBoundsLarge), GreaterThan(x, geo.solidInteriorBounds[0][0])),
                    lambda_weighting={"advection_diffusion_theta_f": cfg.custom.adDifCh * 0.5 * cfg.custom.globalLambdaTf}, # @ All 270
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(lr_flow_interior, "lr_flow_interior")

                # flow interiror high res near three fin
                hr_flow_interior = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelInterior,
                    outvar={"advection_diffusion_theta_f": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.hr_interior_f),
                    criteria=geo.hrBoundsLarge,
                    lambda_weighting={"advection_diffusion_theta_f": cfg.custom.adDifCh * cfg.custom.globalLambdaTs}, # @ All 270
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(hr_flow_interior, "hr_flow_interior")
                
                # channel walls insulating
                channel_walls = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelExternalWall,
                    outvar={"normal_gradient_theta_f": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.channelWall),
                    lambda_weighting={"normal_gradient_theta_f": 1},
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(channel_walls, "channel_walls")

            #-------------------------------------------------------------------- Channel Specific Constraints--------------------------------------------------------------------

            # --------------------------------------------------------------------Solid Specifc Thermal Constraints--------------------------------------------------------------------
            if cfg.custom.optimizeTs:
                 # solid interior
                solid_interior = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.solidInterior,
                    outvar={"diffusion_theta_s": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.interiorSolid/4),
                    criteria=solidCriteria,
                    lambda_weighting={"diffusion_theta_s": cfg.custom.solidDiff * cfg.custom.globalLambdaTs,
                                      },
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(solid_interior, "solid_interior")

                solid_interiorH1 = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finExterior1,
                    outvar={"diffusion_theta_s": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.interiorSolid*3/8),
                    criteria=And(Not(geo.solidExteriorCriteriaXZ), And(StrictGreaterThan(y,geo.solidInteriorBounds[1][0]), StrictLessThan(y,geo.solidInteriorBounds[1][1]))),
                    lambda_weighting={"diffusion_theta_s": cfg.custom.solidDiff * 1.5 * cfg.custom.globalLambdaTs},
                    # lambda_weighting={"diffusion_theta_s": (500 + 500*tanh(100 * Symbol("sdf"))) * cfg.custom.globalLambdaTs},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(solid_interiorH1, "solid_interiorH1")

                solid_interiorH2 = PointwiseInteriorConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finExterior2,
                    outvar={"diffusion_theta_s": 0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.interiorSolid*3/8),
                    criteria=And(Not(geo.solidExteriorCriteriaXZ), And(StrictGreaterThan(y,geo.solidInteriorBounds[1][0]), StrictLessThan(y,geo.solidInteriorBounds[1][1]))),
                    lambda_weighting={"diffusion_theta_s": cfg.custom.solidDiff * 1.5 * cfg.custom.globalLambdaTs},
                    # lambda_weighting={"diffusion_theta_s": (500 + 500*tanh(100 * Symbol("sdf"))) * cfg.custom.globalLambdaTs},
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    parameterization=geo.pr,
                    quasirandom=False
                )
                thermal_domain.add_constraint(solid_interiorH2, "solid_interiorH2")

                # Solid Exterior walls insulating
                solid_walls = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.solidExteriorWall,
                    outvar={"normal_gradient_theta_s": 0.0},
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.solidWall),
                    criteria=solidExteriorWallCriteria,
                    lambda_weighting={"normal_gradient_theta_s": 1 * cfg.custom.globalLambdaTs},
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(solid_walls, "solid_walls")

                if cfg.custom.useHTC:
                    # # Solid Ambient Interface
                    fluid_solid_interface = PointwiseBoundaryConstraint(
                        nodes=thermal_nodes,
                        geometry=geo.channelSolidInterface,
                        outvar={
                            "diffusion_interface_HTC_theta_s": 0,
                            "theta_s": 0.95,
                        },
                        batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.channelInterface),
                        lambda_weighting={"diffusion_interface_HTC_theta_s": 0.01 * cfg.custom.globalLambdaTs,
                                          "theta_s": 1 * cfg.custom.globalLambdaTs},
                        parameterization=geo.pr,
                        batch_per_epoch=cfg.batch_size.batchPerEpoch,
                        quasirandom=False
                    )
                    thermal_domain.add_constraint(fluid_solid_interface, "fluid_solid_interface")

            # --------------------------------------------------------------------Solid Specifc Thermal Constraints--------------------------------------------------------------------

            # --------------------------------------------------------------------Mixed Constraints--------------------------------------------------------------------
            if not cfg.custom.useHTC and (cfg.custom.optimizeTf or cfg.custom.optimizeTs):
                # Solid Channel Interface
                fluid_solid_interface = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.channelSolidInterface,
                    outvar={
                        "diffusion_interface_dirichlet_theta_f_theta_s": 0,
                        "diffusion_interface_neumann_theta_f_theta_s": 0,
                        # "theta_f": nd_channel_inlet_temp*0.8
                    },
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.channelInterface),
                    lambda_weighting={"diffusion_interface_dirichlet_theta_f_theta_s": cfg.custom.dirichletInterfaceCh * cfg.custom.globalLambdaTs,
                        "diffusion_interface_neumann_theta_f_theta_s": cfg.custom.neumanInterfaceCh * cfg.custom.globalLambdaTs,
                        # "theta_f": 10,
                    },
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(fluid_solid_interface, "fluid_solid_interface")


            if cfg.custom.optimizeTfco1 or cfg.custom.optimizeTfco2 or cfg.custom.optimizeTs:
                # Solid Cooling Interface 1
                fluid_solid_interface_co = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finInteriorCriteriaGeo,
                    outvar={
                        "diffusion_interface_dirichlet_theta_fco_theta_s": 0,
                        "diffusion_interface_neumann_theta_fco_theta_s": 0,
                        # "theta_fco": nd_cooling_inlet_temp*1.2
                    },
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.coolingInterface),
                    criteria=And(GreaterThan(y, geo.nd_channel_origin[1]), LessThan(y, geo.nd_channel_end[1])),
                    lambda_weighting={
                        "diffusion_interface_dirichlet_theta_fco_theta_s": cfg.custom.dirichletInterfaceCo * cfg.custom.globalLambdaTs,
                        "diffusion_interface_neumann_theta_fco_theta_s": cfg.custom.neumanInterfaceCo * cfg.custom.globalLambdaTs,
                        # "theta_fco": 10,
                        },
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(fluid_solid_interface_co, "fluid_solid_interface_co")

                # Solid Cooling Interface 2
                fluid_solid_interface_co2 = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.finInteriorCriteriaGeo2,
                    outvar={
                        "diffusion_interface_dirichlet_theta_fco2_theta_s": 0,
                        "diffusion_interface_neumann_theta_fco2_theta_s": 0,
                        # "theta_fco2": nd_cooling_inlet_temp*1.2
                    },
                    batch_size=int(cfg.batch_size.globalModifier*cfg.batch_size.coolingInterface),
                    criteria=And(GreaterThan(y, geo.nd_channel_origin[1]), LessThan(y, geo.nd_channel_end[1])),
                    lambda_weighting={
                        "diffusion_interface_dirichlet_theta_fco2_theta_s": cfg.custom.dirichletInterfaceCo * cfg.custom.globalLambdaTs,
                        "diffusion_interface_neumann_theta_fco2_theta_s": cfg.custom.neumanInterfaceCo * cfg.custom.globalLambdaTs,
                        # "theta_fco2": 10,
                        },
                    parameterization=geo.pr,
                    batch_per_epoch=cfg.batch_size.batchPerEpoch,
                    quasirandom=False
                )
                thermal_domain.add_constraint(fluid_solid_interface_co2, "fluid_solid_interface_co2")

            # --------------------------------------------------------------------Mixed Constraints--------------------------------------------------------------------

        # --------------------------------------------------------------------Validators--------------------------------------------------------------------
        if cfg.run_mode == "train" or cfg.run_mode == "plot":
            if cfg.run_mode == "plot":
                 # Dummy Constraint to Load Models
                coolingInlet = PointwiseBoundaryConstraint(
                    nodes=thermal_nodes,
                    geometry=geo.coolingInlet,
                    outvar={"theta_fco": 0, "theta_fco2": 0, "theta_f": 0, "theta_s": 0, "u": 0, "u_co": 0, "u_co2": 0},
                    batch_size=1,
                    parameterization=geo.pr,
                    batch_per_epoch=1
                )
                thermal_domain.add_constraint(coolingInlet, "coolingInlet")

            if not cfg.custom.useHTC and cfg.custom.optimizeTf:
                # validator channel
                ansysVarNames = ("Pressure [ Pa ]", "Temperature [ K ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
                modulusVarNames = ("p_scaled", "theta_f_scaled", "u_scaled", "v_scaled", "w_scaled", "x", "y", "z")
                scales = (1, 1, 1, 1, 1, length_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude)

                for root, dirs, files in os.walk(to_absolute_path("./ansys/validators/channel")):
                    for name in files:
                        print(os.path.join(root, name))
                        file_path = str(os.path.join(root, name))
                        thermal_domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames, thermal_nodes, scales, True, channelNonDim), name)

            if cfg.custom.optimizeTs:
                # validator solid
                modif=1
                ansysVarNames = ("Temperature [ K ]", "Temperature.Gradient X [ m^-1 K ]", "Temperature.Gradient Y [ m^-1 K ]", "Temperature.Gradient Z [ m^-1 K ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
                modulusVarNames = ("theta_s_scaled", "theta_s_scaled__x", "theta_s_scaled__y", "theta_s_scaled__z", "x", "y", "z")
                scales = (1, 1, 1, 1, length_scale_ch.magnitude*modif, length_scale_ch.magnitude*modif, length_scale_ch.magnitude*modif)

                for root, dirs, files in os.walk(to_absolute_path("./ansys/validators/solid")):
                    for name in files:
                        print(os.path.join(root, name))
                        file_path = str(os.path.join(root, name))
                        thermal_domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames, thermal_nodes, scales, True, channelNonDim), name)

            if cfg.custom.optimizeTfco1:
            # validator cooling 1
                ansysVarNames = ("Pressure [ Pa ]", "Temperature [ K ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
                modulusVarNames = ("p_co_scaled", "theta_fco_scaled", "u_co_scaled", "v_co_scaled", "w_co_scaled", "x", "y", "z")
                scales = (1, 1, 1, 1, 1, length_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude)

                for root, dirs, files in os.walk(to_absolute_path("./ansys/validators/internal1")):
                    for name in files:
                        print(os.path.join(root, name))
                        file_path = str(os.path.join(root, name))
                        thermal_domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames, thermal_nodes, scales, True, channelNonDim), name)
            
            if cfg.custom.optimizeTfco2:
                # validator cooling 2
                ansysVarNames = ("Pressure [ Pa ]", "Temperature [ K ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
                modulusVarNames = ("p_co2_scaled", "theta_fco2_scaled", "u_co2_scaled", "v_co2_scaled", "w_co2_scaled", "x", "y", "z")
                scales = (1, 1, 1, 1, 1, length_scale_ch.magnitude, length_scale_ch.magnitude, length_scale_ch.magnitude)

                for root, dirs, files in os.walk(to_absolute_path("./ansys/validators/internal2")):
                    for name in files:
                        print(os.path.join(root, name))
                        file_path = str(os.path.join(root, name))
                        thermal_domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames, thermal_nodes, scales, True, channelNonDim), name)


            # --------------------------------------------------------------------Validators--------------------------------------------------------------------

            # # --------------------------------------------------------------------Inferencers--------------------------------------------------------------------
            # outputChannelS = ["u_scaled", "v_scaled", "w_scaled", "p_scaled", "theta_f_scaled"]
            # outputSolidS = ["theta_s_scaled"]
            # outputCooling1S = ["x_st", "y", "z_st", "holeRadius1_s", "inletVelocity1_s", "holePosX1_s", "holePosZ1_s"] #["u_co_scaled", "v_co_scaled", "w_co_scaled", "p_co_scaled", "theta_fco_scaled"]
            # outputCooling2S = ["x_st2", "y", "z_st2", "holeRadius2_s", "inletVelocity2_s", "holePosX2_s", "holePosZ2_s"] # ["u_co2_scaled", "v_co2_scaled", "w_co2_scaled", "p_co2_scaled", "theta_fco2_scaled"]
            
            # outputsNS = []
            # outputsS = []

            # # max
            # # paras = paramRanges_ch.maxval()
            # # print(paras)
            # paramRangs = {
            # holeRadius1: ((5, 5),"mm"),
            # holeRadius2: ((5, 5),"mm"),
            # inletVelocity1: ((15, 15),"m/s"),
            # inletVelocity2: ((15, 15),"m/s"),
            # holePosX1: ((22, 22),"mm"),
            # holePosX2: ((-22, -22),"mm"),
            # holePosZ1: ((8, 8),"mm"),
            # holePosZ2: ((-8, -8),"mm")}

            # specificParas = parameterRangeContainer(paramRangs)
            # specificParas.nondim(channelNonDim)
            # paras=specificParas.minval()
            # nrPoints = cfg.custom.inf_points

            # if not cfg.custom.useHTC and cfg.custom.optimizeTf:
            #     outputsS += outputChannelS
            #     channelInteriorPoints = geo.inferencePlaneXZ.translate((0,0.0001,0)).sample_boundary(nrPoints*2, criteria=geo.channelInfCriteriaHR, parameterization=paras, quasirandom=False)
            #     # channelInteriorPointslr = geo.inferencePlaneXZ.translate((0,0.0002,0)).sample_boundary(int(nrPoints*1.5), criteria=geo.solidExteriorCriteriaXZ, parameterization=paras, quasirandom=False)
            
            #     inferenceChannelInterior = PointwiseInferencer(
            #         nodes=thermal_nodes,
            #         invar=channelInteriorPoints,
            #         output_names=outputsNS + outputsS
            #     )
            #     thermal_domain.add_inferencer(inferenceChannelInterior, "inferenceChannelInteriorMax")
                
            # #     inferenceChannelInteriorlr = PointwiseInferencer(
            # #         nodes=thermal_nodes,
            # #         invar=channelInteriorPointslr,
            # #         output_names=outputsNS + outputsS
            # #     )
            # #     thermal_domain.add_inferencer(inferenceChannelInteriorlr, "inferenceChannelInteriorLr")


            # # if cfg.custom.optimizeTfco1:
            # #     outputsS += outputCooling1S
            # #     coolingInteriorPoints = geo.finInterior.sample_interior(nrPoints,  parameterization=paras, quasirandom=False)

            # #     inferenceCoolingInterior = PointwiseInferencer(
            # #         nodes=thermal_nodes,
            # #         invar=coolingInteriorPoints,
            # #         output_names=outputsNS + outputsS
            # #     )
            # #     thermal_domain.add_inferencer(inferenceCoolingInterior, "inferenceCoolingInteriorMax")
            
            # # if cfg.custom.optimizeTfco2:
            # #     outputsS += outputCooling2S
            # #     coolingInteriorPoints2 = geo.finInterior2.sample_interior(nrPoints, parameterization=paras, quasirandom=False)
            
            # #     inferenceCoolingInterior2 = PointwiseInferencer(
            # #         nodes=thermal_nodes,
            # #         invar=coolingInteriorPoints2,
            # #         output_names=outputsNS + outputsS
            # #     )
            # #     thermal_domain.add_inferencer(inferenceCoolingInterior2, "inferenceCoolingInteriorMax2")

            # if cfg.custom.optimizeTs:
            #     outputsS += outputSolidS
            #     solidInteriorPoints = geo.inferencePlaneXZ.sample_boundary(int(nrPoints*1.5), criteria=solidInteriorCriteria, parameterization=paras, quasirandom=False)
            
            #     inferenceSolidInterior = PointwiseInferencer(
            #         nodes=thermal_nodes,
            #         invar=solidInteriorPoints,
            #         output_names=outputsNS + outputsS
            #     )
            #     thermal_domain.add_inferencer(inferenceSolidInterior, "inferenceSolidInteriorMax")

            # #min
            # # paras = paramRanges_ch.minval()
            # paramRangs = {
            # holeRadius1: (5,"mm"),
            # holeRadius2: (5,"mm"),
            # inletVelocity1: (19,"m/s"),
            # inletVelocity2: (19,"m/s"),
            # holePosX1: (20,"mm"),
            # holePosX2: (-10,"mm"),
            # holePosZ1: (-8,"mm"),
            # holePosZ2: (-6,"mm")}

            # specificParas = parameterRangeContainer(paramRangs)
            # specificParas.nondim(channelNonDim)
            # paras=specificParas.minval()

            # if not cfg.custom.useHTC and cfg.custom.optimizeTf:
            #     outputsS += outputChannelS
            #     channelInteriorPoints = geo.inferencePlaneXZ.translate((0,0.00015,0)).sample_boundary(nrPoints*2, criteria=geo.channelInfCriteriaHR, parameterization=paras, quasirandom=False)

            #     inferenceChannelInteriorMin = PointwiseInferencer(
            #         nodes=thermal_nodes,
            #         invar=channelInteriorPoints,
            #         output_names=outputsNS + outputsS
            #     )
            #     thermal_domain.add_inferencer(inferenceChannelInteriorMin, "inferenceChannelInteriorMin")

            # # if cfg.custom.optimizeTfco1:
            # #     outputsS += outputCooling1S
            # #     coolingInteriorPoints = geo.inferensePlaneYZ1.sample_boundary(nrPoints,  parameterization=paras, quasirandom=False)

            # #     inferenceCoolingInteriorMin = PointwiseInferencer(
            # #         nodes=thermal_nodes,
            # #         invar=coolingInteriorPoints,
            # #         output_names=outputsNS + outputsS
            # #     )
            # #     thermal_domain.add_inferencer(inferenceCoolingInteriorMin, "inferenceCoolingInteriorMin")
            
            # # if cfg.custom.optimizeTfco2:
            # #     outputsS += outputCooling2S
            # #     coolingInteriorPoints2 = geo.inferensePlaneYZ2.sample_boundary(nrPoints, parameterization=paras, quasirandom=False)
            
            # #     inferenceCoolingInterior2Min = PointwiseInferencer(
            # #         nodes=thermal_nodes,
            # #         invar=coolingInteriorPoints2,
            # #         output_names=outputsNS + outputsS
            # #     )
            # #     thermal_domain.add_inferencer(inferenceCoolingInterior2Min, "inferenceCoolingInterior2Min")

            # if cfg.custom.optimizeTs:
            #     outputsS += outputSolidS
            #     solidInteriorPoints = geo.inferencePlaneXZ.sample_boundary(int(nrPoints*1.5), criteria=solidInteriorCriteria, parameterization=paras, quasirandom=False)
            
            #     inferenceSolidInteriorMin = PointwiseInferencer(
            #         nodes=thermal_nodes,
            #         invar=solidInteriorPoints,
            #         output_names=outputsNS + outputsS
            #     )
            #     thermal_domain.add_inferencer(inferenceSolidInteriorMin, "inferenceSolidInteriorMin")
        
            # --------------------------------------------------------------------Inferencers--------------------------------------------------------------------
            
        # --------------------------------------------------------------------For Design Optimization - Eval Validators, Inferences and Monitors--------------------------------------------------------------------

        if cfg.run_mode == "eval":
            
            # Dummy Constraint to Load Models
            coolingInlet = PointwiseBoundaryConstraint(
                nodes=thermal_nodes,
                geometry=geo.coolingInlet,
                outvar={"theta_fco": 0, "theta_fco2": 0, "theta_f": 0, "theta_s": 0, "u": 0, "u_co": 0, "u_co2": 0},
                batch_size=1,
                parameterization=geo.pr,
                batch_per_epoch=1
            )
            thermal_domain.add_constraint(coolingInlet, "coolingInlet")

            # define candidate designs
            optPoints = cfg.custom.optPoints

            output_names_flow=["u_scaled", "v_scaled", "w_scaled", "p_scaled",
                                        "u_co_scaled", "v_co_scaled", "w_co_scaled", "p_co_scaled",
                                        "u_co2_scaled", "v_co2_scaled", "w_co2_scaled", "p_co2_scaled"]
            
            # output_names_thermalChannel=["theta_f_scaled", "theta_f_scaled_x", "theta_f_scaled_y", "theta_f_scaled_z"]
            
            output_names_thermalCooling=["theta_fco_scaled", "theta_fco2_scaled"]

            output_names_thermalSolid=["theta_s_scaled"]
            output_names_thermalSolidGrad=["theta_s_scaled__x", "theta_s_scaled__y", "theta_s_scaled__z"]

            for design in enumerate(designs):

                specific_param_ranges_ns = {
                    holeRadius1:    ((design[1][0]), "mm"),  
                    holeRadius2:    ((design[1][1]), "mm"),
                    inletVelocity1: ((design[1][2]), "m/s"),
                    inletVelocity2: ((design[1][3]), "m/s"),
                    holePosX1:      ((design[1][4]), "mm"),
                    holePosX2:      ((design[1][5]), "mm"),
                    holePosZ1:      ((design[1][6]), "mm"),
                    holePosZ2:      ((design[1][7]), "mm"),
                }

                specificParas = parameterRangeContainer(specific_param_ranges_ns)
                specificParas.nondim(channelNonDim)
                specific_param_ranges=specificParas.minval()
                        
                # cooling
                # coolingInteriorPoints1 = geo.finInterior.sample_interior(optPoints, parameterization=specific_param_ranges)
                # coolingInteriorPoints2 = geo.finInterior2.sample_interior(optPoints, parameterization=specific_param_ranges)
                coolingInletPoints1 = geo.coolingInlet.sample_boundary(int(optPoints/10), criteria=finInteriorCriteria, parameterization=specific_param_ranges)
                coolingInletPoints2 = geo.coolingInlet2.sample_boundary(int(optPoints/10), criteria=finInteriorCriteria2, parameterization=specific_param_ranges)
                
                # inf
                # metric1 = "coolingInterior1_design_" + str(design[0])
                # inferenceCoolingInterior1 = PointwiseInferencer(
                #     nodes=thermal_nodes,
                #     invar=coolingInteriorPoints1,
                #     output_names=output_names_flow + output_names_thermalCooling,
                # )
                # thermal_domain.add_inferencer(inferenceCoolingInterior1, metric1)

                # metric2 = "coolingInterior2_design_" + str(design[0])
                # inferenceCoolingInterior2 = PointwiseInferencer(
                #     nodes=thermal_nodes,
                #     invar=coolingInteriorPoints2,
                #     output_names=output_names_flow + output_names_thermalCooling,
                # )
                # thermal_domain.add_inferencer(inferenceCoolingInterior2, metric2)

                # mon
                # metric5 = "coolingInletPressure1_design_" + str(design[0])
                # coolingInletPressure1 = PointwiseMonitor(
                #     coolingInletPoints1,
                #     output_names=output_names_flow,
                #     metrics={metric5: lambda var: torch.mean(var["p_co_scaled"])},
                #     nodes=thermal_nodes,
                # )
                # thermal_domain.add_monitor(coolingInletPressure1)

                # metric6 = "coolingInletPressure2_design_" + str(design[0])
                # coolingInletPressure2 = PointwiseMonitor(
                #     coolingInletPoints2,
                #     output_names=output_names_flow,
                #     metrics={metric6: lambda var: torch.mean(var["p_co2_scaled"])},
                #     nodes=thermal_nodes,
                # )
                # thermal_domain.add_monitor(coolingInletPressure2)

                # cooling_inlet_flow1
                metric7 = "coolingInletFlow1_design_" + str(design[0])
                coolingInletFlow1 = PointwiseMonitor(
                    coolingInletPoints1,
                    output_names=["normal_dot_vel_scaled_co1", "area_scaled"],
                    metrics={metric7: lambda var: torch.sum(var["area_scaled"] * -var["normal_dot_vel_scaled_co1"])},
                    nodes=thermal_nodes,
                )
                thermal_domain.add_monitor(coolingInletFlow1)

                # cooling_inlet_flow2
                metric8 = "coolingInletFlow2_design_" + str(design[0])
                coolingInletFlow1 = PointwiseMonitor(
                    coolingInletPoints2,
                    output_names=["normal_dot_vel_scaled_co2", "area_scaled"],
                    metrics={metric8: lambda var: torch.sum(var["area_scaled"] * -var["normal_dot_vel_scaled_co2"])},
                    nodes=thermal_nodes,
                )
                thermal_domain.add_monitor(coolingInletFlow1)


                ## -----------------------------------------solid-----------------------------------------------
                solidInteriorPointsGrad = geo.solidNearHoles.sample_interior(
                    nr_points=optPoints, 
                    parameterization=specific_param_ranges,
                    criteria=And(GreaterThan(y, geo.solidInteriorBounds[1][0]), LessThan(y, 0))
                    )

                solidInteriorPoints = geo.solidInteriorCsg.sample_interior(
                    nr_points=optPoints, 
                    parameterization=specific_param_ranges,
                    criteria=solidExteriorWallCriteria
                    )
                
                # # inf
                # metric3 = "solidInterior_design_" + str(design[0])
                # inferenceSolidInterior = PointwiseInferencer(
                #     nodes=thermal_nodes,
                #     invar=solidInteriorPoints,
                #     output_names=output_names_thermalSolidGrad,
                #     requires_grad=True
                # )
                # thermal_domain.add_inferencer(inferenceSolidInterior, metric3)

                # # mon
                # metric31 = "solidMaxTemp_design_" + str(design[0])
                # peak_temp_monitor = PointwiseMonitor(
                #     invar= solidInteriorPoints,
                #     output_names=output_names_thermalSolid,
                #     metrics={metric31: lambda var: torch.max(var["theta_s_scaled"])},
                #     nodes=thermal_nodes,
                # )
                # thermal_domain.add_monitor(peak_temp_monitor)

                metric4 = "solidAveTemp_design_" + str(design[0])
                mean_temp_monitor = PointwiseMonitor(
                    invar= solidInteriorPoints,
                    output_names=output_names_thermalSolid,
                    metrics={metric4: lambda var: torch.mean(var["theta_s_scaled"])},
                    nodes=thermal_nodes,
                )
                thermal_domain.add_monitor(mean_temp_monitor)

                metric9 = "solidMaxTempGrad_design_" + str(design[0])
                peak_tempGrad_monitor = PointwiseMonitor(
                    invar= solidInteriorPointsGrad,
                    output_names=output_names_thermalSolidGrad,
                    requires_grad=True,
                    # metrics={metric9: lambda var: torch.max((var["theta_s_scaled__x"]**2 + var["theta_s_scaled__y"]**2 + var["theta_s_scaled__z"]**2)**(1/2))},
                    metrics={metric9: lambda var: torch.mean(torch.topk(input=((var["theta_s_scaled__x"]**2 + var["theta_s_scaled__y"]**2 + var["theta_s_scaled__z"]**2)**(1/2)), k=60, dim=0, largest=True, sorted=False).values)},
                    nodes=thermal_nodes,
                )
                thermal_domain.add_monitor(peak_tempGrad_monitor)
                

                # metric10 = "solidAveTempGrad_design_" + str(design[0])
                # mean_tempGrad_monitor = PointwiseMonitor(
                #     invar= solidInteriorPoints,
                #     output_names=output_names_thermalSolidGrad,
                #     requires_grad=True,
                #     metrics={metric10: lambda var: torch.mean((var["theta_s_scaled__x"]**2 + var["theta_s_scaled__y"]**2 + var["theta_s_scaled__z"]**2)**(1/2))},
                #     nodes=thermal_nodes,
                # )
                # thermal_domain.add_monitor(mean_tempGrad_monitor)

        # --------------------------------------------------------------------Eval Inferences and Monitors--------------------------------------------------------------------
        
        # --------------------------------------------------------------------Solver--------------------------------------------------------------------

        thermal_slv = Solver(cfg, thermal_domain)

        if cfg.run_mode == "train":
            # start thermal solver
            thermal_slv.solve()

        if cfg.run_mode == "eval" or cfg.run_mode == "plot":
            # start thermal solver
            thermal_slv.eval()
    run()
if __name__ == "__main__": 
    modHeat()
    # designs = [[3.9682423312465, 4.347807349228687, 11.44440994112209, 11.199246500509634, 13.511303693551328, -14.94396743984046, -0.4866958623154112, 0.17457115471716378],
    #            [3.76690079396928, 4.172218306487319, 11.88943877288943, 12.847620209601896, 12.971872261596083, -13.34492621986694, -0.2508390890740806, -2.2745349133730635],
    #            [3.44527238348337, 4.366404264075123, 11.97142738334767, 13.046668201927961, 12.947045610414397, -12.94595616907899, -1.2407294745890098, -3.3269627747960975]]
    # modHeat(designs=designs)