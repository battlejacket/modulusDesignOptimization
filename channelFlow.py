import sys
import torch
from sympy import Symbol, tanh, Not
import numpy as np
import os

import modulus
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from csv_rw_skiprows import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.non_dim import Scaler
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.siren import SirenArch
from modulus.sym.models.modified_fourier_net import ModifiedFourierNetArch
from modulus.sym.models.dgm import DGMArch



from channelProperties import *
from geometry import dLightGeo


@modulus.sym.main(config_path="conf", config_name="conf_channelFlow")
def run(cfg: ModulusConfig) -> None:

    geo = dLightGeo(paramRanges_ch)

    # make navier stokes equations
    if cfg.custom.turbulent:
        ze = ZeroEquation(nu=nd_fluid_kinematic_viscosity_ch, dim=3, time=False, max_distance=0.5) #nu default 0.002
        ns = NavierStokes(nu=ze.equations["nu"], rho=nd_fluid_density_ch, dim=3, time=False)
        navier_stokes_nodes = ns.make_nodes() + ze.make_nodes()
    else:
        ns = NavierStokes(nu=nd_fluid_kinematic_viscosity_ch, rho=nd_fluid_density_ch, dim=3, time=False) #nu default 0.01
        navier_stokes_nodes = ns.make_nodes()
    
    normal_dot_vel = NormalDotVec()

    equation_nodes = navier_stokes_nodes + normal_dot_vel.make_nodes()

    # determine inputs outputs of the network
    
    input_keys = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

    # select the network and the specific configs
    if cfg.custom.arch == "FullyConnectedArch":
        flow_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "FourierNetArch":
        flow_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )

    elif cfg.custom.arch == "SirenArch":
        flow_net = SirenArch(
            input_keys=input_keys,
            output_keys=output_keys,
            normalization={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (-2.5, 2.5)},
        )
    elif cfg.custom.arch == "ModifiedFourierNetArch":
        flow_net = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "DGMArch":
        flow_net = DGMArch(
            input_keys=input_keys,
            output_keys=output_keys,
            layer_size=128,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )

    # make list of nodes to unroll graph on
    equation_nodes += [flow_net.make_node(name="flow_network")]
    # Scaler (de-normalize)
    equation_nodes += Scaler(
        ["u", "v", "w", "p"],
        ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
        ["m/s", "m/s", "m/s", "pascal"],
        channelNonDim).make_node()
    equation_nodes += Scaler(
        ["normal_dot_vel", "area"],
        ["normal_dot_vel_scaled", "area_scaled"],
        ["m/s", "m^2"],
        channelNonDim).make_node()
    
    flow_nodes = equation_nodes
    # make flow domain
    flow_domain = Domain()


    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.channelInterior.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)
    # inlet velocity profile

    if cfg.run_mode == "train" and (cfg.custom.includeData or cfg.custom.dataOnly):

        def ansysDataConstraint(file_path, ansysVarNames, modulusVarNames, batches):
            if os.path.exists(to_absolute_path(file_path)):
                mapping = {}
                for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
                    mapping[ansVarName] = modulusVarName

                openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping, skiprows=6)
                openfoam_var["x"] /= length_scale_ch.magnitude
                openfoam_var["y"] /= length_scale_ch.magnitude
                openfoam_var["z"] /= length_scale_ch.magnitude
                openfoam_var["p"] /= (density_scale_ch.magnitude*velocity_scale_ch.magnitude**2)
                openfoam_var["u"] /= velocity_scale_ch.magnitude
                openfoam_var["v"] /= velocity_scale_ch.magnitude
                openfoam_var["w"] /= velocity_scale_ch.magnitude

                openfoam_invar_numpy = {
                    key: value
                    for key, value in openfoam_var.items()
                    if key in ["x", "y", "z"]
                }    
                        
                openfoam_outvar_numpy = {
                    key: value for key, value in openfoam_var.items() if key not in ["x", "y", "z"]
                }
                # print(openfoam_var['x'].size)
                dataConstraint = PointwiseConstraint.from_numpy(
                    nodes=flow_nodes, 
                    invar=openfoam_invar_numpy, 
                    outvar=openfoam_outvar_numpy, 
                    batch_size=int(openfoam_var['x'].size/batches),
                    )
                return dataConstraint
            
        ansysVarNames = ("X [ m ]", "Y [ m ]", "Z [ m ]", "Pressure [ Pa ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]", )
        modulusVarNames = ("x", "y", "z", "p", "u", "v", "w")

        file_path = "ansys/channelFM.csv"
        flow_domain.add_constraint(ansysDataConstraint(file_path, ansysVarNames, modulusVarNames, 50), "channelFull")


    if cfg.run_mode == "train" and not cfg.custom.dataOnly:
        # channel inlet
        # # u_profile = nd_channel_inlet_velocity * tanh((geo.channelInletBounds[1][0] - Abs(y - geo.channelInletCenter[1])) / 0.02) * tanh((geo.channelInletBounds[2][0] - Abs(z - geo.channelInletCenter[2])) / 0.02)
        # inlet = PointwiseBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.channelInlet,
        #     outvar={"u": nd_channel_inlet_velocity, "v": nd_noslip_velocity_ch, "w": nd_noslip_velocity_ch},
        #     batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.inlet),
        #     lambda_weighting={"u": 10*tanh((geo.channelInletBounds[1][0] - Abs(y - geo.channelInletCenter[1])) / 0.02) * tanh((geo.channelInletBounds[2][0] - Abs(z - geo.channelInletCenter[2])) / 0.02), "v": 1.0, "w": 1.0},
        #     #parameterization=geo.pr,
        #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
        #     #importance_measure=importance_measure,
        # )
        # flow_domain.add_constraint(inlet, "channelInlet")

        # #channel outlet
        # outlet = PointwiseBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.channelOutlet,
        #     outvar={"p": nd_channel_outlet_p},
        #     batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.outlet),
        #     #parameterization=geo.pr,
        #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
        #     #importance_measure=importance_measure,
        # )
        # flow_domain.add_constraint(outlet, "channelOutlet")
        
        # # no slip channel LR
        # no_slip = PointwiseBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.channelNoSlip,
        #     outvar={"u": nd_noslip_velocity_ch, "v": nd_noslip_velocity_ch, "w": nd_noslip_velocity_ch},
        #     batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.channelNoSlip),
        #     lambda_weighting={"u": 5, "v": 5, "w": 5},
        #     criteria=geo.lrBounds,
        #     #parameterization=geo.pr,
        #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
        #     #importance_measure=importance_measure,
        # )
        # flow_domain.add_constraint(no_slip, "noSlipCh")

        # # no slip channel HR
        # no_slipS = PointwiseBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.channelNoSlip,
        #     outvar={"u": nd_noslip_velocity_ch, "v": nd_noslip_velocity_ch, "w": nd_noslip_velocity_ch},
        #     batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.solidNoSlip),
        #     lambda_weighting={"u": 5, "v": 5, "w": 5},
        #     criteria=Not(geo.lrBounds),
        #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
        #     #importance_measure=importance_measure,
        # )
        # flow_domain.add_constraint(no_slipS, "noSlipS")

        # interior channel low res
        lr_interior = PointwiseInteriorConstraint(
            nodes=flow_nodes,
            geometry=geo.channelInterior,
            outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
            batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.channelInteriorLR),
            compute_sdf_derivatives=True,
            lambda_weighting={
                "continuity": tanh(5 * Symbol("sdf")),
                "momentum_x": tanh(5 * Symbol("sdf")),
                "momentum_y": tanh(5 * Symbol("sdf")),
                "momentum_z": tanh(5 * Symbol("sdf")),
            },
            criteria=Not(geo.hrBoundsFlow),
            batch_per_epoch=cfg.batch_size.batchPerEpoch,
            #importance_measure=importance_measure,
            #parameterization=geo.pr
        )
        flow_domain.add_constraint(lr_interior, "channelInteriorLR")

        # interior channel hi res
        hr_interior = PointwiseInteriorConstraint(
            nodes=flow_nodes,
            geometry=geo.channelInterior,
            outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
            batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.channelInteriorHR),
            compute_sdf_derivatives=True,
            lambda_weighting={
                "continuity": tanh(5 * Symbol("sdf")),
                "momentum_x": tanh(5 * Symbol("sdf")),
                "momentum_y": tanh(5 * Symbol("sdf")),
                "momentum_z": tanh(5 * Symbol("sdf")),
            },
            criteria=geo.hrBoundsFlow,
            batch_per_epoch=cfg.batch_size.batchPerEpoch,
            #importance_measure=importance_measure,
            #parameterization=geo.pr
        )
        flow_domain.add_constraint(hr_interior, "channelInteriorHR")        

        # # integral continuity
        # integral_continuity = IntegralBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.channelIntegralPlane,
        #     outvar={"normal_dot_vel": geo.channelInletArea * nd_channel_inlet_velocity},
        #     batch_size=int(cfg.batch_size.num_integral_continuity),
        #     integral_batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.integral_continuity),
        #     lambda_weighting={"normal_dot_vel": 1.0},
        #     parameterization={**geo.pr, **geo.channelIntegralRange},
        #     criteria=geo.solidExteriorCriteriaXZ,
        #     fixed_dataset=False
        # )
        # flow_domain.add_constraint(integral_continuity, "channelIntegralContinuity")

        # # Symmety XZ
        # symmetryXZ = PointwiseBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.symmetryXZ,
        #     outvar={"v": 0, "u__y": 0, "w__y": 0, "p__y": 0},
        #     batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.symmetry),
        # )
        # flow_domain.add_constraint(symmetryXZ, "symmetryXY")

        # # Symmety XY
        # symmetryXY = PointwiseBoundaryConstraint(
        #     nodes=flow_nodes,
        #     geometry=geo.symmetryXY,
        #     outvar={"w": 0, "u__z": 0, "v__z": 0, "p__z": 0},
        #     batch_size=int(cfg.batch_size.channelGlobalModifier*cfg.batch_size.symmetry),
        # )
        # flow_domain.add_constraint(symmetryXY, "symmetryXZ")


    # Validators

    def ansysValidator(file_path, ansysVarNames, modulusVarNames):
        if os.path.exists(to_absolute_path(file_path)):
            mapping = {}
            for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
                mapping[ansVarName] = modulusVarName

            openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping, skiprows=6)
            openfoam_var["x"] /= length_scale_ch.magnitude
            openfoam_var["y"] /= length_scale_ch.magnitude
            openfoam_var["z"] /= length_scale_ch.magnitude

            openfoam_invar_numpy = {
                key: value
                for key, value in openfoam_var.items()
                if key in ["x", "y", "z", "holeRadius1", "inletVelocity1", "holeRadius2", "inletVelocity2", "holePosX1", "holePosX2", "holePosZ1", "holePosZ2"]
            }    
                    
            openfoam_outvar_numpy = {
                key: value for key, value in openfoam_var.items() if key not in ["x", "y", "z"]
            }

            openfoam_validator = PointwiseValidator(
                nodes=flow_nodes,
                invar=openfoam_invar_numpy,
                true_outvar=openfoam_outvar_numpy,
                batch_size=3000,
                # plotter=CustomValidatorPlotter(),
                # requires_grad=True,
            )
            return openfoam_validator
        else:
            print("Missing File - Validator")

    # add validator channel
    file_path = "ansys/channelXZ.csv"
    ansysVarNames = ("Pressure [ Pa ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]", "X [ m ]", "Y [ m ]", "Z [ m ]")
    modulusVarNames = ("p_scaled", "u_scaled", "v_scaled", "w_scaled", "x", "y", "z")
    flow_domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames), "channel")
    
    # file_path = "ansys/channelYZ.csv"
    # flow_domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames), "channelYZ")

    # inferencers
    outputChannelS = ["u_scaled", "v_scaled", "w_scaled", "p_scaled", "p"]

    nrPoints = 10000

    outputsS = outputChannelS
    channelInteriorPoints = geo.inferencePlaneXZ.translate((0,0.0001,0)).sample_boundary(nrPoints, quasirandom=False)
    # channelInteriorPointslr = geo.inferencePlaneXZ.translate((0,0.0002,0)).sample_boundary(int(nrPoints*1.5), criteria=geo.solidExteriorCriteriaXZ, parameterization=paras, quasirandom=False)

    inferenceChannelInterior = PointwiseInferencer(
        nodes=flow_nodes,
        invar=channelInteriorPoints,
        output_names= outputsS
    )
    flow_domain.add_inferencer(inferenceChannelInterior, "inferenceChannelInteriorMax")

    # add monitor
    paras = paramRanges_ch.minval()
    # paras=geo.pr
    nrPoints = 1024

    # channel_inlet_flow
    channelInletPoints = geo.channelInlet.sample_boundary(
        nrPoints,
        parameterization=paras,
    )
    channelInlet = PointwiseMonitor(
        channelInletPoints,
        output_names=["normal_dot_vel", "area", "p_scaled"],
        metrics={"channel_inlet_flow": lambda var: torch.sum(var["area"] * -var["normal_dot_vel"]), "channel_inlet_pressure": lambda var: torch.mean(var["p_scaled"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(channelInlet)
    # channel outlet flow
    channelOutletPoints = geo.channelOutlet.sample_boundary(
        nrPoints,
        parameterization=paras,
    )
    channelOutlet = PointwiseMonitor(
        channelOutletPoints,
        output_names=["normal_dot_vel", "area", "p_scaled"],
        metrics={"channel_outlet_flow": lambda var: torch.sum(var["area"] * var["normal_dot_vel"]), "channel_outlet_pressure": lambda var: torch.mean(var["p_scaled"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(channelOutlet)

    # make solver
    flow_slv = Solver(cfg, flow_domain)

    # start flow solver
    flow_slv.solve()


if __name__ == "__main__":
    run()
