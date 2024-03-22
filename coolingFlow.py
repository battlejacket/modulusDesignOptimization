import sys
import torch
from sympy import Symbol, tanh, Max
import numpy as np
import os

import modulus
from modulus.sym.hydra import to_absolute_path, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.non_dim import Scaler
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.siren import SirenArch
from modulus.sym.models.modified_fourier_net import ModifiedFourierNetArch
from modulus.sym.models.dgm import DGMArch

from coolingProperties import *
from coolingGeometry import dLightGeoCo


@modulus.sym.main(config_path="conf", config_name="conf_coolingFlow")
def run(cfg: ModulusConfig) -> None:

    geo = dLightGeoCo(paramRanges)



    # make navier stokes equations
    
    ze = ZeroEquation(nu=nd_fluid_kinematic_viscosity, dim=2, time=False, max_distance=holeRadius1) #nu default 0.002
    ns = NavierStokes(nu=ze.equations["nu"], rho=nd_fluid_density, dim=2, time=False)
    navier_stokes_nodes = ns.make_nodes() + ze.make_nodes()
    
    normal_dot_vel = NormalDotVec(["u", "v"])

    equation_nodes = navier_stokes_nodes + normal_dot_vel.make_nodes()

    input_keys = [Key("x_s"), Key("y_s"), Key("holeRadius1"), Key("inletVelocity1")]

    output_keys =  [Key("u_s_co"), Key("v_s_co"), Key("p_s_co")]

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
    equation_nodes += [flow_net.make_node(name="flow_network_co")]
    # Scaler (de-normalize)
    equation_nodes += [
            Node.from_sympy(Symbol("x")*1, "x_s")
        ] + [
            Node.from_sympy(Symbol("y")*1, "y_s")
        ] 
    
    equation_nodes += [
            Node.from_sympy(Symbol("u_s_co")*1, "u")
        ] + [
            Node.from_sympy(Symbol("v_s_co")*1, "v")
        ] + [
            Node.from_sympy(Symbol("p_s_co")*1, "p")
        ]

    equation_nodes += Scaler(
        ["u", "v", "p"],
        ["u_scaled", "v_scaled", "p_scaled"],
        ["m/s", "m/s", "pascal"],
        coolingNonDim).make_node()
    equation_nodes += Scaler(
        ["normal_dot_vel", "area"],
        ["normal_dot_vel_scaled", "area_scaled"],
        ["m/s", "m"],
        coolingNonDim).make_node()
    
    flow_nodes = equation_nodes
    # make flow domain
    flow_domain = Domain()

    if cfg.run_mode == "train":

        # ---------------------------COOLING--------------------------

        def parabolaT(y, radius, max_vel):
            distance = abs(y)
            parabola = max_vel * Max((1 - (distance / radius))**(1/7), 0)
            return parabola

        inlet_parabola = parabolaT(
            y, radius=holeRadius1, max_vel=inletVelocity1
        )

        coolingInlet = PointwiseBoundaryConstraint(
            nodes=flow_nodes,
            geometry=geo.coolingInlet,
            outvar= {"u": inlet_parabola, "v": 0},
            #outvar= {"p": nd_cooling_inlet_p},
            batch_size=int(cfg.batch_size.finGlobalModifier*cfg.batch_size.inlet),
            lambda_weighting={"u": 1, "v": 1},
            # criteria=finInteriorCriteria,
            parameterization=geo.pr,
            batch_per_epoch=cfg.batch_size.batchPerEpoch,
            #importance_measure=importance_measure,
        )
        flow_domain.add_constraint(coolingInlet, "coolingInlet")
        
        # cooling No Slip
        coolingNoSlip = PointwiseBoundaryConstraint(
            nodes=flow_nodes,
            geometry=geo.holeInterior,
            outvar={"u": nd_noslip_velocity, "v": nd_noslip_velocity},
            batch_size=int(cfg.batch_size.finGlobalModifier*cfg.batch_size.coolingNoSlip),
            parameterization=geo.pr,
            batch_per_epoch=cfg.batch_size.batchPerEpoch,
            lambda_weighting={"u": 1, "v": 1},
            # criteria=And(StrictGreaterThan(y, geo.nd_channel_origin[1]+0.0001), StrictLessThan(y, geo.nd_channel_end[1]-0.0001))
            # criteria = finInteriorCriteria
        )
        flow_domain.add_constraint(coolingNoSlip, "coolingNoSlip")

        # flow interior fin high res
        lambdaW = 1*tanh(25 * Symbol("sdf"))
        fin_interior = PointwiseInteriorConstraint(
            nodes=flow_nodes,
            geometry=geo.holeInterior,
            outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            # outvar={"u": nd_noslip_velocity, "v": nd_cooling_inlet_velocity},
            batch_size=int(cfg.batch_size.finGlobalModifier*cfg.batch_size.coolingInterior),
            compute_sdf_derivatives=True,
            #criteria=GreaterThan(x, geo.coolingNormalX), #if not channel
            #criteria=channel_exterior_sdf,
            lambda_weighting={
                "continuity": lambdaW,
                "momentum_x": lambdaW,
                "momentum_y": lambdaW,
            },
            batch_per_epoch=cfg.batch_size.batchPerEpoch,
            #importance_measure=importance_measure,
            parameterization=geo.pr
        )
        flow_domain.add_constraint(fin_interior, "finInterior")
        
        
        # cooling outlet
        coolingOutlet = PointwiseBoundaryConstraint(
            nodes=flow_nodes,
            geometry=geo.coolingOutlet,
            outvar={"p": nd_cooling_outlet_p},
            batch_size=int(cfg.batch_size.finGlobalModifier*cfg.batch_size.outlet),
            parameterization=geo.pr,
            lambda_weighting={"p": 1},
            # criteria=finInteriorCriteria,
            batch_per_epoch=cfg.batch_size.batchPerEpoch
        )
        flow_domain.add_constraint(coolingOutlet, "coolingOutlet")

    # integral continuity cooling flow
    coolingIntegral = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.coolingIntegral,
        outvar={"normal_dot_vel": inletVelocity1*(7/8)*holeRadius1*2}, #((1-(y/holeRadius1))^(1/7))dy integral from 0 to holeRadius1 = (7/8)*holeRadius1
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=int(cfg.batch_size.finGlobalModifier*cfg.batch_size.integral_continuity),
        parameterization={**geo.pr, **geo.coolingIntegralRange},
        lambda_weighting={"normal_dot_vel": 0.1},
        # criteria=finInteriorCriteria,
        fixed_dataset=False
    )
    flow_domain.add_constraint(coolingIntegral, "coolingIntegral")


    # add monitor
    paras = paramRanges.maxval()
    # print(paras)
    nrPoints = 200

    coolingInletPoints = geo.coolingInlet.sample_boundary(nrPoints, parameterization=paras)
    coolingOutletPoints = geo.coolingOutlet.sample_boundary(nrPoints, parameterization=paras)
    interiorPoints = geo.holeInterior.sample_interior(nrPoints*100, parameterization=paras)

    # cooling_inlet_flow
    coolingInlet = PointwiseMonitor(
        coolingInletPoints,
        output_names=["normal_dot_vel", "area"],
        metrics={"cooling_inlet_flow": lambda var: torch.sum(var["area"] * var["normal_dot_vel"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(coolingInlet)

    # # cooling_inlet_flow
    # coolingInlet_scaled = PointwiseMonitor(
    #     coolingInletPoints,
    #     output_names=["normal_dot_vel_scaled", "area_scaled"],
    #     metrics={"cooling_inlet_flow_scaled": lambda var: torch.sum(var["area_scaled"] * var["normal_dot_vel_scaled"])},
    #     nodes=flow_nodes,
    # )
    # flow_domain.add_monitor(coolingInlet_scaled)

    # # cooling_inlet_pressure
    # coolingInlet = PointwiseMonitor(
    #     coolingInletPoints,
    #     output_names=["p", "p_scaled"],
    #     metrics={"cooling_inlet_pressure[nondim]": lambda var: torch.mean(var["p"]), "cooling_inlet_pressure[pascal]": lambda var: torch.mean(var["p_scaled"])},
    #     nodes=flow_nodes,
    # )
    # flow_domain.add_monitor(coolingInlet)

    # cooling outlet flow
    coolingOutlet = PointwiseMonitor(
        coolingOutletPoints,
        output_names=["normal_dot_vel", "area"],
        metrics={"cooling_outlet_flow": lambda var: torch.sum(var["area"] * var["normal_dot_vel"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(coolingOutlet)
        
    # inferencers
    outputsNS = ["u", "v", "p"]
    outputsS = ["u_scaled", "v_scaled", "p_scaled"]

    inferenceCoolingInlet = PointwiseInferencer(
        nodes=flow_nodes,
        invar=coolingInletPoints,
        output_names=outputsNS + outputsS
    )
    flow_domain.add_inferencer(inferenceCoolingInlet, "coolingInletMax")

    inferenceCoolingOutlet = PointwiseInferencer(
        nodes=flow_nodes,
        invar=coolingOutletPoints,
        output_names=outputsNS + outputsS
    )
    flow_domain.add_inferencer(inferenceCoolingOutlet, "coolingOutletMax")

    inferenceInterior = PointwiseInferencer(
        nodes=flow_nodes,
        invar=interiorPoints,
        output_names=outputsNS + outputsS
    )
    flow_domain.add_inferencer(inferenceInterior, "coolingInteriorMax")

    #min
    parasmin=paramRanges.minval()
    
    coolingInletPointsMin = geo.coolingInlet.sample_boundary(nrPoints, parameterization=parasmin)
    coolingOutletPointsMin = geo.coolingOutlet.sample_boundary(nrPoints, parameterization=parasmin)
    interiorPointsMin = geo.holeInterior.sample_interior(nrPoints*100, parameterization=parasmin)


     # MONITOR cooling_inlet_flow
    coolingInletMin = PointwiseMonitor(
        coolingInletPointsMin,
        output_names=["normal_dot_vel", "area"],
        metrics={"cooling_inlet_flow_MIN": lambda var: torch.sum(var["area"] * var["normal_dot_vel"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(coolingInletMin)

    # cooling outlet flow
    coolingOutletMin = PointwiseMonitor(
        coolingOutletPointsMin,
        output_names=["normal_dot_vel", "area"],
        metrics={"cooling_outlet_flow_MIN": lambda var: torch.sum(var["area"] * var["normal_dot_vel"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(coolingOutletMin)

    # Inferencers
    inferenceCoolingInlet = PointwiseInferencer(
        nodes=flow_nodes,
        invar=coolingInletPointsMin,
        output_names=outputsNS + outputsS
    )
    flow_domain.add_inferencer(inferenceCoolingInlet, "coolingInletMin")

    inferenceCoolingOutlet = PointwiseInferencer(
        nodes=flow_nodes,
        invar=coolingOutletPointsMin,
        output_names=outputsNS + outputsS
    )
    flow_domain.add_inferencer(inferenceCoolingOutlet, "coolingOutletMin")

    inferenceInterior = PointwiseInferencer(
        nodes=flow_nodes,
        invar=interiorPointsMin,
        output_names=outputsNS + outputsS
    )
    flow_domain.add_inferencer(inferenceInterior, "coolingInteriorMin")


    # make solver
    flow_slv = Solver(cfg, flow_domain)

    # start flow solver
    flow_slv.solve()


if __name__ == "__main__":
    run()
