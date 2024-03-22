from sklearn.metrics import precision_score
#import torch
from torch.utils.data import DataLoader, Dataset

from sympy import LessThan, GreaterThan, Or, And, StrictGreaterThan, StrictLessThan, Equality

from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_3d import Box, Channel, Plane, Cylinder
from modulus.sym.geometry.primitives_2d import Circle
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.hydra import to_absolute_path
from modulus.sym.utils.io.vtk import var_to_polyvtk

from channelProperties import *

#stl_center = (0, 0, 0)
stl_scale = 0.001
#stl files location
point_path = to_absolute_path("./stl_files")

# define geometry
class dLightGeo(object):
    def __init__(self, param_ranges):
     
        pa = param_ranges.nondimRanges

        pr = Parameterization(pa)
        self.pr = pa

        # read STL geo
        
        self.channelInlet = self.readSTL('channelInlet', False)
        self.channelOutlet = self.readSTL('channelOutlet', False)
        self.channelInterior = self.readSTL('channelInterior', True)
        self.channelNoSlip = self.readSTL('channelOpen', False)
        self.channelExternalWall = self.readSTL('channelExternalWall', False) #FIXA
        self.channelSolidInterface = self.readSTL('channelSolidInterface', True)
        self.solidInterior = self.readSTL('solidVolumeNoHole', True)
        self.solidExteriorWall = self.readSTL('solidExteriorWall', False)
        # self.coolingSolidInterface = self.readSTL('coolingSolidInterface', False)
        # self.coolingNoSlip = self.readSTL('coolingNoSlip', False)
        self.coolingInterior = self.readSTL('coolingInterior', True)
        self.coolingInletSTL = self.readSTL('coolingInlet', False)
        self.coolingOutletSTL = self.readSTL('coolingOutlet', False)
        
        def getBounds(geo, center = False):
            bounds = geo.bounds.bound_ranges
            keys = list(bounds.keys())
            if center:
                boundRanges = (np.mean(bounds[keys[0]]), np.mean(bounds[keys[1]]), np.mean(bounds[keys[2]]))
            else:
                boundRanges = (bounds[keys[0]], bounds[keys[1]], bounds[keys[2]])
            return boundRanges

        self.channelInletBounds = getBounds(self.channelInlet)
        self.solidInteriorBounds = getBounds(self.solidInterior)
        self.coolingInteriorBounds = getBounds(self.coolingInterior)
        self.channelInletArea = (self.channelInletBounds[1][1] - self.channelInletBounds[1][0]) * (self.channelInletBounds[2][1] - self.channelInletBounds[2][0])
        self.coolingInletBounds = getBounds(self.coolingInletSTL)
        # self.coolingInletRadius = holeRadius1
        # self.coolingInletArea = (self.coolingInletBounds[0][1] - self.coolingInletBounds[0][0]) * (self.coolingInletBounds[2][1] - self.coolingInletBounds[2][0])
        self.coolingInletCenter = getBounds(self.coolingInletSTL, True)
        self.coolingOutletCenter = getBounds(self.coolingOutletSTL, True)
        self.channelInletCenter = getBounds(self.channelInlet, True)
        channelBounds = getBounds(self.channelInterior)

        nd_channel_origin = (channelBounds[0][0], channelBounds[1][0], channelBounds[2][0])
        nd_channel_end = (channelBounds[0][1], channelBounds[1][1], channelBounds[2][1])

        # solid csg
        self.solidInteriorCsg = Box((self.solidInteriorBounds[0][0], self.solidInteriorBounds[1][0], self.solidInteriorBounds[2][0]), (self.solidInteriorBounds[0][1], self.solidInteriorBounds[1][1], self.solidInteriorBounds[2][1]), parameterization=pr)
        self.solidInteriorCsgGrad = Box((self.solidInteriorBounds[0][0], self.solidInteriorBounds[1][0], self.solidInteriorBounds[2][0]), (self.solidInteriorBounds[0][1], -0.1, self.solidInteriorBounds[2][1]) )

        # channel integral plane
        self.channelIntegralRange = {
            x_pos: (nd_channel_origin[0], nd_channel_end[0])
        }
        self.channelIntegralPlane = Plane(
            (x_pos, nd_channel_origin[1], nd_channel_origin[2]),
            (x_pos, nd_channel_end[1], nd_channel_end[2]),
            1
        )


        # Cooling GEO
        # parameterized CoolingHole
        cylinderHeight = self.coolingInteriorBounds[1][1] - self.coolingInteriorBounds[1][0]
        #Hole1
        self.coolingHole = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius1, height=cylinderHeight, parameterization=pr)
        self.coolingHoleE = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius1+0.1, height=cylinderHeight, parameterization=pr)
        self.coolingHoleES = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius1+0.02, height=cylinderHeight, parameterization=pr)
        self.coolingHoleS = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius1-0.01, height=cylinderHeight, parameterization=pr)
        self.coolingHole = self.coolingHole.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.coolingHoleS = self.coolingHoleS.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.finExterior1 = self.coolingHoleE.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.finExteriorS1 = self.coolingHoleES.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.finInterior = self.coolingHole.translate((0, 0.5*(self.coolingInteriorBounds[1][0] + self.coolingInteriorBounds[1][1]), 0))
        self.finInteriorLR = self.coolingHoleS.translate((0, 0.5*(self.coolingInteriorBounds[1][0] + self.coolingInteriorBounds[1][1]), 0))
        
        #Hole2
        self.coolingHole2 = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius2, height=cylinderHeight, parameterization=pr)
        self.coolingHoleE2 = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius2+0.1, height=cylinderHeight, parameterization=pr)
        self.coolingHoleES2 = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius2+0.02, height=cylinderHeight, parameterization=pr)
        self.coolingHoleS2 = Cylinder(center=(self.coolingInletCenter[0], 0, self.coolingInletCenter[2]), radius=holeRadius2-0.01, height=cylinderHeight, parameterization=pr)
        self.coolingHole2 = self.coolingHole2.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.coolingHoleS2 = self.coolingHoleS2.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.finExterior2 = self.coolingHoleE2.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.finExteriorS2 = self.coolingHoleES2.rotate(np.pi/2, "x", (self.coolingInletCenter[0], 0, self.coolingInletCenter[2]))
        self.finInterior2 = self.coolingHole2.translate((0, 0.5*(self.coolingInteriorBounds[1][0] + self.coolingInteriorBounds[1][1]), 0))
        self.finInteriorLR2 = self.coolingHoleS2.translate((0, 0.5*(self.coolingInteriorBounds[1][0] + self.coolingInteriorBounds[1][1]), 0))

        # self.finExterior1 = self.coolingHole.scale(2)
        # self.finExterior2 = self.coolingHole2.scale(2)
       
        #posXZ
        self.finInterior = self.finInterior.translate((holePosX1, 0, holePosZ1), parameterization=pr)
        self.finInterior2 = self.finInterior2.translate((holePosX2, 0, holePosZ2), parameterization=pr)
        self.finInteriorLR = self.finInteriorLR.translate((holePosX1, 0, holePosZ1), parameterization=pr)
        self.finInteriorLR2 = self.finInteriorLR2.translate((holePosX2, 0, holePosZ2), parameterization=pr)
        self.finExterior1 = self.finExterior1.translate((holePosX1, 0, holePosZ1), parameterization=pr)
        self.finExteriorS1 = self.finExteriorS1.translate((holePosX1, 0, holePosZ1), parameterization=pr)
        self.finExterior2 = self.finExterior2.translate((holePosX2, 0, holePosZ2), parameterization=pr)
        self.finExteriorS2 = self.finExteriorS2.translate((holePosX2, 0, holePosZ2), parameterization=pr)
        self.finExteriorCriteriaGeo = self.finExterior1+self.finExterior2
        self.finExterior1 = self.finExterior1-self.finInterior
        self.finExteriorS1 = self.finExteriorS1-self.finInterior
        self.finExterior2 = self.finExterior2-self.finInterior2
        self.finExteriorS2 = self.finExteriorS2-self.finInterior2

        self.solidNearHoles = self.finExteriorS1+self.finExteriorS2
        
        self.finInteriorCriteriaGeo = self.finInterior.repeat(spacing=cylinderHeight/2, repeat_lower=(0, -1, 0), repeat_higher=(0, 1, 0))
        self.finInteriorCriteriaGeo2 = self.finInterior2.repeat(spacing=cylinderHeight/2, repeat_lower=(0, -1, 0), repeat_higher=(0, 1, 0))

        self.solidExteriorWallCriteriaGeo = self.finInteriorCriteriaGeo
        self.solidExteriorWallCriteriaGeo += self.finInteriorCriteriaGeo2

        # self.finInterior = self.finInterior-self.finInteriorLR
        # self.finInterior2 = self.finInterior2-self.finInteriorLR2

        self.solidFluxBalance = self.solidInterior-self.solidExteriorWallCriteriaGeo

        #CoolingNormalPlane
        #Plane1
        self.coolingNormalPlane1 = Plane(
            (0, -holeRadius1, -holeRadius1),
            (0, holeRadius1, holeRadius1),
            normal=-1,
            parameterization=pr).rotate(np.pi/2, axis='z').translate([self.coolingInletCenter[0], 0.0, self.coolingInletCenter[2]])
        #Plane2
        self.coolingNormalPlane2 = Plane(
            (0, -holeRadius2, -holeRadius2),
            (0, holeRadius2, holeRadius2),
            normal=-1,
            parameterization=pr).rotate(np.pi/2, axis='z').translate([self.coolingInletCenter[0], 0.0, self.coolingInletCenter[2]])
        

        #posXZ
        self.coolingNormalPlane1 = self.coolingNormalPlane1.translate((holePosX1, 0, holePosZ1), parameterization=pr)
        self.coolingNormalPlane2 = self.coolingNormalPlane2.translate((holePosX2, 0, holePosZ2), parameterization=pr)

        self.coolingInlet = self.coolingNormalPlane1.translate([0, self.coolingInletCenter[1], 0])
        self.coolingInlet2 = self.coolingNormalPlane2.translate([0, self.coolingInletCenter[1], 0])
        self.coolingOutlet = self.coolingNormalPlane1.translate([0, self.coolingOutletCenter[1], 0])
        self.coolingOutlet2 = self.coolingNormalPlane2.translate([0, self.coolingOutletCenter[1], 0])


        self.nd_channel_end = nd_channel_end
        self.nd_channel_origin = nd_channel_origin
        
        # Symmetry Planes
        self.symmetryXY= Plane((0, nd_channel_origin[0], nd_channel_origin[1]), (0, nd_channel_end[0],nd_channel_end[1])).rotate(angle= np.pi/2, axis='z').rotate(angle= np.pi/2, axis='x')
        self.symmetryXZ= Plane((0, nd_channel_origin[0], nd_channel_origin[2]), (0, nd_channel_end[0],nd_channel_end[2])).rotate(angle= np.pi/2, axis='z')

        # criterias
        lrBoxBoundsX = (self.solidInteriorBounds[0][0] - 0.05, self.solidInteriorBounds[0][1] + 0.25)
        lrBoxBoundsZ = (self.solidInteriorBounds[2][0] - 0.05, self.solidInteriorBounds[2][1] + 0.05)
        lrBoxBoundsFlowX = (self.solidInteriorBounds[0][0] - 0.15, self.solidInteriorBounds[0][1] + 0.5)
        lrBoxBoundsFlowZ = (self.solidInteriorBounds[2][0] - 0.15, self.solidInteriorBounds[2][1] + 0.15)
        lrBoxBoundsLargeX = (self.solidInteriorBounds[0][0] - 0.1, self.solidInteriorBounds[0][1] + 0.1) # thermal # 0.3 > 1.7 @ All RS7
        lrBoxBoundsLargeZ = (self.solidInteriorBounds[2][0] - 0.1, self.solidInteriorBounds[2][1] + 0.1)
        self.hrBoxMinBound = lrBoxBoundsLargeX[0]

        self.channelIneriorHR = Box((lrBoxBoundsLargeX[0], self.solidInteriorBounds[1][0], lrBoxBoundsLargeZ[0]),(lrBoxBoundsLargeX[1], self.solidInteriorBounds[1][1], lrBoxBoundsLargeZ[1]))
        self.channelInletHR = Plane((lrBoxBoundsLargeX[0], self.solidInteriorBounds[1][0], lrBoxBoundsLargeZ[0]),(lrBoxBoundsLargeX[0], self.solidInteriorBounds[1][1], lrBoxBoundsLargeZ[1]))

        self.solidExteriorCriteriaXZ = Or(Or(StrictLessThan(x, self.solidInteriorBounds[0][0]), StrictGreaterThan(x, self.solidInteriorBounds[0][1])),
                                            Or(StrictLessThan(z, self.solidInteriorBounds[2][0]), StrictGreaterThan(z, self.solidInteriorBounds[2][1])))

        self.lrBounds = Or(LessThan(x, lrBoxBoundsX[0]), GreaterThan(x, lrBoxBoundsX[1]))
        self.hrBounds = And(And(GreaterThan(x, lrBoxBoundsX[0]), LessThan(x, lrBoxBoundsX[1])), And(GreaterThan(z, lrBoxBoundsZ[0]), LessThan(z, lrBoxBoundsZ[1])))
        
        self.hrBoundsFlow = And(And(GreaterThan(x, lrBoxBoundsFlowX[0]), LessThan(x, lrBoxBoundsFlowX[1])), And(GreaterThan(z, lrBoxBoundsFlowZ[0]), LessThan(z, lrBoxBoundsFlowZ[1])))
        
        self.hrBoundsLarge = And(And(GreaterThan(x, lrBoxBoundsLargeX[0]), LessThan(x, lrBoxBoundsLargeX[1])), And(GreaterThan(z, lrBoxBoundsLargeZ[0]), LessThan(z, lrBoxBoundsLargeZ[1])))
        self.channelInfCriteriaHR = And(self.solidExteriorCriteriaXZ, self.hrBoundsLarge)

        self.channelHRWallCriteria = And(self.solidExteriorCriteriaXZ,StrictGreaterThan(x, lrBoxBoundsLargeX[0]))

        # inference Planes
        self.inferencePlaneYZ1 = Plane(
            (holePosX1, self.coolingInteriorBounds[1][0], holePosZ1-holeRadius1),
            (holePosX1, self.coolingInteriorBounds[1][1], holePosZ1+holeRadius1),
            normal=-1,
            parameterization=pr)

        self.inferencePlaneYZ2 = Plane(
            (holePosX2, self.coolingInteriorBounds[1][0], holePosZ2-holeRadius2),
            (holePosX2, self.coolingInteriorBounds[1][1], holePosZ2+holeRadius2),
            normal=-1,
            parameterization=pr)

        self.inferencePlaneXZ = Plane((0, lrBoxBoundsLargeX[0],lrBoxBoundsLargeZ[0]), (0, lrBoxBoundsLargeX[1], lrBoxBoundsLargeZ[1])).rotate(np.pi/2)
        self.inferencePlaneXZFull = Plane((0, nd_channel_origin[0],nd_channel_origin[2]), (0, nd_channel_end[0], nd_channel_end[2])).rotate(np.pi/2)


    def readSTL(self, fileName, airtight):
        mesh = Tessellation.from_stl(point_path + '/' + fileName + '.stl', airtight)
        return mesh.scale(stl_scale/length_scale_ch.magnitude)
    
    



# #----------------------------------TEST-----------------------------------
geot = dLightGeo(paramRanges_ch)
nrPoints = 10000
# paras = paramRanges_ch.maxval()

# def solidExteriorWallCriteria(invar, params):
#             sdf = geot.solidExteriorWallCriteriaGeo.sdf(invar, params)
#             return np.less(sdf["sdf"], 0)

# def finInteriorCriteria(invar, params):
#             sdf = geot.finInteriorCriteriaGeo.sdf(invar, params)
#             return np.greater(sdf["sdf"], 0)

# def finInteriorCriteria2(invar, params):
#             sdf = geot.finInteriorCriteriaGeo2.sdf(invar, params)
#             return np.greater(sdf["sdf"], 0)

paramRangs = {
   holeRadius1: (4.5,"mm"),
   holeRadius2: (5,"mm"),
   inletVelocity1: (20,"m/s"),
   inletVelocity2: (20,"m/s"),
   holePosX1: (10,"mm"),
   holePosX2: (-10,"mm"),
   holePosZ1: (8,"mm"),
   holePosZ2: (8,"mm")}

specificParas = parameterRangeContainer(paramRangs)
specificParas.nondim(channelNonDim)
paras=specificParas.minval()
print(paras)

# var_to_polyvtk(geot.symmetryXY.sample_interior(
#     nr_points=nrPoints, parameterization=paras, criteria= geot.solidExteriorCriteriaXZ), './vtp/symmetryXY')
# var_to_polyvtk(geot.symmetryXZ.sample_interior(
#     nr_points=nrPoints, parameterization=paras, criteria= geot.solidExteriorCriteriaXZ), './vtp/symmetryXZ')

# var_to_polyvtk(geot.solidNearHoles.sample_boundary(
#    nr_points=nrPoints*10, parameterization=paras, criteria=And(GreaterThan(y, geot.solidInteriorBounds[1][0]), LessThan(y, 0))), './vtp/solidNearHoles')

# var_to_polyvtk(geot.finExteriorCriteriaGeo.sample_boundary(
#     nr_points=nrPoints*10, parameterization=paras, criteria=And(GreaterThan(y,geot.solidInteriorBounds[1][0]), LessThan(y,geot.solidInteriorBounds[1][1]))), './vtp/finExteriorCriteriaGeo')

var_to_polyvtk(geot.finInterior.sample_interior(
   nr_points=nrPoints, parameterization=paras), './vtp/finInterior')
#var_to_polyvtk(geot.finExterior1.sample_interior(
#    nr_points=nrPoints, parameterization=paras, criteria=And(GreaterThan(y,geot.solidInteriorBounds[1][0]), LessThan(y,geot.solidInteriorBounds[1][1]))), './vtp/finExterior')

# var_to_polyvtk(geot.finInterior2.sample_interior(
#     nr_points=nrPoints*10, parameterization=paras), './vtp/finInterior2')

# var_to_polyvtk(geot.inferencePlaneYZ1.sample_boundary(
#    nr_points=nrPoints, parameterization=paras), './vtp/inferencePlaneYZ1')
# var_to_polyvtk(geot.inferencePlaneYZ2.sample_boundary(
#    nr_points=nrPoints, parameterization=paras), './vtp/inferencePlaneYZ2')
# var_to_polyvtk(geot.inferencePlaneXZ.sample_boundary(
#    nr_points=nrPoints, criteria=geot.channelInfCriteriaHR, parameterization=paras), './vtp/inferencePlaneXZ')

# var_to_polyvtk(geot.coolingOutlet2.sample_boundary(
#    nr_points=nrPoints*10, criteria=finInteriorCriteria2, parameterization=paras), './vtp/coolingOutlet2')


# var_to_polyvtk(geot.solidInterior.sample_interior(
#     nr_points=nrPoints,
#     parameterization=geot.pr), './vtp/solidInterior')

# var_to_polyvtk(geot.solidFluxBalance.sample_boundary(
#     nr_points=nrPoints*10, parameterization=paras), './vtp/solidFluxBalance')

# var_to_polyvtk(geot.solidExteriorWallCriteriaGeo.sample_boundary(int(nrPoints*10), criteria=And(GreaterThan(y, geot.nd_channel_origin[1]), LessThan(y, geot.nd_channel_end[1])), parameterization=paras, quasirandom=False), './vtp/solidFluxBalanceInt')


# var_to_polyvtk(geot.channelInterior.sample_interior(
#     nr_points=nrPoints, criteria=geot.hrBoundsLarge, parameterization=paras), './vtp/channelInterior')
