from sklearn.metrics import precision_score

from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_2d import Line, Channel2D
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.hydra import to_absolute_path

from globalProperties import *
from coolingProperties import *

#stl_center = (0, 0, 0)
stl_scale = 0.001
#stl files location
point_path = to_absolute_path("./stl_files")

# define geometry
class dLightGeoCo(object):
    def __init__(self, param_ranges):

        pr = Parameterization(param_ranges.nondimRanges)
        self.pr = param_ranges.nondimRanges

        # read STL geo
        
        self.coolingNoSlip = self.readSTL('coolingNoSlip', False)
        self.finInterior = self.readSTL('coolingInterior', True)
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

        # self.coolingInletCenter = getBounds(geo = self.coolingInlet, center= True)
        
        # self.finVolumeBounds = getBounds(self.finVolume)
        self.coolingOutletBounds = getBounds(self.coolingOutletSTL)
        self.finInteriorBounds = getBounds(self.finInterior)
        self.finInteriorCenter = getBounds(self.finInterior, True)
        # self.inletArea = (self.inletBounds[1][1] - self.inletBounds[1][0]) * (self.inletBounds[2][1] - self.inletBounds[2][0])
        self.coolingInletBounds = getBounds(self.coolingInletSTL)
        # self.coolingInletRadius = (self.coolingInletBounds[0][1] - self.coolingInletBounds[0][0])/2 #holeRadius1
        # self.coolingInletArea = np.pi*self.coolingInletRadius**2

        # print("rad: ", self.coolingInletRadius)
        # print("area: ", self.coolingInletArea)

        self.coolingInletCenter = getBounds(self.coolingInletSTL, True)
        # channelBounds = getBounds(self.channelInterior)


        self.coolingInlet = Line((self.coolingInletBounds[1][0], -holeRadius1), (self.coolingInletBounds[1][0], holeRadius1), parameterization=pr)
        self.coolingOutlet = Line((self.coolingOutletBounds[1][0], -holeRadius1), (self.coolingOutletBounds[1][0], holeRadius1), parameterization=pr)

        self.coolingIntegralRange = {
            # x_pos: (self.finVolumeBounds[0][0] + 0.01, self.finInteriorBounds[0][0]),
            x_pos: (self.finInteriorBounds[1][0], self.finInteriorBounds[1][1])
            }

        self.coolingIntegral = Line((x_pos, -holeRadius1), (x_pos, holeRadius1), parameterization=pr)
        
        self.holeInterior = Channel2D((self.finInteriorBounds[1][0], -holeRadius1), (self.finInteriorBounds[1][1], holeRadius1), parameterization=pr)

        self.symmetryX = Line((0,self.coolingInletBounds[1][0]), (0, self.coolingOutletBounds[1][1])).rotate(np.pi/2)

    def readSTL(self, fileName, airtight):
        mesh = Tessellation.from_stl(point_path + '/' + fileName + '.stl', airtight)
        return mesh.scale(stl_scale/length_scale.magnitude)
    
    
    

#----------------------------------TEST-----------------------------------
# geot = dLightGeoCo(paramRanges)
# nrPoints = 1000

# paras = paramRanges.maxval()

# var_to_polyvtk(geot.holeInterior.sample_interior(
#     nr_points=nrPoints, parameterization=paras), './vtp/coolingInterior')

# var_to_polyvtk(geot.symmetryX.sample_interior(
#     nr_points=nrPoints, parameterization=paras), './vtp/symmetryX')

# var_to_polyvtk(geot.coolingInlet.sample_boundary(
#     nr_points=nrPoints, parameterization=paras), './vtp/coolingInlet')

# var_to_polyvtk(geot.holeInterior.sample_boundary(
#     nr_points=nrPoints, parameterization=paras), './vtp/coolingNoSlip')

# var_to_polyvtk(geot.coolingOutlet.sample_boundary(
#     nr_points=nrPoints, parameterization=paras), './vtp/coolingOutlet')

# var_to_polyvtk(geot.coolingIntegral.sample_boundary(
#     nr_points=nrPoints*10, parameterization={**paras, **{"x_pos": 0.5}}), './vtp/coolingIntegralPlane')
