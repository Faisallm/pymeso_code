import numpy as np
import random
import math
import vtk
from vtk.util import numpy_support
import os

directory = os.chdir(os.getcwd())

class Mesoscale:
    def __init__(self, L, W, H, dmin, dmax, number):
        self.L = L
        self.W = W
        self.H = H
        self.dmin = dmin
        self.dmax = dmax
        # self.e = int((1/4) * self.dmin)
        self.e = 1
        self.l = int(self.L // self.e)
        self.m = int(self.W // self.e)
        self.n = int(self.H // self.e)
        self.number_of_cells = self.l * self.m * self.n
        self.number = number
        print((self.L * self.W * self.H))
        print((self.l * self.e * self.m * self.e * self.n * self.e))
        assert((self.L * self.W * self.H) == (self.l * self.e * self.m * self.e * self.n * self.e))

        self.total_elements = self.l * self.m * self.n
        self.glob = self._global_matrix(self.l, self.m, self.n)
        # size of the aggregate particles
        self.aggregate_size_ranges = [[5, 10], [10, 15], [15, 20]]
        
        # aggregate size volume fraction.
        self.volume_fractions = [self._aggregate_volume_fraction(i) for i in self.aggregate_size_ranges]
        # number of aggregate of each particle size.
        self.agg_fraction = [int(i * self.number) for i in self.volume_fractions]
        self.aggs = []
        
        for i, size in enumerate(self.aggregate_size_ranges):
            aggu = [self._generate_polyhedron(size[0], size[1]) for _ in range(self.agg_fraction[i])]
            self.aggs.extend(aggu)

        # sorting aggregates (Descending order, largest to smallest)
        self.arranged_aggs = sorted(self.aggs, key=lambda x: self._polyhedral_area(x), reverse=True)
        self.agg_counts = 0
    
        self._condition = True
        self._count = 0

        for i, a in enumerate(self.arranged_aggs):
            while self._condition and self._count < 100:
                self.agg = self._translate(a, self._random_translation_point(self.l, self.m, self.n, self.e))
                self.loc, self._condition = self._identify_aggregate_and_itz(self.agg)
                if not self._condition:
                    self._condition = False
                    self._count = 0
                    print("No intrusion")
                else:
                    self._count += 1
                    print("Intrusion detected!")
            self._condition = True
            print(f"Placed #{i+1} aggregate.")
        
        print(f"Local: {self.loc}")
        print(f"Global: {self.glob}")
        
        self._save_vti(self.glob, "aggregate.vti")

    def _element_coordinates(self, i, j, k, e):
        coordinates = np.array((
            (i * e, j * e, k * e),
            ((i + 1) * e, j * e, k * e),
            (i * e, (j + 1) * e, k * e),
            ((i + 1) * e, (j + 1) * e, k * e),
            (i * e, j * e, (k + 1) * e),
            ((i + 1) * e, j * e, (k + 1) * e),
            (i * e, (j + 1) * e, (k + 1) * e),
            ((i + 1) * e, (j + 1) * e, (k + 1) * e)
        ))
        return coordinates

    def _aggregate_volume_fraction(self, aggregate_size_range):
        size1, size2 = aggregate_size_range[0], aggregate_size_range[1]
        pdii, pdi = self._fuller(size2), self._fuller(size1)
        pdmax, pdmin = self._fuller(self.dmax), self._fuller(self.dmin)
        
        return ((pdii - pdi) / (pdmax - pdmin))
    
    def _fuller(self, d):
        n = 0.5
        D = 20
        return ((d/D)**n)

    def _element_central_point(self, i, j, k, e):
        """This prints the coordinates of the central point
        of the element."""
        return np.array(((i * e) + e / 2, (j * e) + e / 2, (k * e) + e / 2))

    def _locate_element(self, x, y, z, e):
        return np.array((int(x // e), int(y // e), int(z // e)))

    def _global_matrix(self, l, m, n):
        matrix = np.ones((l, m, n), dtype=int)
        return matrix

    def _generate_points(self, radius):
        i_neeta = random.random()
        i_eeta = random.random()
        azimuth = i_neeta * 2 * math.pi
        zenith = i_eeta * 2 * math.pi
        xi = radius * math.sin(zenith) * math.cos(azimuth)
        yi = radius * math.sin(zenith) * math.sin(azimuth)
        zi = radius * math.cos(azimuth)
        return np.array((xi, yi, zi))

    def _generate_polyhedron(self, dmin, dmax):
        neeta = random.random()
        r = (dmin / 2) + (neeta * ((dmax - dmin) / 2))
        choices = [i for i in range(15, 25)]
        polyhedron = [self._generate_points(r) for _ in range(random.choice(choices))]
        return np.array(polyhedron)

    def _random_translation_point(self, l, m, n, e):
        """This will translate us to a mortar element."""
        while True:
            n7, n8, n9 = random.random(), random.random(), random.random()
            xp, yp, zp = (n7 * l * e, n8 * m * e, n9 * n * e)
            if (0 <= xp < l * e) and (0 <= yp < m * e) and (0 <= zp < n * e):
                element = self._locate_element(xp, yp, zp, e)
                if self.glob[element[0], element[1], element[2]] == 1:
                    break
        return np.array((xp, yp, zp))

    def _translate(self, aggregate, translation_point):
        xp, yp, zp = translation_point
        translated_aggregate = [(i[0] + xp, i[1] + yp, i[2] + zp) for i in aggregate]
        return np.array(translated_aggregate)

    def _minimum_and_maximum(self, array):
        ir, jr, kr = np.amin(array, axis=0)
        Ir, Jr, Kr = np.amax(array, axis=0)
        return (int(ir), int(jr), int(kr)), (int(Ir), int(Jr), int(Kr))
    
    def _initialize_local_matrix(self, lower_limit, upper_limit):
        """
        We have to generate the local background grid from the global
        background grid.

        Always remember that range excludes the outer limit, so don't
        forget to add + 1.
        All elements in the local background grid are initialized
        as mortar elements.
        """
        # they are being mapped to int.
        (ib, jb, kb), (Ib, Jb, Kb) = map(int, lower_limit), map(int, upper_limit)

        ooo_array = []
        for i in range(ib, (Ib+1)):
            oo_array = []
            for j in range(jb, (Jb+1)):
                o_array = []
                for k in range(kb, (Kb+1)):
                    o_array.append(1)
                oo_array.append(o_array)
            ooo_array.append(oo_array)

        return np.array(ooo_array)

    def _section_global_background_grid(self, aggregate):
        """The bounding box ((ir, jr, kr), (Ir, Jr, Kr)) of the newly placed aggregate can be
        determined using(red dashes) without considering the ITZ.

        While the coordinates ((ib, jb, kb), (Ib, Jb, Kb)) represents the 
        bounding box without considering the ITZ."""


        ((x_min, y_min, z_min), (x_max, y_max, z_max)) = self._minimum_and_maximum(aggregate)

        # bounding box of the newly placed aggregate (consider aggregate elemennts).
        (ir, jr, kr) = np.max([0, x_min//self.e]), np.max([0, y_min//self.e]), np.max([0, z_min//self.e])
        (Ir, Jr, Kr) = np.min([self.l-1, x_max//self.e]), np.min([self.m-1, y_max//self.e]), np.min([self.n-1, z_max//self.e])

        # bounding box of the newly placed aggregate considering ITZ elements
        (ib, jb, kb) = np.max([0, (x_min//self.e)-1]), \
            np.max([0, (y_min//self.e)-1]), np.max([0, (z_min//self.e)-1])
        
#         print(f"x_max: {(x_max//1)+1}, y_max: {(y_max//1)+1}, z_max: {(z_max//1)+1}")
        (Ib, Jb, Kb) = np.min([self.l-1, (x_max//self.e)+1]),\
            np.min([self.m-1, (y_max//self.e)+1]), np.min([self.n-1, (z_max//self.e)+1])

        return ((ir, jr, kr), (Ir, Jr, Kr)), ((ib, jb, kb), (Ib, Jb, Kb)) 

    def _point_in_polyhedron(self, point, vertices):
        intersection_count = 0
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            if (v1[1] > point[1]) != (v2[1] > point[1]):
                x_intersect = (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]
                if point[0] < x_intersect:
                    intersection_count += 1
        return intersection_count % 2 == 1

    def _local_grid(self, lower_limit, upper_limit):
        (ir, jr, kr), (Ir, Jr, Kr) = lower_limit, upper_limit
        ooo_array = np.ones(((Ir - ir) + 1, (Jr - jr) + 1, (Kr - kr) + 1), dtype=int)
        return ooo_array

    def _is_adjacent_to_aggregate(self, space, i, j, k):
        adjacent_positions = [
            (i + 1, j, k), (i - 1, j, k),
            (i, j + 1, k), (i, j - 1, k),
            (i, j, k + 1), (i, j, k - 1)
        ]
        for x, y, z in adjacent_positions:
            if 0 <= x < len(space) and 0 <= y < len(space[0]) and 0 <= z < len(space[0][0]):
                if space[x][y][z] == 3:
                    return True
        return False
    
    import numpy as np

    def _aggregate_intrusion(self, ir: int, Ir: int, jr: int, Jr: int, kr: int, Kr: int):
        """For a newly determined aggregate element (i, j, k) in the local
        background, in order to ensure to that the new aggregate element (i, j, k)
        does not overlap or contact with the old aggregate elements, its
        corresponding element (i + ib, j + jb, k + kb) in the global
        background grid cannot be an aggregate element or ITZ element.
        
        This will return a boolean indicating whether the newly placed aggregate
        in the local background grid is interfaring with a placed old aggregate
        in the global background grid."""
        
        sub_glob = self.glob[ir-1:Ir, jr-1:Jr, kr-1:Kr]
        contains = np.any((sub_glob == 3) | (sub_glob == 2))
        
        return contains
        
    def _identify_aggregate_and_itz(self, aggregate):
        lower_limit, upper_limit = self._section_global_background_grid(aggregate)
        # (ir, jr, kr), (Ir, Jr, Kr) = lower_limit, upper_limit
        (ir, jr, kr), (Ir, Jr, Kr) = lower_limit[0], lower_limit[1]
        (ib, jb, kb), (Ib, Jb, Kb) = upper_limit[0], upper_limit[1]
        local = self._initialize_local_matrix(upper_limit[0], upper_limit[1])
        # print(f"shape: {local.shape}")
        # print(f"ib: {ib}, Ib: {Ib}")
        intrusion = self._aggregate_intrusion(ib, Ib, jb, Jb, kb, Kb)
        sub_glob = self.glob[ib:Ib, jb:Jb, kb:Kb]
        intru=False

        # print(f"local shape: {local.shape}")
        # print(f"global shape: {sub_glob.shape}")

        for a, i in enumerate(range(ib, Ib + 1)):
            for b, j in enumerate(range(jb, Jb + 1)):
                for c, k in enumerate(range(kb, Kb + 1)):
                    elem_centroid = self._element_central_point(i, j, k, self.e)
                    if self._point_in_polyhedron(elem_centroid, aggregate):
                        # print(f"location: {local[a, b, c]}")
                        # let this indicate aggregate
                        if self.glob[a+ib, b+jb, c+kb] != 1:
                            intru = True
                            print("Intrusion detected inside aggregate")
                            break
                        local[a, b, c] = 2
                        self.glob[a+ib, b+jb, c+kb] = local[a, b, c]
                    elif self._is_adjacent_to_aggregate(local, a, b, c):
                        # let this indicate itz
                        local[a, b, c] = 3
                        self.glob[a+ib, b+jb, c+kb] = local[a, b, c]
                if intru:
                    break
            if intru:
                break

        return local, intru
    
    def _save_vti(self, data, filename):
        data_array = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_INT)
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(data.shape)
        image_data.SetSpacing((self.L/self.l, self.W/self.m, self.H/self.n))
        image_data.GetPointData().SetScalars(data_array)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(image_data)
        writer.Write()

    def _polyhedral_area(self, vertices):
    
        if not np.allclose(vertices[0], vertices[-1]):
            vertices = np.vstack((vertices, vertices[0]))

        # Initialize area accumulator
        area = 0.0

        # Calculate the area using the Shoelace formula
        for i in range(len(vertices) - 1):
            cross_product = np.cross(vertices[i], vertices[i + 1])
            area += np.linalg.norm(cross_product)

        # Divide by 2 to get the actual area
        area /= 2.0

        return area

# Trying to generate and place aggregates.
m = Mesoscale(300, 300, 300, 5, 20, 10000)
print(m.glob[90][13])
m._save_vti(m.glob, "mesoscale_model_6.vti")
print("saved vtk file.")

unique, counts = np.unique(m.glob, return_counts=True)
dic = dict(zip(unique, counts))
print(dic)
# print(300*300*300)
co = 0
for key, value in dic.items():
    print(f"item: {key}, value: {value}, percent: {(value/(100*100*100))*100}%")
