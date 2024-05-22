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
        self.e = int((1/4) * self.dmin)
        self.l = int(self.L // self.e)
        self.m = int(self.W // self.e)
        self.n = int(self.H // self.e)
        self.number = number
        print((self.L * self.W * self.H))
        print((self.l * self.e * self.m * self.e * self.n * self.e))
        assert((self.L * self.W * self.H) == (self.l * self.e * self.m * self.e * self.n * self.e))

        self.total_elements = self.l * self.m * self.n
        self.glob = self._global_matrix(self.l, self.m, self.n)
        self.aggs = [self._generate_polyhedron(self.dmin, self.dmax) for _ in range(self.number)]

        # aggregates arranged in order of decreasing size.
        self.aggs = sorted(self.aggs, key=lambda x: self._polyhedral_area(x), reverse=True)
    
        self._condition = True
        self._count = 0

        for i, a in enumerate(self.aggs):
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

    def _element_central_point(self, i, j, k, e):
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

    def _initialize_local_matrix(self, lower_limit_coordinate, upper_limit_coordinate):
        ((x_min, y_min, z_min), (x_max, y_max, z_max)) = lower_limit_coordinate, upper_limit_coordinate
        ooo_array = np.ones((x_max - x_min, y_max - y_min, z_max - z_min), dtype=int)
        return ooo_array

    def _section_global_background_grid(self, aggregate):
        ((x_min, y_min, z_min), (x_max, y_max, z_max)) = self._minimum_and_maximum(aggregate)
        (ir, jr, kr) = max(0, x_min // self.e), max(0, y_min // self.e), max(0, z_min // self.e)
        (Ir, Jr, Kr) = min(self.l - 1, x_max // self.e), min(self.m - 1, y_max // self.e), min(self.n - 1, z_max // self.e)
        (ib, jb, kb) = max(0, ir - 1), max(0, jr - 1), max(0, kr - 1)
        (Ib, Jb, Kb) = min(self.l - 1, Ir + 1), min(self.m - 1, Jr + 1), min(self.n - 1, Kr + 1)
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

    def _identify_aggregate_and_itz(self, agg):
        """
        1, loop through all the elements in the local background grid of current aggregate.
        i.e all the elements in the red dotted block.

        2, According to the element number of the local background grid,
        the corresponding element of the global background grid is obtained.
        and then the coordinates of the center points are obtained (eq 4).
        """

        # a section of the global matrix
        ((ir, jr, kr), (Ir, Jr, Kr)), ((ib, jb, kb), (Ib, Jb, Kb)) = self._section_global_background_grid(agg)
        local_grids = self._local_grid((ib, jb, kb), (Ib, Jb, Kb))
#         print(local_grids.shape)
        intrusion = self._aggregate_intrusion(ib, Ib, jb, Jb, kb, Kb)
        intru = False

        ir, jr, kr = int(ir), int(jr), int(kr)
        Ir, Jr, Kr = int(Ir), int(Jr), int(Kr)
        for i in range(ir, Ir+1):
            for j in range(jr, Jr+1):
                for k in range(kr, Kr+1):
                    global_element = i, j, k
                    center_point = self._element_central_point(i, j, k, self.e)
                    li, lj, lk = (i-ir, j-jr, k-kr)

                    # if the element is aggregate
                    if self._point_in_polyhedron(center_point, agg):
                        
                        # aggregate intrusion detection.
                        if not intrusion:
                            # change the global matrix
                            self.glob[i][j][k] = 3
                            # change the local matrix
                            # how do I know that its changing the correct position in the global matrix?
                            local_grids[li][lj][lk] = 3 
                        else:
                            intru = True
                            break
                            
#                     elif local_grids[li][lj][lk] == 1 and self._is_adjacent_to_aggregate(local_grids, li, lj, lk):
#                         # change element in local to ITZ.
#                         local_grids[li][lj][lk] = 2
#                         # change corresponding element in global to ITZ.
#                         self.glob[i][j][k] = 2
#                     else:
#                         # change element in local to ITZ.
#                         local_grids[li][lj][lk] = 1
#                         # change corresponding element in global to ITZ.
#                         self.glob[i][j][k] = 1
                        
                if intru:
                    break
            if intru:
                break

        ib, jb, kb = int(ib), int(jb), int(kb)
        Ib, Jb, Kb = int(Ib), int(Jb), int(Kb)
        if not intru:
            for i in range(ib, Ib+1):
                for j in range(jb, Jb+1):
                    for k in range(kb, Kb+1):
                        # coordinates of the global grid
                        global_element = i, j, k
                        # coordinates of the local background grid
                        li, lj, lk = (i-ib, j-jb, k-kb)
#                         print("itz")
#                         print(f"global: i: {i}, j:{j}, k: {k}")
#                         print(f"local: li: {li}, lj:{lj}, lk: {lk}")

                        if intru:
                            break

                        if self.glob[i][j][k] == 1:
                            if self._is_adjacent_to_aggregate(self.glob, i, j, k):
                                # change corresponding element in global to ITZ.
                                self.glob[i][j][k] = 2   
                    if intru:
                        break

                if intru:
                    break

        return local_grids, intru


    
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
m = Mesoscale(100, 100, 100, 4, 20, 10000)
print(m.glob[90][13])
m._save_vti(m.glob, "mesoscale_model_3.vti")
print("saved vtk file.")

unique, counts = np.unique(m.glob, return_counts=True)
dic = dict(zip(unique, counts))
print(dic)
# print(300*300*300)
co = 0
for key, value in dic.items():
    print(f"item: {key}, value: {value}, percent: {(value/(100*100*100))*100}%")