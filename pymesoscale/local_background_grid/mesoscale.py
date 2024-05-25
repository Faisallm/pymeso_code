import numpy as np
import random
import math
import vtk
from vtk.util import numpy_support
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import meshio
import pyvista as pv

directory = os.chdir(os.getcwd())

class Mesoscale:
    def __init__(self, L, W, H, dmin, dmax, number):

        # initialization of the global background grid
        # the global background grid of concrete specimen.
        # let's take unit in mm.
        # L (length)
        # W (width)
        # H (height) of the concrete specimen.

        self.L = L
        self.W = W
        self.H = H

        # minimum and maximum aggregate size value.
        # this value helps in the generation of the polyhedral aggregate.
        self.dmin = dmin
        self.dmax = dmax
        # the mesh element size (e)
        # this is taken as (1/4) ~ (1/8) of minimum aggregate particle size dmin.
        # if the size of mesh element is too small, it will greatly increase...
        # the amount of unneccessary storage and calculation.

        # (we used the largest size possible for our dmin to decrease computation)
        # we could also use (1/8), leading to larger computation
        # self.e = int((1/4) * self.dmin)

        # 1 is selected to ease computation, to prevent fractions...
        # and to pass the assertion test.
        self.e = 1

        # represents the number of elements along x, y and z direction.
        self.l = int(self.L // self.e)
        self.m = int(self.W // self.e)
        self.n = int(self.H // self.e)
        self.number_of_cells = self.l * self.m * self.n

        # self.number represent the number of test aggregates to be generated for placement into the new area
        self.number = number
        print((self.L * self.W * self.H))
        print((self.l * self.e * self.m * self.e * self.n * self.e))
        assert((self.L * self.W * self.H) == (self.l * self.e * self.m * self.e * self.n * self.e))

        # the total number of elements in the concrete specimen
        self.total_elements = self.l * self.m * self.n

        # initialization of the global background grid
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
            while self._condition and self._count < 5000:
                self.agg = self._translate(a, self._random_translation_point(self.l, self.m, self.n, self.e))
                # local grid
                # If their is an intrusion, I want the aggregate to be randomly translated to a new
                # location for placing for atleast self._count times.
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

        self.aggregate_size_ranges = [[5, 10]]
        
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
            while self._condition and self._count < 5000:
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
        """This code determines the eight nodes of any element
       (i, j, k) in the global 3D background grid.
       Where
       i = element no along x-axis
       j = element no along y-axis
       k = element no along z-axis.
       The type of array returned is a numpy array to optimize
       numerical calculations.
       """
        
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
        """calculates the aggregate volume fraction of each aggregate size
        using the fuller curve function."""
        size1, size2 = aggregate_size_range[0], aggregate_size_range[1]
        pdii, pdi = self._fuller(size2), self._fuller(size1)
        pdmax, pdmin = self._fuller(self.dmax), self._fuller(self.dmin)
        
        return ((pdii - pdi) / (pdmax - pdmin))
    
    def _fuller(self, d):
        """This function represents the fuller curve which is one
        of the ideal gradation used for aggregate size distribution
        in order to achieve maximum density of packing."""
        n = 0.5
        D = 20
        return ((d/D)**n)

    def _element_central_point(self, i, j, k, e):
        """This prints the coordinates of the central point
        of the element."""
        return np.array(((i * e) + e / 2, (j * e) + e / 2, (k * e) + e / 2))

    def _locate_element(self, x, y, z, e):
        """If the coordinates of any point in the background
        mesh block are (x, y, z), the number (i, j, k) of the element
        where the point can be located is defined in this function.
        The type of array returned is a numpy array to optimize
           numerical calculations."""
        return np.array((int(x // e), int(y // e), int(z // e)))

    def _global_matrix(self, l, m, n):
        """This a 3D matrix Mijk(0 <= i <= l-1, 0 <= j <= m-1, 0<= k<= n-1).
        is used to store the material attributes of all the global
        background elements.
        It will store elements such as Mijk=1,2,3.
        Meaning that the grid element (i, j, k) is a mortar element,
        an ITZ element or an aggregate element, respectively.

        1 = mortar
        2 = ITZ (Interfacial Transition zone)
        3 = aggregate(spherical, ellipsoidal, polyhedral)

        It is an array of matrices.
        This will probably use alot of compute.
        But the idea is to produce a 3D array that you can easily
        access and change its contents.

        Initialliy all the contents of this array will be initialized 
        as mortar having a value of 1.
        """
        matrix = np.ones((l, m, n), dtype=int)
        return matrix

    def _generate_points(self, radius):
        """This function is responsible for the generation of the coordinates
        (x, y, z) of the vertex of a polygon.
        radius: radius of the aggregate."""

        i_neeta = random.random()
        i_eeta = random.random()
        azimuth = i_neeta * 2 * math.pi
        zenith = i_eeta * 2 * math.pi
        xi = radius * math.sin(zenith) * math.cos(azimuth)
        yi = radius * math.sin(zenith) * math.sin(azimuth)
        zi = radius * math.cos(azimuth)
        return np.array((xi, yi, zi))

    def _generate_polyhedron(self, dmin, dmax):
        """
        This function will return a  single polyhedron.
        where n is the number of vertices.
        we are going to store the vertices of the coordinates in numpy.
        n: represent the number of vertices of the polygon you want to generate.
        dmin: represent the minimum aggregate size.
        dmax: represent the maximum aggregate size.

        The number of three-dimensional random points is preferably
        controlled between 15 and 25.
        """

        neeta = random.random()
        r = (dmin / 2) + (neeta * ((dmax - dmin) / 2))
        choices = [i for i in range(15, 25)]
        polyhedron = [self._generate_points(r) for _ in range(random.choice(choices))]
        return np.array(polyhedron)

    def _random_translation_point(self, l, m, n, e):
        """Represent the coordinates (xp, yp, zp) of the
        random translation points given by: (n7le, n8me, n9ne).
        where n7, n8 and n9 are random translation points
        between 0 and 1.
        l = L//e
        m = W//e
        n = H//e
        L = length of the concrete specimen
        W = Width of the concrete specimen
        H = Height of the concrete specimen
        e = element size (1/4)~(1/8) of dmin.
        To further improve the efficiency of aggregate random placement,
        the random translation point P can be controlled to fall inside the
        mortar elements.
        To ensure that the random translation point falls within the
        dimensions of the concrete specimen
        0 <= Xi"""

        while True:
            n7, n8, n9 = random.random(), random.random(), random.random()
            xp, yp, zp = (n7 * l * e, n8 * m * e, n9 * n * e)
            if (0 <= xp < l * e) and (0 <= yp < m * e) and (0 <= zp < n * e):
                element = self._locate_element(xp, yp, zp, e)
                if self.glob[element[0], element[1], element[2]] == 1:
                    break
        return np.array((xp, yp, zp))

    def _translate(self, aggregate, translation_point):
        """The newly generated random polyhedral aggregate is directly put into
        the background grid space by RANDOM TRANSLATION.
        Assuming the coordinates of the random translation point are
        (xp, yp, zp)=(n7le, n8me, n9ne).
        Therefore, the coordinates of the vertices after translation are:
        Xi = xi + xp, Yi = yi + yp, Zi = zi + zp."""

        xp, yp, zp = translation_point
        translated_aggregate = [(i[0] + xp, i[1] + yp, i[2] + zp) for i in aggregate]
        return np.array(translated_aggregate)

    def _minimum_and_maximum(self, array):
        """According to the new coordinates of the vertices of the...
        polyhedral aggregate. The minimum values (Xmin, Ymin, Zmin) and
        the maximum values (Xmax, Ymax, Zmax) of all the
        vertex coordinates in the three-dimensional direction
        are obtained in this function."""

        ir, jr, kr = np.amin(array, axis=0)
        Ir, Jr, Kr = np.amax(array, axis=0)
        return (int(ir), int(jr), int(kr)), (int(Ir), int(Jr), int(Kr))
    
    def _initialize_local_matrix(self, lower_limit, upper_limit):
        """Represents a 3D matrix B used to temporarily store
        the material attributes of the current local background grid.
        The matrix B is to be initialized to 1, that is all elements in the
        local background grid are initialized as MORTAR ELEMENTS."""

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
        """
        Check if a point is inside a polyhedron defined by its vertices.

        Args:
        - point (numpy array): The point coordinates [x, y, z].
        - vertices (numpy array): The vertices of the polyhedron, where each row represents a vertex [x, y, z].

        Returns:
        - bool: True if the point is inside the polyhedron, False otherwise.
        (deprecated, no longer needed.
        kept here for history)
        """
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
        """
        We have to generate the local background grid from the global
        background grid.

        Always remember that range excludes the outer limit, so don't
        forget to add + 1.
        """

        (ir, jr, kr), (Ir, Jr, Kr) = lower_limit, upper_limit
        ooo_array = np.ones(((Ir - ir) + 1, (Jr - jr) + 1, (Kr - kr) + 1), dtype=int)
        return ooo_array

    def _is_adjacent_to_aggregate(self, space, local, centroid, i, j, k):
        """
        The purpose of this function is to identify elements
        that are adjacent to aggregate elements. Thus help
        in classifying such elements as ITZ (interfacial transition zone).
        """

        # self.delaunay = Delaunay(space.points[space.vertices])
        adjacent_positions = [
            (i + 1, j, k), (i - 1, j, k),
            (i, j + 1, k), (i, j - 1, k),
            (i, j, k + 1), (i, j, k - 1)
        ]
        
        # Check if any neighboring cell is outside the convex hull
        if self.is_point_inside_hull(centroid, space):
            for x, y, z in adjacent_positions:
                #  debugging
                #  print(f"x: {y}, local shape : {local.shape[1]}")
                 if 0 <= x < local.shape[0] and 0 <= y < local.shape[1] and 0 <= z < local.shape[2]:
                    # print("here1")
                    # if local[x][y][z] == 2:
                    #     return True
                    if not self.is_point_inside_hull((x, y, z), space):
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
        
        sub_glob = self.glob[ir:Ir+1, jr:Jr+1, kr:Kr+1]
        contains = np.any((sub_glob == 3) | (sub_glob == 2))
        
        return contains, sub_glob

    def is_point_inside_huller(self, point):
        """
            This function allows us to classify points that are inside
            the convex hull which represents the aggregate. This also aids
            in ITZ (Interfacial Transition Zone) detection.
        """
        return self.delaunay.find_simplex(point) >= 0

    def is_point_inside_hull(self, point, hull, tolerance=1e-12):
        """
            This function allows us to classify points that are inside
            the convex hull which represents the aggregate. This also aids
            in ITZ (Interfacial Transition Zone) detection.
        """
        return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)
        
    def _identify_aggregate_and_itz(self, aggregate):
        """
        1, loop through all the elements in the local background grid of current aggregate.
        i.e all the elements in the red dotted block.

        2, According to the element number of the local background grid,
        the corresponding element of the global background grid is obtained.
        and then the coordinates of the center points are obtained (eq 4).
        """

        # print(f"Aggregate:{aggregate}")
        # Create a 3D plot
        # vertices = np.array(aggregate)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Create the polygon and add it to the plot
        # poly3d = [[vertices[j] for j in range(len(vertices))]]
        # ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        # # Set the limits and labels for better visualization
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # ax.set_xlim(min(vertices[:, 0]), max(vertices[:, 0]))
        # ax.set_ylim(min(vertices[:, 1]), max(vertices[:, 1]))
        # ax.set_zlim(min(vertices[:, 2]), max(vertices[:, 2]))

        # plt.show()
        A = aggregate
        huller = ConvexHull(aggregate)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # for cube, color in zip([A], ['r', 'g', 'b']):
        #     hull = ConvexHull(cube)
        #     print(f"Hull: {hull}")
        #     print(f"Directory: {dir(hull)}")
        #     # draw the polygons of the convex hull
        #     for s in hull.simplices:
        #         tri = Poly3DCollection([cube[s]])
        #         tri.set_color(color)
        #         tri.set_alpha(0.5)
        #         ax.add_collection3d(tri)
        #     # draw the vertices
        #     ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], marker='o', color='purple')
        # plt.show()

        lower_limit, upper_limit = self._section_global_background_grid(aggregate)
        # (ir, jr, kr), (Ir, Jr, Kr) = lower_limit, upper_limit
        (ir, jr, kr), (Ir, Jr, Kr) = lower_limit[0], lower_limit[1]
        (ib, jb, kb), (Ib, Jb, Kb) = upper_limit[0], upper_limit[1]
        local = self._initialize_local_matrix(upper_limit[0], upper_limit[1])
        # print(f"shape: {local.shape}")
        # print(f"ib: {ib}, Ib: {Ib}")
        intrusion, sub_glob = self._aggregate_intrusion(ib, Ib, jb, Jb, kb, Kb)
        intru=False

        # print(f"local shape: {local.shape}")
        # print(f"global shape: {sub_glob.shape}")

        for a, i in enumerate(range(ib, Ib + 1)):
            for b, j in enumerate(range(jb, Jb + 1)):
                for c, k in enumerate(range(kb, Kb + 1)):
                    elem_centroid = self._element_central_point(i, j, k, self.e)
                    if self.is_point_inside_hull(elem_centroid, huller):
                        # print(f"location: {local[a, b, c]}")
                        # let this indicate aggregate
                        if intrusion:
                            intru = True
                            print("Intrusion detected inside aggregate")
                            break
                        local[a, b, c] = 2
                        # print(f"glob: {self.glob[a+ib, b+jb, c+kb]}")
                        self.glob[a+ib, b+jb, c+kb] = local[a, b, c]
                if intru:
                    break
            if intru:
                break

        # print(f"intru boolean:{intru}, intrusion: {intrusion}")
        if not intru:
            for a, i in enumerate(range(ib, Ib + 1)):
                for b, j in enumerate(range(jb, Jb + 1)):
                    for c, k in enumerate(range(kb, Kb + 1)):
                        elem_centroid = self._element_central_point(i, j, k, self.e)
                        if self._is_adjacent_to_aggregate(huller, self.glob, elem_centroid, i, j, k) and not intrusion:
                            # print("inside here")
                            # let this indicate itz
                            local[a, b, c] = 3
                            self.glob[a+ib, b+jb, c+kb] = local[a, b, c]

        # Let's say we want to plot slices along the first axis
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Extract non-zero indices
        # non_zero_indices = np.argwhere(sub_glob)

        # x, y, z = non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]

        # # Scatter plot
        # ax.scatter(x, y, z, c=sub_glob[x, y, z], cmap='plasma', marker='^', s=50, edgecolor='k', alpha=0.7)

        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')

        # plt.show()
        

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
