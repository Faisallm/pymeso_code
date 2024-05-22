import numpy as np
import random
import math

def element_coordinates(i: int, j: int, k: int, e):
    """This code determines the eight nodes of an element
    (i, j, k) in the global 3D background grid.
    Where
    i = element no along x-axis
    j = element no along y-axis
    k = element no along z-axis.
    The type of array returned is a numpy array to optimize
    numerical calculations.
    """
    coordinates = np.array([
        (i * e, j * e, k * e),
        ((i + 1) * e, j * e, k * e),
        (i * e, (j + 1) * e, k * e),
        ((i + 1) * e, (j + 1) * e, k * e),
        (i * e, j * e, (k + 1) * e),
        ((i + 1) * e, j * e, (k + 1) * e),
        (i * e, (j + 1) * e, (k + 1) * e),
        ((i + 1) * e, (j + 1) * e, (k + 1) * e)
    ])
    return coordinates

def element_central_point(i: int, j: int, k: int, e):
    """The coordinates of the central point of any element
    (i, j, k) in the global background grid.
    The type of array returned is a numpy array to optimize
    numerical calculations."""
    return np.array([i * e + e / 2, j * e + e / 2, k * e + e / 2])

def locate_element(x: int, y: int, z: int, e):
    """If the coordinates of any point in the background
    mesh block are (x, y, z), the number (i, j, k) of the elements
    where the point can be located is defined in this function.
    The type of array returned is a numpy array to optimize
    numerical calculations."""
    return np.array([x // e, y // e, z // e], dtype=int)

def global_matrix(l: int, m: int, n: int):
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
    This will probably use a lot of compute.
    But the idea is to produce a 3D array that you can easily
    access and change its contents.
    Initially, all the contents of this array will be initialized 
    as mortar having a value of 1.
    """
    return np.ones((l, m, n), dtype=int)

# initialization of my global matrix
l, m, n, e = 100, 100, 100, 1  # Example dimensions
glob = global_matrix(l, m, n)

def generate_points(radius):
    """This function is responsible for the generation of the coordinates
    (x, y, z) of the vertex of a polygon.
    radius: radius of the aggregate."""
    i_neeta = random.random()
    i_eeta = random.random()

    # azimuth angle
    azimuth = i_neeta * 2 * math.pi
    zenith = i_eeta * 2 * math.pi

    # coordinates
    xi = radius * math.sin(zenith) * math.cos(azimuth)
    yi = radius * math.sin(zenith) * math.sin(azimuth)
    zi = radius * math.cos(azimuth)

    return np.array([xi, yi, zi])

def generate_polyhedron(dmin, dmax):
    """
    This function will return a single polyhedron.
    where n is the number of vertices.
    we are going to store the vertices of the coordinates in numpy.
    n: represent the number of vertices of the polygon you want to generate.
    dmin: represent the minimum aggregate size.
    dmax: represent the maximum aggregate size.
    """
    neeta = random.random()
    r = (dmin / 2) + (neeta * ((dmax - dmin) / 2))
    choices = [i for i in range(6, 12)]
    n_vertices = random.choice(choices)
    polyhedron = np.array([generate_points(r) for _ in range(n_vertices)])
    return polyhedron

def random_translation_point(l: int, m: int, n: int, e):
    """Represent the coordinates (xp, yp, zp) of the
    random translation points given by: (n7le, n8me, n9ne).
    where n7, n8 and n9 are random translation points
    between 0 and 1.
    l = L // e
    m = W // e
    n = H // e
    L = length of the concrete specimen
    H = Width of the concrete specimen
    H = Height of the concrete specimen
    e = element size (1/4)~(1/8) of dmin.
    To further improve the efficiency of aggregate random placement,
    the random translation point P can be controlled to fall inside the
    mortar elements.
    To ensure that the random translation point falls within the
    dimensions of the concrete specimen
    """
    global glob

    while True:
        xp = random.random() * l * e
        yp = random.random() * m * e
        zp = random.random() * n * e
        
        element = locate_element(xp, yp, zp, e)
        
        if glob[element[0], element[1], element[2]] == 1:
            break
    
    return np.array([xp, yp, zp])

def translate(aggregate, translation_point):
    """The newly generated random polyhedral aggregate is directly put into
    the background grid space by RANDOM TRANSLATION.
    Assuming the coordinates of the random translation point are
    (xp, yp, zp)=(n7le, n8me, n9ne).
    Therefore, the coordinates of the vertices after translation are:
    Xi = xi + xp, Yi = yi + yp, Zi = zi + zp."""
    translated_aggregate = aggregate + translation_point
    return translated_aggregate

def minimum_and_maximum(array):
    """According to the new coordinates of the vertices of the
    polyhedral aggregate. The minimum values (Xmin, Ymin, Zmin) and
    the maximum values (Xmax, Ymax, Zmax) of all the
    vertex coordinates in the three-dimensional direction
    are obtained in this function."""
    min_vals = np.amin(array, axis=0).astype(int)
    max_vals = np.amax(array, axis=0).astype(int)
    return min_vals, max_vals

def initialize_local_matrix(lower_limit_coordinate, upper_limit_coordinate):
    """Represents a 3D matrix B used to temporarily store
    the material attributes of the current local background grid.
    The matrix B is to be initialized to 1, that is all elements in the
    local background grid are initialized as MORTAR ELEMENTS."""
    x_min, y_min, z_min = lower_limit_coordinate
    x_max, y_max, z_max = upper_limit_coordinate
    return np.ones((x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1), dtype=int)

def section_global_background_grid(aggregate):
    """The bounding box ((ir, jr, kr), (Ir, Jr, Kr)) of the newly placed aggregate can be
    determined using (red dashes) without considering the ITZ.
    
    While the coordinates ((ib, jb, kb), (Ib, Jb, Kb)) represents the 
    bounding box without considering the ITZ."""
    (x_min, y_min, z_min), (x_max, y_max, z_max) = minimum_and_maximum(aggregate)
    
    ir, jr, kr = x_min // e, y_min // e, z_min // e
    Ir, Jr, Kr = x_max // e, y_max // e, z_max // e
    
    ib, jb, kb = max(0, ir - 1), max(0, jr - 1), max(0, kr - 1)
    Ib, Jb, Kb = min(l, Ir + 1), min(m, Jr + 1), min(n, Kr + 1)
    
    return (ir, jr, kr), (Ir, Jr, Kr), (ib, jb, kb), (Ib, Jb, Kb)

# def point_in_polyhedron(point, vertices):
#     """
#     Check if a point is inside a polyhedron defined by its vertices.

#     Args:
#     - point (numpy array): The point coordinates [x, y, z].
#     - vertices (numpy array): The vertices of the polyhedron, where each row represents a vertex [x, y, z].

#     Returns:
#     - bool: True if the point is inside the polyhedron, False otherwise.
#     """
#     vertices = np.array(vertices)
#     v0 = vertices[:-1] - vertices[-1]
#     v1 = vertices[1:] - vertices[-1]
#     n = np.cross(v0, v1)
#     d = np.einsum('ij,ij->i', n, vertices[-1])
#     return (np.dot(n, point) >= d).all()

def point_in_polyhedron(point, vertices):
    """
    Check if a point is inside a polyhedron defined by its vertices.

    Args:
    - point (numpy array): The point coordinates [x, y, z].
    - vertices (numpy array): The vertices of the polyhedron, where each row represents a vertex [x, y, z].

    Returns:
    - bool: True if the point is inside the polyhedron, False otherwise.
    """

    # Initialize counters for intersections with edges and faces
    intersection_count = 0

    # Iterate through each face of the polyhedron
    for i in range(len(vertices)):
        # Define the vertices of the current face
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]

        # Check if the point is on the same side of the plane defined by the face
        # as the line segment connecting the two vertices of the face
        if (v1[1] > point[1]) != (v2[1] > point[1]):
            # Calculate the intersection point with the plane
            x_intersect = (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]

            # If the intersection point is to the right of the test point, increment the intersection count
            if point[0] < x_intersect:
                intersection_count += 1

    # If the number of intersections is odd, the point is inside the polyhedron
    return intersection_count % 2 == 1

def local_grid(lower_limit, upper_limit):
    """We have to generate the local background grid from the global
    background grid."""
    ir, jr, kr = lower_limit
    Ir, Jr, Kr = upper_limit
    return np.ones((Ir - ir + 1, Jr - jr + 1, Kr - kr + 1), dtype=int)

def identify_aggregate(agg):
    """
    1, loop through all the elements in the local background grid of current aggregate.
    i.e all the elements in the red dotted block.
    
    2, According to the element number of the local background grid,
    the corresponding element of the global background grid is obtained.
    and then the coordinates of the center points are obtained (eq 4).
    """
    global glob
    
    (ir, jr, kr), (Ir, Jr, Kr), (ib, jb, kb), (Ib, Jb, Kb) = section_global_background_grid(agg)
    b_matrix = initialize_local_matrix((ir, jr, kr), (Ir, Jr, Kr))
    local_grids = local_grid((ir, jr, kr), (Ir, Jr, Kr))

    for i in range(ir, Ir + 1):
        for j in range(jr, Jr + 1):
            for k in range(kr, Kr + 1):
                center_point = element_central_point(i, j, k, e)
                li, lj, lk = i - ir, j - jr, k - kr
                
                if point_in_polyhedron(center_point, agg):
                    glob[i, j, k] = 3
                    local_grids[li, lj, lk] = 3
    
    return local_grids

# Example usage
agg = translate(generate_polyhedron(4, 10), random_translation_point(l, m, n, e))
print(section_global_background_grid(agg))
print(identify_aggregate(agg))
print(agg)
