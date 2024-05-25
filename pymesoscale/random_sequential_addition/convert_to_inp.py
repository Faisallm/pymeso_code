import numpy as np
import os
import pyvista as pv

# filename = "meso6.npy"

# if os.path.exists(os.path.join(os.getcwd(), filename)):
#     print("This file exists.")

# data = np.load(filename)
mesh = pv.read("cells_with_material_phase_3.vtu")

print(mesh.n_cells)
print(mesh.n_points)
print(mesh.n_arrays)
print(mesh.bounds)
print(mesh.center)

# # Access voxel cell type
# # print(mesh.get_cell(0).cell_data)
# # Access data contained in voxel cells
# # print(dir(mesh))
# print(mesh.array_names)
# print(4 in mesh['Material_phases'])

# # Check if 'Material_phases' array exists
# if 'Material_phases' in mesh.array_names:
#     # Get the array containing material phases
#     material_phases = mesh['Material_phases']
    
#     # Define the material phase you want to extract (let's say phase 4)
#     target_material_phase = 3
    
#     # Find cells with the target material phase
#     cells_with_material = np.where(material_phases == target_material_phase)[0]
    
#     # Extract the sub-mesh containing cells with the target material phase
#     sub_mesh = mesh.extract_cells(cells_with_material)
    
#     # Save the sub-mesh to a new file or do further processing
#     sub_mesh.save('cells_with_material_phase_{}.vtu'.format(target_material_phase))
    
#     print("Cells with material phase {} extracted and saved.".format(target_material_phase))
# else:
#     print("Mesh does not contain 'Material_phases' array.")



# print(dir(mesh))

# cell = mesh.get_cell(89900)
# print(dir(cell))
# print(cell.type)
# print(cell)
pv.save_meshio("cells_with_material_phase_3.inp", mesh)
# pv.save_meshio("mesher.xml", mesh)
# n = pv.save_meshio("faisal.inp", mesh)
# pv.save_meshio("mesher.stl", mesh)
# pv.save_meshio("mesher.stp", mesh)
# print(dir(n))
