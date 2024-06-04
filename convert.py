import numpy as np
import os
import pyvista as pv



mesh = pv.read("mesoscale_model_14.vti")

print(mesh.n_cells)
print(mesh.n_points)
print(mesh.n_arrays)
print(mesh.bounds)
print(mesh.center)

print(mesh.array_names)
print('Material_phases' in mesh.array_names)

for i in range(1, 4):
    if 'Material_phases' in mesh.array_names:
        # Get the array containing material phases
        material_phases = mesh['Material_phases']
        
        # Define the material phase you want to extract (let's say phase 4)
        target_material_phase = i
        
        # Find cells with the target material phase
        cells_with_material = np.where(material_phases == target_material_phase)[0]
        
        # Extract the sub-mesh containing cells with the target material phase
        sub_mesh = mesh.extract_cells(cells_with_material)
        
        # Save the sub-mesh to a new file or do further processing
        sub_mesh.save('topology{}.vtu'.format(target_material_phase))
        
        print("Cells with material phase {} extracted and saved.".format(target_material_phase))

else:
    print("Mesh does not contain 'Material_phases' array.")


for i in range(1, 4):
    mesh1 = pv.read(f"topology{i}.vtu")
    pv.save_meshio(f"topology{i}.inp", mesh1)