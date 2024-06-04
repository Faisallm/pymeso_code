from pymesoscale.local_background_grid.mesoscale import Mesoscale


# Trying to generate and place aggregates.
m = Mesoscale(100, 100, 100, 5, 20, 5000)
# export 3D array to vti file for visualization with Paraview
m._export_data(m.glob, export_type="vtk", fileName="mesoscale_model_16.vti")
# convert vti file -> vtu file -> .inp file for analysis in abaqus.
m.convert_vti_to_inp("mesoscale_model_16.vti", "output_topology_16")