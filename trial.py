from pymesoscale.local_background_grid.mesoscale import Mesoscale

# length, width and height, minimum and maximum aggregate, max_attempts
m = Mesoscale(100, 100, 100, 5, 20, 5000)

# providing file names to save .vti file and .inp file...
# these files are stored in our current working directory
m._save_vti(m.glob, "mesoscale_model_12.vti")
m.convert_vti_to_inp("mesoscale_model_12.vti", "mesoscale_model_12.inp")