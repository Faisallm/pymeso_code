
# pyMesoscale: A friendly python library for generating 3D Concrete Mesoscale Models

Concrete can be modeled and analyzed as a multiscale material, incorporating macroscale, mesoscale, and microscale perspectives, as illustrated in Fig. 1 [2]. While concrete appears homogeneous at the macroscale, it is heterogeneous internally. Mesoscale models are extensively used to understand concrete’s mechanical properties and failure mechanisms, as well as the contributions of its various phases to overall behavior. These models provide accurate homogenized responses at the macroscale by accounting for heterogenous properties. Mesoscale modelling offers a distinctive approach to studying the initiation and coalescence of microcracks into larger cracks that lead to concrete failure [3], and captures localized deformations that homogenous continuum models miss. To fully grasp concrete’s failure process, its inherent heterogeneity must be considered, with mesoscale modelling being the most effective method for understanding fracture behavior due to its ability to represent these heterogeneities [4,5]. 

![App Screenshot](https://drive.google.com/file/d/1r7Gjn6_9T3yqxgP4lb0FK-k5GaoJFons/preview)


In general, the development of a concrete meso-model involves two main steps: first, creating the meso-model geometric model of concrete, and then meshing this geometric model to generate the meso-scale finite element model. Recent literature indicates that there are three primary methods for generating meso-geometric models of concrete: the random aggregate placement method [31–35], Voronoi graphics method [36–40], and the XCT scanning method [41,42].
## Authors

- [@Faisallm](https://github.com/Faisallm)


## Visualization with Paraview (file_format: .vti and .vtu)


![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/combined_mesoscale.png?token=GHSAT0AAAAAACSPOZ37MBP5YKNZ4LSBSXSGZS655RQ)

### Visualizing "mortar" material phase with Paraview
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/paraview_mortar.png?token=GHSAT0AAAAAACSPOZ37Y4X7JMEHBO2BVOLWZS65YTQ)


### Visualizing "aggregate" material phase with Paraview
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/paraview_aggregate.png?token=GHSAT0AAAAAACSPOZ37HRO6DCNSJ3XYRM4WZS65ZJA)

### Visualizing "interfacial transition zone (ITZ)" material phase with Paraview
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/paraview_interfacial_transition_zone.png?token=GHSAT0AAAAAACSPOZ374RSIGDQQAEE4WSXGZS65Z3A)

### Visualizing "volume" with Paraview
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/paraview_volume.png?token=GHSAT0AAAAAACSPOZ37UJUXOW3MYVQGSFUQZS652KA)

## Imported Parts ready for Analysis

### imported "mortar" path in Abaqus CAE.
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/abaqus_mortar.png?token=GHSAT0AAAAAACSPOZ37PWAX2CK4WCSHDKY6ZS652YA)

### imported "aggregate" path in Abaqus CAE.
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/abaqus_aggregate.png?token=GHSAT0AAAAAACSPOZ36JWFXKVA5LSA73Q4YZS653KQ)

### imported "ITZ" path in Abaqus CAE.
![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/abaqus_interfacial_transition_zone.png?token=GHSAT0AAAAAACSPOZ36JHVKXHC3OTOQP7JGZS653ZQ)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Installation

Install pyMesoscale with pip

```bash
  pip install pyMesoscale
```
    
## Usage/Examples

```python
from pymesoscale.local_background_grid.mesoscale import Mesoscale


# Trying to generate and place aggregates.
m = Mesoscale(100, 100, 100, 5, 20, 5000)

# export 3D array to vti file for visualization with Paraview
m._export_data(m.glob, export_type="vtk", fileName="mesoscale_model_16.vti")

# convert vti file -> vtu file -> .inp file for analysis in abaqus.
m.convert_vti_to_inp("mesoscale_model_16.vti", "output_topology_16")
```


## Roadmap

- Addition of various aggregate shape geometry.

- Addition of various concrete sample geometry Eg. cylinder

- Addition of various parametric mesoscale model generation algorithm such as Random sequential addition (RSA), Random fractal method (RFM), Random walking algorithm etc.


## Used By

This project is used by the following companies/universities:

- King Fahd University of Petroleum and Minerals, Dhahran, Saudi Arabia.


![App Screenshot](https://raw.githubusercontent.com/Faisallm/pymeso_code/main/images/KFUPM.png?token=GHSAT0AAAAAACSPOZ36SVCFFGIOA3WM6BB2ZS65OIQ)
## Contact

For enquires, email faisallawan1997@gmail.com.

