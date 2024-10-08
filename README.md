
# pyMesoscale: A friendly python library for generating 3D Concrete Mesoscale Models

Concrete can be modeled and analyzed as a multiscale material, incorporating macroscale, mesoscale, and microscale perspectives, as illustrated in Fig. 1 [2]. While concrete appears homogeneous at the macroscale, it is heterogeneous internally. Mesoscale models are extensively used to understand concrete’s mechanical properties and failure mechanisms, as well as the contributions of its various phases to overall behavior. These models provide accurate homogenized responses at the macroscale by accounting for heterogenous properties. Mesoscale modelling offers a distinctive approach to studying the initiation and coalescence of microcracks into larger cracks that lead to concrete failure [3], and captures localized deformations that homogenous continuum models miss. To fully grasp concrete’s failure process, its inherent heterogeneity must be considered, with mesoscale modelling being the most effective method for understanding fracture behavior due to its ability to represent these heterogeneities [4,5]. 

![1-s2 0-S0013794420300291-gr1_lrg](https://github.com/Faisallm/pymeso_code/assets/13560871/4a0c2944-ec26-4f4d-8264-632a2ad9c7d6)


In general, the development of a concrete meso-model involves two main steps: first, creating the meso-model geometric model of concrete, and then meshing this geometric model to generate the meso-scale finite element model. Recent literature indicates that there are three primary methods for generating meso-geometric models of concrete: the random aggregate placement method [31–35], Voronoi graphics method [36–40], and the XCT scanning method [41,42].


## Demo

![demo_compressed](https://github.com/Faisallm/pymeso_code/assets/13560871/a16443f1-815e-4c42-b0fb-16d0f1cd7fb0)

## Authors

- [@Faisallm](https://github.com/Faisallm)


## Visualization with Paraview (file_format: .vti and .vtu)


![combined_mesoscale](https://github.com/Faisallm/pymeso_code/assets/13560871/28a0c378-273f-472a-aea7-c1d984c3224c)

### Visualizing "mortar" material phase with Paraview
![paraview_mortar](https://github.com/Faisallm/pymeso_code/assets/13560871/f8ff5783-2435-4b6c-a363-a6e6435eb55c)


### Visualizing "aggregate" material phase with Paraview
![paraview_aggregate](https://github.com/Faisallm/pymeso_code/assets/13560871/f43f0fe7-f27b-4904-a765-2e4021da1175)

### Visualizing "interfacial transition zone (ITZ)" material phase with Paraview
![paraview_interfacial_transition_zone](https://github.com/Faisallm/pymeso_code/assets/13560871/daa1d4a9-dde3-45a0-a4d4-b0cdb9b38006)

### Visualizing "volume" with Paraview
![paraview_volume](https://github.com/Faisallm/pymeso_code/assets/13560871/ca512ff0-ca51-4439-864f-ceac64ab4823)

## Imported Parts ready for Analysis  (file_format: .inp)

### imported "mortar" path in Abaqus CAE.
![abaqus_mortar](https://github.com/Faisallm/pymeso_code/assets/13560871/4b89fbd9-f6e2-4675-bcc2-51846c9df70b)

### imported "aggregate" path in Abaqus CAE.
![abaqus_aggregate](https://github.com/Faisallm/pymeso_code/assets/13560871/09d02f2e-7d40-4c1e-bfdb-f7f8bc7d22fb)

### imported "ITZ" path in Abaqus CAE.
![abaqus_interfacial_transition_zone](https://github.com/Faisallm/pymeso_code/assets/13560871/846e54d1-8970-46bd-b6f3-072756fe4228)


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


![KFUPM](https://github.com/Faisallm/pymeso_code/assets/13560871/4627a62c-8eb2-4351-9fa1-111e9f4340e3)
## Contact

For enquires, email faisallawan1997@gmail.com.

## Contributing

Contributions are always welcomed!. Submit pull requests!

email faisallawan1997@gmail.com for ways to get started.