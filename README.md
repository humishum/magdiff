# MagDiff 
### A differentiable magnetics simulation library built in Python, powered by Jax
--- 

The goal of this Python Package is to provide an interface for calculating magnetic fields for, well, magnets(and soon anything else that creates a magnetic field!).
One of the main focuses of this library is that it is created with differentiability in mind, using a Jax native backend. 

With the current version(02/12/26),all calculations rely on the magnetic dipole approximation, except for the coil component, which makes use of the Biot-Savart Law. Additional closed form solutions for differeent volumes coming soon! 

Future plans include(and are not limited to): 
- implementation of common (closed-form) analytical solutions for magnetic volumes. 
    - this should be scalable to support mesh volumes loaded from CAD
- Time varying magnetic fields 
- Implementation of a "Sensor" with the option to adjust gain and simulate ADC 
- Integration with physics backends to support collisions, movement, force, etc. (this one is a bit of a stretch, but is nonetheless a long term goal of the project)

Current high level todo list(as of 02/12/26): 
- closed form solutions for cubes, spheres, cylinders 
- Improve visualization and plotting functions
- Cleanup repo documentation and examples


## Installation 
The package isn't published to any indexes, so we'll rely on a local editable install. 
```
git clone git@github.com:humishum/magdiff.git
cd magdiff
uv add . # or pip install -e .
```

## Usage 

```python 
import jax, jax.numpy as jnp
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.visualize import visualize_field

dip1 = Dipole(name="dipole_1 moment=jnp.array([50.0, 0.0, 0.0]), position=jnp.array([1.0, 0.0, 0.0]),rotation_vector=jnp.zeros(3))
dip2 = Dipole(name="dipole_2 moment=jnp.array([-50.0, 0.0, 0.0]), position=jnp.array([1.0, 1.0, 0.0]),rotation_vector=jnp.zeros(3))
system = MagneticSystem([dip1, dip2], name="sample system")
observer_position = [0.0, 0.0, 0.0]
print(f"B Field at {observer_position} is {system.field_at(observer_position)}")
# B Field at [0.0, 0.0, 0.0] is [ 9.1161164e-06 -2.6516507e-06  0.0000000e+00]
```




## Dependencies 
```
python >=3.12
jax>=0.4.20
jaxlib>=0.4.20
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
plotly>=6.2.0
ruff>=0.14.14
```

