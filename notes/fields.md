# Methods for Calculating Magnetic Fields


## Dipole-Dipole Approximation 
This is the simplest method of the bunch. What we do here is approximate each "magnet" in a system to be represented as a magnetic dipole moment, rather than a volume. In other words this is a single point in space, in which there is a radiating magnetic field, 

This is particularly useful for quick calcuations, but 

This method isn't so great at shorter distances (think 10^-2 m ), or where geometry of the magnet affects gradients. For example, if I have a coil, or some some of large polygon shaped magnet, by reducing this down to a single point in space(i.e. the dipole) we will filter out any of the fine-grained details resulting from the shape of the object. 

The field at a point in space is defined as 

$$
\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi r^3}
\left[ 3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m} \right]
$$

where `m` is the magnetic moment of the dipole,  `r` is the distance at which we are observing the mangetic field. 

## 
