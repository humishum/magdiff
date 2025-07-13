import jax, jax.numpy as jnp
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.visualize import visualize_field

# Target where we want |B| maximal
target = jnp.array([0.5, 0.0, 0.0])

dip = Dipole(moment=jnp.array([0,0,1]), position=jnp.array([-1.0,0,0]))
system = MagneticSystem([dip])

def neg_Bmag(pos):
    tmp = Dipole(dip.moment, pos)
    return -jnp.linalg.norm(tmp.field_at(target))

grad_fn = jax.grad(neg_Bmag)

pos = dip.position
lr = 0.1
for i in range(5):
    g = grad_fn(pos)
    pos = pos - lr * g          # gradient *ascent*
    print(f"{i+1}: pos={pos}, |B|={-neg_Bmag(pos):.4e} T")

dip.position = pos              # update system
fig = visualize_field(system)
fig.show()
