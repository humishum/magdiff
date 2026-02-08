import plotly.graph_objects as go
import numpy as np 
from magdiff.components.dipole import Dipole

def visualize_field(system, region=((-1,1),(-1,1),(-1,1)), grid_count=(10,10,10)):
    """
    Create a 3D Plotly figure showing magnetic field lines (as streamtubes) and magnet positions.
    :param system: MagneticSystem object
    :param region: tuple of (x_range, y_range, z_range) each itself a (min, max)
    :param grid_count: tuple (Nx, Ny, Nz) number of grid points in each dimension
    :return: plotly.graph_objects.Figure
    """
    (xmin,xmax), (ymin,ymax), (zmin,zmax) = region
    Nx, Ny, Nz = grid_count
    # Compute field on grid
    X, Y, Z, Bx, By, Bz = system.field_on_grid((xmin,xmax), (ymin,ymax), (zmin,zmax), (Nx,Ny,Nz))
    # Flatten all arrays for streamtube
    x = X.flatten(); y = Y.flatten(); z = Z.flatten()
    u = Bx.flatten(); v = By.flatten(); w = Bz.flatten()
    # Create streamtube trace
    tube = go.Streamtube(x=x, y=y, z=z, u=u, v=v, w=w,
                         starts=dict(x=[0], y=[0], z=[0]),  # start one tube at origin (for example)
                         # The 'starts' can be used to seed streamtubes at desired locations.
                         sizeref=0.5,  # scale of tubes
                         maxdisplayed=100)  # limit number of displayed tubes for clarity
    # Create scatter markers for each magnet (e.g., a sphere or point)
    comps = system.components
    scatter_data = []
    for comp in comps:
        pos = np.array(comp.position)  # convert to numpy for plotting
        if isinstance(comp, Dipole):
            # represent dipole as an arrow (start at position, pointing in direction of moment)
            # We'll use a cone for direction. Plotly cone requires a vector for direction.
            m = np.array(comp.moment)
            cone = go.Cone(x=[pos[0]], y=[pos[1]], z=[pos[2]],
                           u=[m[0]], v=[m[1]], w=[m[2]],
                           colorscale='Blues', showscale=False, sizemode="absolute", sizeref=0.3)
            scatter_data.append(cone)
        else:
            # For other components, just mark position with a symbol
            scatter_data.append(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]],
                                             mode='markers', marker=dict(size=3, color='red'),
                                             name=str(comp.__class__.__name__)))
    # Combine traces
    fig = go.Figure(data=[tube] + scatter_data)
    fig.update_layout(title="Magnetic Field Visualization",
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    return fig
