from geomdl import fitting
from geomdl.visualization import VisMPL as vis

# The NURBS Book Ex9.1
points =
degree = 3  # cubic curve

# Do global curve approximation
curve = fitting.approximate_curve(points, size_u=3, size_v=3, degree_u=2, degree_v=2)

# Plot the interpolated curve
curve.delta = 0.1
curve.vis = vis.VisCurve2D()
curve.render()