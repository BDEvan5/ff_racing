import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Example lidar scan data (distances from car to points on the track)
# lidar_data = np.random.rand(1080) * 50

lidar_data = np.load("Data/MyFrenetPlanner/ScanData/MyFrenetPlanner_0.npy")
fov2 = 4.7 / 2
angles = np.linspace(-fov2, fov2, 1080)
coses = np.cos(angles)
sines = np.sin(angles)

x = lidar_data*coses
y = lidar_data*sines


# Example lidar scan data (x and y coordinates of points on the track)
# x = np.random.rand(1080) * 50
# y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)




# Define a polynomial function to fit through the points
def poly_func(x, *coeffs):
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y += c * x**i
    return y

# Fit a polynomial curve through the points using the least squares method
degree = 4  # polynomial degree to fit
initial_guess = np.ones(degree + 1)  # initial guess for polynomial coefficients
coeffs, _ = curve_fit(poly_func, x, y, p0=initial_guess)

# Evaluate the polynomial at each x coordinate to obtain the curve
curve_x = np.linspace(np.min(x), np.max(x), num=100)
curve_y = poly_func(curve_x, *coeffs)

print("Curve x:", curve_x)
print("Curve x:", curve_y)

plt.figure(figsize=(10, 10))
plt.plot(x, y, label="Data")
plt.plot(curve_x, curve_y, label="Curve")

plt.show()

# Find the minimum distance between each point and the curve to identify the center line
distances = np.abs(y - poly_func(x, *coeffs))
center_line_x = x[np.argmin(distances)]
center_line_y = poly_func(center_line_x, *coeffs)

print("Center line x:", center_line_x)
print("Center line y:", center_line_y)
