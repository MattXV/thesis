import numpy as np

def linear_extrapolation(x, y, x_new):
    """
    Performs linear extrapolation.

    Parameters:
    x (array-like): Independent variable data points
    y (array-like): Dependent variable data points
    x_new (float or array-like): New x value(s) for extrapolation

    Returns:
    float or array-like: Extrapolated y value(s)
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Fit a linear model (y = mx + c)
    coefficients = np.polyfit(x, y, 1)
    m, c = coefficients
    
    # Calculate the new y value(s) based on the linear model
    y_new = m * np.array(x_new) + c
    
    return y_new

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]



x_new = 6  # Extrapolate for x = 6

extrapolated_value = linear_extrapolation(x, y, x_new)
print(f"Extrapolated value at x = {x_new}: {extrapolated_value}")
target_frequencies = [125, 250, 500, 1000, 2000, 4000]

x = [1000,	    2000,  	3000, 	4000]
y =	[[0.0013,	0.0037,	0.0069,	0.0242],
	 [0.0013,	0.0027,	0.0060,	0.0207],
	 [0.0013,	0.0027,	0.0055,	0.0169],
	 [0.0013,	0.0027,	0.0050,	0.0145]]

air_absorption

for humidity_level in range(len(y)):

    m, c = np.polyfit(x, y[humidity_level], 1)
    ex = lambda newx: m * np.array(newx) + c
    new_f = np.zeros((len(target_frequencies),))
    new_f[0] = ex(125)
    new_f[1] = ex(250)
    new_f[2] = ex(500)
    new_f[3] = y[humidity_level][0]
    new_f[4] = y[humidity_level][1]
    new_f[5] = y[humidity_level][3]

    print(new_f)
    