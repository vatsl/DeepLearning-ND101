from numpy import *


def compute_error_for_line_given_points(b, m, points):
    #initialize error at 0
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        # get the x and y values
        x = points[i, 0]
        y = points[i, 1]
        #get the difference, square it and add to the totalError
        totalError += (y - (m*x + b))**2

    # get the average
    return totalError/ float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        # update b and m with the new more accurate b and m by performing
        # this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    # starting points for our gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # computing partial derivatives w.r.t b and m
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    # update b and m values using this partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]



def run():
    #Step 1 - Collect Data
    points = genfromtxt('data.csv', delimiter=',')

    #Step 2 - Define the Hyperparameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    # y = mx + b
    num_iterations = 2000

    #Step 3 - Train our model
    print("Starting gradient descent at b = {}, m = {}, error = {}"
           .format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))

    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print("After {} iterations, b = {}, m = {}, error = {}"
           .format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':
    run()
