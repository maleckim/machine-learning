import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl


min_val = -50
max_val = 50
num_points = 160

x = np.linspace(min_val, max_val, num_points)
y = -5 * np.square(x) + 1
y /= np.linalg.norm(y)
# print(x)


data = x.reshape(num_points, 1)
labels = y.reshape(num_points,1)
print(labels)
print(data)

#plot the data
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data-points')

neural_net = nl.net.newff([[min_val, max_val]], [10, 6, 1])

neural_net.trainf = nl.train.train_gd

error = neural_net.train(data, labels, epochs = 1000, show = 100, goal = 0.01)

output = neural_net.sim(data)
y_pred = output.reshape(num_points)

#plotting error progress
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

#plotting actual vs predicted
x_dense = np.linspace(min_val, max_val, num_points)
y_dense_pred = neural_net.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')
plt.show()

