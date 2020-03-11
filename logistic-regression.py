import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import tkinter as tk



#define sample data

X = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4],
              [3.2, 0.9], [2.9, 1.9],[2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

#create logistics regression classifier

classifier_LR = linear_model.LogisticRegression(solver = 'liblinear', C = 75)

#train the classifier

classifier_LR.fit(X, y)

#create a visualize function
#define min and max values to be used in the grid

# def Logistic_visualize(classifier_LR, X, y):
min_x = X[:, 0].min() - 1.0
max_x = X[:, 0].max() + 1.0
min_y = X[:, 1].min() - 1.0
max_y = X[:, 1].max() + 1.0


#define step size
#sudo dnf install python3-tkinter
mesh_step_size = 0.02

#define mesh grid

x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))


## ????

output = classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)
plt.figure()
plt.pcolormesh(x_vals, y_vals, output, cmap = plt.cm.gray)
 
plt.scatter(X[:, 0], X[:, 1], c = y, s = 75, edgecolors = 'black', 
linewidth=1, cmap = plt.cm.Paired)


plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
plt.show()