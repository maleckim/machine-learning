import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

# svm or supervised machine learning can be used for both regression and classification the idea behind SVM
# is to plot each data item as a point in n-dimensional space with the value of each feature being the value 
# of a particular coordinate

#data

# We will use the iris dataset which contains 3 classes of 50 instances each, where each class refers to a type of iris plant. 
# Each instance has the four features namely sepal length, sepal width, petal length and petal width. The SVM classifier to 
# predict the class of the iris plant based on 4 features is shown below.

iris = datasets.load_iris()



X = iris.data[:, :2]
y = iris.target

#create a mesh to plot

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

C = 1.0

svc_classifier = svm.SVC(kernel='linear', 
C=C, decision_function_shape = 'ovr').fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize = (15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.3)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()