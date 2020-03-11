import matplotlib.pyplot as plt
import neurolab as nl

#data and input data (supervised)
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

#create net with 2 input 1 neuron
net = nl.net.newp([[0, 1],[0, 1]], 1)

#train the buddy
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1)

#plot
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.show()