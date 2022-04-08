import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([21.04, 14.16, 15.34, 8.52, 18.74, 11.34])
y_train = np.array([460., 232., 315., 178., 434., 203.])

W = 0.0
b = 0.0

n_data = len(x_train)

epochs = 5000
learning_rate = 0.0001

for i in range(epochs):
    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data
    gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data

    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, W, b))

print('W: {:10f}'.format(W))
print('b: {:10f}'.format(b))
print('result : ')
print(x_train * W + b)

x_predict = x_train
y_predict = x_predict * W + b

plt.plot(x_train, y_train, 'or', label='origin data')
plt.plot(x_predict, y_predict, 'b', label='predict')
plt.legend(['origin', 'predict'])
plt.show()
