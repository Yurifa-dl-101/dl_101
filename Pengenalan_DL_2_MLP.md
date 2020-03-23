

Upgrade
YurioWindiatmoko

Pengenalan Deep Learning Part 2 : Multilayer Perceptron
Samuel Sena
Samuel Sena
Follow
Oct 30, 2017 · 2 min read



Pada Part 1 kita sudah mengenal apa itu `neural network`,
`activation function` dan sudah mencoba `implementasi forward propagation` `untuk melakukan regresi` `terhadap fungsi linear f(x) = 3x + 2` (persamaan yg akan merupakan model utk inference)

`Fungsi linear diatas` adalah `fungsi yang sangat simple` `sehingga dengan menggunakan 2 layer` `(Input dan Output)` `saja kita sudah bisa menyelesaikan permasalahan tersebut`. 

Lalu bagaimana dengan `fungsi non-linear`? 
`Tentu saja` `kita tidak bisa` menggunakan `arsitektur` `2 layer tersebut`.

Sehingga untuk `non-linear regression` `kita membutuhkan` `setidaknya` `3 layer neural network` `atau yang biasa disebut Multilayer Perceptron (MLP)` atau `Fully-Connected Layer` dengan menggunakan `non-linear` `activation function` `pada` `seluruh neuron` `di hidden layer`.

Let’s Code
Kita akan mencoba melakukan `forward pass` `pada MLP` `masih dengan Numpy saja`. 
Untuk `contoh kasusnya` adalah kita akan melakukan `regresi` `untuk data` `yang sebenarnya` adalah `sebuah fungsi non-linear` `sebagai berikut:`

	f(x) = sqrt(2x^2+1)

f(-2) = sqrt(2(-2)^2+1) = sqrt(9) = 3
f(-1) = sqrt(2(-1)^2+1) = sqrt(3) = 1.7
f(0) = sqrt(2(0)^2+1) = sqrt(1) = 1
f(1) = sqrt(2(1)^2+1) = sqrt(3) = 1.7
f(2) = sqrt(2(2)^2+1) = sqrt(9) = 3
f(3) = sqrt(2(3)^2+1) = sqrt(13) = 3.6
f(4) = sqrt(2(4)^2+1) = sqrt(17) = 4.12

Non-Linear Function

Sedangkan arsitektur neural networknya terdiri dari :
- 1 node pada input layer
- 8 node pada hidden layer pertama (ReLU)
- 1 node pada output layer (Linear)

`Neural network` `diatas` `sudah saya train` `dan nanti` `kita akan melakukan` `forward pass` `terhadap` `weight dan bias` `yang sudah didapat pada saat training`.


Forward Propagation
Method `forwardPass` `yang kita pakai` `di part sebelumnya` `akan dimodifikasi` `sedikit` `dengan menambahkan` `argument baru` `untuk memilih` `activation function`.

```python

def forwardPass(inputs, weight, bias, activation = 'linear'):
	w_sum = np.dot(inputs, weight) + bias

	if activation is 'relu' :
		# ReLU Activation f(x) = max(0, x)
		act = np.maximum(w_sum, 0)
	else :
		# Linear Activation f(x) = x
		act = w_sum

	return act

```

```python

import numpy as np

def forwardPass(inputs, weight, bias, activation = 'linear'):
	w_sum = np.dot(inputs, weight) + bias

	if activation is 'relu' :
		# ReLU Activation f(x) = max(0, x)
		act = np.maximum(w_sum, 0)
	else :
		# Linear Activation f(x) = x
		act = w_sum

	return act

# Pre-Trained Weights & Biases after Training
W_H = np.array([[0.00192761, -0.78845304, 0.30310717, 0.44131625, 0.32792646, -0.02451803, 1.43445349, -1.12972116]])

b_H = np.array([-0.02657719, -1.15885878, -0.79183501, -0.33550513, -0.23438406, -0.25078532, 0.22305705, 0.80253315])

W_o = np.array([[-0.77540326], 
				[ 0.5030424 ], 
				[ 0.37374797], 
				[-0.20287184], 
				[-0.35956827], 
				[-0.54576212], 
				[ 1.04326093], 
				[ 0.8857621 ]])  # 8x1

b_o = np.array([0.04351173])

# Initialize Input Data
inputs = np.array([[-2], 
				   [0], 
				   [2]])

#Output of Hidden Layer
h_out = forwardPass(inputs, W_H, b_H, 'relu')
""" 3x1 dot 1x8 = 3x8 + 8, (list of scalar with 8 value) = 3x8 than filtering minus to 0 with relu """

print('Hidden Layer Output (ReLU)')
print('================================')
print(h_out, "\n")

# Output of Output Layer
o_out = forwardPass(h_out, W_o, b_o, 'linear')
""" 3x8 dot 8x1 = 3x1 + 1, (list of scalar with 1 value) = 3x1 than activation function linear same as before f(x) = x """

print('Output Layer Output (Linear)')
print('================================')
print(o_out, "\n")

"""[[ 2.96598907]
    [ 0.98707188]
    [ 3.00669343]]"""

```
Complete Code

Pada percobaan `non-linear regression` `kali ini` `kita akan melakukan perdiksi` `nilai` `dari` `-2, 0 dan 2`. 

Output yang dihasilkan seharusnya adalah 3, 1, 3 dan hasil prediksi adalah 2.96598907, 0.98707188 dan 3.00669343. 

`Masih ada sedikit error` tapi paling tidak `hasil diatas menunjukkan` `bahwa MLP` `dapat melakukan regresi` `terhadap fungsi non-linear` `dengan cukup baik`.

Pada `part selanjutnya` kita akan sama-sama coba untuk `melakukan training` pada `neural network` `untuk mendapatkan` `weight dan bias` `yang optimal`.

Dibawah ini adalah series Pengenalan Deep Learning yang bisa kamu ikuti :
Part 1 : Artificial Neural Network
Part 2 : Multilayer Perceptron
Part 3 : BackPropagation Algorithm
Part 4 : Deep Learning Framework Introduction (TensorFlow & Keras)
Part 5 : Dota 2 Heroes Classification (Multiclass Classification)
Part 6 : Deep Autoencoder
Part 7 : Convolutional Neural Network (CNN)
Part 8 : Gender Classification using Pre-Trained Network (Transfer Learning)
Deep Learning
Machine Learning
Artificial Intelligence
Neural Networks
185 claps



Samuel Sena
WRITTEN BY

Samuel Sena
Follow
Deep Reinforcement Learning Student
See responses (4)
Discover Medium
Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch
Make Medium yours
Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore
Become a member
Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade
About
Help
Legal






## numpy fundamental
```python
# count
np.dot(np.array([[-2], 
		  		 [0], 
		  		 [2]]), 
np.array([[1, -3, 3, 4, 3, -2, 1, -1]])) + np.array([-2, -1, -7, -3, -2, -2, 2, 8])

# step 1
np.dot(np.array([[-2],
    			 [0],
      			 [2]]),
    np.array([[1, -3, 3, 4, 3, -2, 1, -1]]))

>> array([[-2,  6, -6, -8, -6,  4, -2,  2],
>>        [ 0,  0,  0,  0,  0,  0,  0,  0],
>>        [ 2, -6,  6,  8,  6, -4,  2, -2]])

# activation function relu
np.maximum(
	np.array([[-2,  6, -6, -8, -6,  4, -2,  2],
          		     [ 0,  0,  0,  0,  0,  0,  0,  0],
          			 [ 2, -6,  6,  8,  6, -4,  2, -2]]), 
	0
)

>> array([[0, 6, 0, 0, 0, 4, 0, 2],
>>        [0, 0, 0, 0, 0, 0, 0, 0],
>>        [2, 0, 6, 8, 6, 0, 2, 0]])

# step 2
np.dot(np.array([[-2],
    [0],
    [2]]),
    np.array([[1, -3, 3, 4, 3, -2, 1, -1]])) + np.array([-2, -1, -7,-3, -2, -2, 2, 8])

>> array([[ -4,   5, -13, -11,  -8,   2,   0,  10],
>>        [ -2,  -1,  -7,  -3,  -2,  -2,   2,   8],
>>        [  0,  -7,  -1,   5,   4,  -6,   4,   6]])

# activation function linear same as before f(x) = x
>> array([[ -4,   5, -13, -11,  -8,   2,   0,  10],
>>        [ -2,  -1,  -7,  -3,  -2,  -2,   2,   8],
>>        [  0,  -7,  -1,   5,   4,  -6,   4,   6]])
```









