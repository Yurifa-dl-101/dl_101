

Upgrade
YurioWindiatmoko
Pengenalan Deep Learning Part 4 : Deep Learning Framework Introduction (TensorFlow & Keras)
Samuel Sena
Samuel Sena
Follow
Nov 5, 2017 · 4 min read



Kita sudah sama-sama belajar tentang konsep Backpropagation pada part sebelumnya. 
Pada dasarnya untuk melakukan training pada neural network, kita melakukan proses pada diagram dibawah ini secara terus menerus hingga loss atau error yang didapatkan memiliki nilai yang relatif kecil.

img
Neural Network Training

Kita bisa saja mengimplementasikan algoritma tersebut dan membuat sebuah neural network dengan menggunakan semua bahasa pemrograman yang kita bisa. 
- Namun bagaimana jika masalah yang akan kita selesaikan itu adalah permasalahan yang sangat kompleks? Atau 
- mungkin kita membutuhkan arsitektur yang unik dan kompleks? Atau 
- kita ingin menggunakan GPU untuk mempercepat training?

Semua hal diatas dapat diatasi dengan menggunakan sebuah framework. 
Sama halnya seperti semua framework, deep learning framework ada untuk memudahkan kita untuk menyelesaikan masalah menggunakan deep learning.

Sebenarnya ada banyak sekali framework untuk deep learning. 
Bisa dikatakan setiap tech company besar yang ada sekarang memiliki framework masing-masing.
Google mempunyai TensorFlow, 
Facebook dengan Caffe2, 
Microsoft dengan CNTK dan 
masih banyak lagi framework lain seperti `Theano` dan `PyTorch`. 
Kali ini yang akan kita coba sama-sama yaitu TensorFlow (TF).

Keras

Ada lagi satu package yang akan kita gunakan yaitu Keras. 
Sebenarnya TensorFlow sudah cukup jelas cara penggunaannya, tapi kadang dalam riset kita `sering` `sekali` untuk `mencoba arsitektur lain`, `mencari optimizer` `yang paling cepat` `dan bagus`, `tweaking hyperparameter`, dll.

Dengan menggunakan `Keras` `kita bisa melakukan` `semua itu` `dengan relatif lebih cepat` `dari pada ‘pure’ TensorFlow`. 
Karena jika dibandingkan dengan Keras, TensorFlow serasa lebih `“low level”` `meskipun sudah ada tf.layer` `yang baru`.

Jadi Keras ini sebenarnya adalah wrapper dari TensorFlow untuk lebih memudahkan kita lagi. 
Oh ya, tidak hanya TensorFlow aja yang disupport, tapi kita bisa mengganti backend yang akan kita gunakan. 
Saat ini kita bisa gunakan `TensorFlow`, `Theano` dan `CNTK` `sebagai backend` `dari Keras`.

Oh ya..untuk instalasi, pada post ini tidak akan dibahas karena proses instalasi relatif mudah dan kalau memang ada trouble kita bisa googling dengan mudah untuk mencari solusinya, karena banyak sekali tutorial diluar sana tentang bagaimana menginstall TF.

Let’s Code
Ok, kali ini kita akan mencoba untuk melakukan regresi terhadap sebuah fungsi non-linear seperti berikut :

	f(x) =sqrt(2x^2+1)

Sebelumnya kita akan membuat data dengan menggunakan numpy. 
`Input data` nya `dari rentang` `-20 sampai 20` `dengan step 0.25`.

Kita juga buat targetnya sesuai dengan persamaan diatas.

```python
# Generate data from -20, -19.75, -19.5, .... , 20
train_x = np.arange(-20, 20, 0.25)

# Calculate Target : sqrt(2x^2 + 1)
train_y = np.sqrt((2*train_x**2)+1)
```

Setelah data ada, kita bisa mulai membuat modelnya. Arsitektur yang akan kita coba adalah :

- 1 Input Node
- 8 node pada Hidden Layer 1 dengan ReLU activation
- 4 node pada Hidden Layer 2 dengan ReLU activation
- 1 Output node dengan Linear activation

Disini kita juga `menentukan optimizer` `yang akan kita gunakan`, `disini kita akan menggunakan SGD` dan `Mean Squared Error (MSE)` `sebagai loss functionnya`. 
Sebelum kita bisa `melakukan training`, `kita harus` `meng-”compile”` `model` `kita terlebih dahulu`.

```python
# Create Network
inputs = Input(shape=(1,))
h_layer = Dense(8, activation='relu')(inputs)
h_layer = Dense(4, activation='relu')(h_layer)
outputs = Dense(1, activation='linear')(h_layer)
model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)
# Compile the model with Mean Squared Error Loss
model.compile(optimizer=sgd, loss='mse')

```

`Setelah model siap`, kita bisa `mulai melakukan training` `dengan data` `yang kita sudah buat diawal`. 
Untuk melakukan training, kita harus memanggil `method` `fit`.

Pada method ini ada `param` `batch_size` `dengan nilai` `20` `yang artinya` `kita gunakan` `mini-batch SGD`. 
Kalau kita `mau gunakan` `Batch SGD` `kita bisa` `set` `batch_size` nya `sesuai dengan jumlah data kita`. 
Tapi itu silakan dicoba sendiri ya ..:D

Kita akan `lakukan ini` `hingga 10000 epoch` dan `menyimpan semua parameter` `(weights dan bias)` `kedalam` sebuah `file`.

`Epoch`, `learning rate`, `batch_size`, `dll` `ini` adalah `hyperparameter` `yang bisa` `kita tentukan`. 
Sedangkan `nilai` `hyperparameter` `yang ideal`, `sampai saat ini` `masih belum ada riset` `yang bisa memecahkan masalah tersebut`.

Sebenarnya `ada metode` `seperti` `Grid Search` `contohnya` `untuk mencari` `hyperparameter`, `tapi tetap saja` `tidak terjamin kualitasnya`. 
Kapan-kapan kita bahas masalah ini

```python
# Train the network and save the weights after training
model.fit(train_x, train_y, batch_size=20, epochs=10000, verbose=1)
model.save_weights('weights.h5')
```

`Setelah 10000 epoch`, `saya mendapatkan` `MSE` `sebesar` `0.0005` `untuk training data`. 
`Pada tahap ini` `kita akan lakukan` `prediksi` `terhadap angka lain` `diluar training data` `yaitu` `26` dan `akan membandingkan` `hasil prediksi` `seluruh training data` dengan `target`.

Kita bisa `gunakan` `matplotlib` `untuk membuat` `dua grafik` `dan melihat perbandingannya`. 
`Line merah` `untuk target` dan `line biru` `untuk hasil prediksi`.

```python
# Predict training data
predict = model.predict(np.array([26]))
print('f(26) = ', predict)

predict_y = model.predict(train_x)

# Draw target vs prediction
plt.plot(train_x, train_y, 'r')
plt.plot(train_x, predict_y, 'b')
plt.show()
```

Untuk `hasil prediksi` `dari` `26`, `saya dapatkan` `36.755` `sedangkan kalau dihitung` `seharusnya 36.783`. 
`Masih ada error` `tapi not bad` lah.. 

Dan grafik prediction vs target untuk semua training data `sangat identik sekali`. 

img
Prediction vs Target

Complete Code

```python
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD

# Generate data from -20, -19.75, -19.5, .... , 20
train_x = np.arange(-20, 20, 0.25)

# Calculate Target : sqrt(2x^2 + 1)
train_y = np.sqrt((2*train_x**2)+1)

# Create Network
inputs = Input(shape=(1,))
h_layer = Dense(8, activation='relu')(inputs)
h_layer = Dense(4, activation='relu')(h_layer)
outputs = Dense(1, activation='linear')(h_layer)
model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)
# Compile the model with Mean Squared Error Loss
model.compile(optimizer=sgd, loss='mse')

# Train the network and save the weights after training
model.fit(train_x, train_y, batch_size=20, epochs=10000, verbose=1)
model.save_weights('weights.h5')

# Predict training data
predict = model.predict(np.array([26]))
print('f(26) = ', predict)

predict_y = model.predict(train_x)

# Draw target vs prediction
plt.plot(train_x, train_y, 'r')
plt.plot(train_x, predict_y, 'b')
plt.show()
```

Karena semua post ke belakang yang dibahas adalah `regresi`, `nanti pada part` `selanjutnya` `kita mau coba klasifikasi`. 
`Pada prinsipnya` `semua sama` `dengan regresi`, 
hanya saja `kita akan gunakan` `loss` dan `activation function` `yang berbeda`. 

Detailnya nanti kita bahas lagi..So Stay Tune guys…
Dibawah ini adalah series Pengenalan Deep Learning yang bisa kamu ikuti :

Part 1 : Artificial Neural Network
Part 2 : Multilayer Perceptron
Part 3 : BackPropagation Algorithm
Part 4 : Deep Learning Framework Introduction (TensorFlow & Keras)
Part 5 : Dota 2 Heroes Classification (Multiclass Classification)
Part 6 : Deep Autoencoder
Part 7 : Convolutional Neural Network (CNN)
Part 8 : Gender Classification using Pre-Trained Network (Transfer Learning)
Machine Learning
Deep Learning
Artificial Intelligence
Neural Networks
TensorFlow
285 claps



Samuel Sena
WRITTEN BY

Samuel Sena
Follow
Deep Reinforcement Learning Student
See responses (5)
Discover Medium
Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch
Make Medium yours
Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore
Become a member
Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade
About
Help
Legal