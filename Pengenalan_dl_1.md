Upgrade
YurioWindiatmoko
Top highlight

Pengenalan Deep Learning Part 1 : Neural Network
Samuel Sena
Samuel Sena
Follow
Oct 28, 2017 · 6 min read

Deep Learning merupakan topik yang sedang nge-trend dikalangan akademisi ataupun professional. 
Apa sih Deep Learning itu? Deep Learning adalah salah satu cabang Machine Learning (ML) yang menggunakan Deep Neural Network untuk menyelesaikan permasalahan pada domain ML.

Mungkin nanti akan saya bagi dalam beberapa part. Untuk Part I, kita akan sama-sama belajar tentang apa itu Neural Network yang merupakan bagian yang paling penting dari Deep Learning.

Artificial Neural Network

Neural network adalah model yang terinspirasi oleh bagaimana neuron dalam otak manusia bekerja. 
Tiap neuron pada otak manusia saling berhubungan dan informasi mengalir dari setiap neuron tersebut. 
Gambar di bawah adalah ilustrasi neuron dengan model matematisnya.



Credits : Stanford Course
Tiap neuron menerima input dan melakukan `operasi dot` dengan sebuah weight, menjumlahkannya `(weighted sum)` dan `menambahkan bias`. 
`Hasil` dari operasi ini akan `dijadikan parameter` `dari (untuk) activation function` `yang akan dijadikan output dari neuron tersebut`.

Activation Function
Nah, mungkin banyak yang bingung apa dan untuk apa activation function? Sesuai dengan namanya, `activation function` befungsi untuk `menentukan apakah neuron tersebut harus “aktif” atau tidak` `berdasarkan dari weighted sum` `dari input`. 
`Secara umum` terdapat `2 jenis activation function`, `Linear` dan `Non-Linear Activation function`.

Linear Function

Linear Function ; `f(x) = x`
Bisa dikatakan secara “default” activation function dari sebuah neuron adalah Linear. 
Jika sebuah neuron menggunakan linear function, maka keluaran dari neuron tersebut adalah `weighted sum dari input` + `bias`.


Sigmoid and Tanh Function (Non-Linear)

- `Sigmoid function` `mempunyai rentang antara` 
`0 hingga 1`, sedangkan rentang dari
- `Tanh` adalah `-1 hingga 1`. 
`Kedua fungsi` `ini` `biasanya` `digunakan untuk klasifikasi` `2 class` `atau kelompok data`. 
Namun terdapat kelemahan dari kedua fungsi ini, nanti akan coba saya jelaskan di part berikutnya.


ReLU (Non-Linear)

`ReLU Function`
Pada dasarnya `ReLU melakukan` `“treshold”` dari `0 hingga infinity`. 
`ReLU juga dapat menutupi kelemahan` `yang dimiliki oleh Sigmoid dan Tanh` `yang nanti akan saya coba jelaskan di part berikutnya.. :D`

Sebenarnya masih banyak activation function yang lain, namun beberapa fungsi yang saya sebutkan diatas merupakan fungsi yang sering digunakan. 
Sebenarnya masih ada satu lagi `Softmax Function`, tapi nanti akan saya jelaskan pada part `Multiclass Classification`. 
Untuk referensi lengkap tentang activation function bisa dibaca di page wikipedia.

Neural Network Architectures

Credits : Stanford Course
Arsitektur diatas biasa disebut sebagai Multi Layer Perceptron (MLP) atau `Fully-Connected Layer.` 

`Arsitektur pertama` `mempunyai 3 buah neuron` `pada Input Layer` `dan 2 buah node Output Layer`. 
`Diantara Input dan Output`, `terdapat 1 Hidden Layer` `dengan 4 buah neuron`. 

Sedangkan `spesifikasi` `Weight dan Activation function` adalah sebagai berikut:

`Weight and Bias`

`Setiap neuron pada MLP` `saling berhubungan` `yang ditandai` `dengan tanda panah` pada gambar diatas. 
`Tiap koneksi memiliki weight` `yang nantinya` `nilai dari tiap weight` `akan berbeda-beda`.

`Hidden layer` dan `output layer` `memiliki tambahan “input”` `yang biasa disebut dengan bias` `(Tidak disebutkan pada gambar diatas)`.

`Sehingga pada arsitektur pertama terdapat `
`3x4 weight + 4 bias` dan `4x2 weight + 2 bias`. 
`Total` adalah `26 parameter` `yang pada proses training` `akan mengalami perubahan` `untuk mendapatkan hasil yang terbaik`. 

Sedangkan pada arsitektur kedua terdapat 
`41 parameter`.

3x4 + 4 = 16 , 4x4 + 4 = 20 , 4x1 + 1 = 5
total = 41


`Activation Function`

`Neuron pada input layer` `tidak memiliki activation function`, sedangkan 
`neuron pada hidden layer` dan `output layer` `memiliki activation function` `yang kadang` `berbeda` `tergantung daripada data` `atau problem yang kita miliki`.


`Training a Neural Network`
Pada `Supervised Learning` menggunakan `Neural Network`, pada umumnya `Learning` `terdiri dari 2 tahap`, yaitu `training` dan `evaluation`. 
Namun kadang terdapat `tahap tambahan` `yaitu testing`, namun sifatnya tidak wajib.

Pada tahap training `setiap weight` `dan bias` `pada tiap neuron` `akan diupdate` `terus menerus` `hingga output` `yang dihasilkan` `sesuai dengan harapan`. 
Pada `tiap iterasi` `akan dilakukan proses evaluation` `yang biasanya` `digunakan` `untuk menentukan` `kapan harus menghentikan` `proses training (stopping point)`
Pada part selanjutnya, akan saya bahas bagaimana proses training pada neural network lebih mendalam. 
Namun pada part ini akan dijelaskan garis besarnya saja. Proses training terdiri dari 2 tahap :

- Forward Pass
- Backward Pass

Forward Pass
`Forward pass` atau `biasa juga disebut forward propagation` adalah `proses dimana kita membawa data` `pada input` `melewati tiap neuron` `pada hidden layer` `sampai kepada output layer` `yang nanti akan dihitung errornya`

dotj = [3-i]Sigma wji.xi + bj

hj = sig(dotj) = max(0,dotj) 

Persamaan diatas adalah contoh `forward pass` `pada arsitektur pertama` `(lihat gambar arsitektur diatas)` yang `menggunakan ReLU` `sebagai activation function`.
Dimana `i adalah node` `pada input layer` `(3 node input)`, `j adalah node pada hidden layer` sedangkan `h` adalah `output` `dari node pada hidden layer`.

Backward Pass
`Error yang kita dapat pada forward pass` `akan digunakan` `untuk mengupdate` `setiap weight` `dan bias` `dengan learning rate tertentu`.

`Kedua proses` `diatas akan dilakukan berulang-ulang` `sampai didapatkan` `nilai weight` `dan bias` `yang dapat memberikan nilai error` `sekecil mungkin` `pada output layer` `(pada saat forward pass)`

Let’s Code
Pada bagian ini kita mau mencoba implementasi forward pass menggunakan `Python dan Numpy` dulu saja tanpa framework biar lebih jelas. 
Nanti pada part-part selanjutnya akan kita coba dengan `Tensorflow dan Keras`.

Untuk contoh kasusnya adalah kita akan melakukan 
`regresi untuk data` `yang sebenarnya adalah sebuah fungsi linear` sebagai berikut:
	
	f(x) = 3x + 2
Sedangkan arsitektur neural networknya terdiri dari :
`1 node pada input layer => (x)`
`1 node pada output layer => f(x)`
`Neural network diatas sudah saya train`(??) `dan nanti` kita akan `melakukan` `forward pass` `terhadap weight` dan `bias` `yang sudah didapat` `pada saat training`.

`Forward Propagation`
Method `forwardPass` `dibawah ini sangat simple sekali`, `operasi dot` `akan dilakukan pada setiap elemen` `pada input` `dan tiap weight` `yang terhubung dengan input` dan `ditambahkan dengan bias`. 

Hasil dari operasi ini akan dimasukkan ke dalam activation function.

```python
def forwardPass(inputs, weight, bias):
	w_sum = np.dot(inputs, weight) + bias

	# Linear Activation f(x) = x
	act = w_sum

	return act

```


`Pre-Trained Weight`
Untuk `weight dan bias` `yang akan kita coba`, `nilai keduanya sudah didapatkan` `pada proses training` `yang telah saya lakukan sebelumnya`. 
Bagaimana cara mendapatkan kedua nilai tersebut akan dijelaskan pada part-part berikutnya.

```python
# Pre-Trained Weights & Biases after Training
W = np.array([[2.99999928]])
b = np.array([1.99999976])

```

Kalau dilihat dari weight dan bias diatas, nilai keduanya identik dengan fungsi linear kita tadi :

	f(x) = 3x + 2 ≈ f(x) = 2.99999928x + 1.99999976
Complete Code

```python

import numpy as np

def forwardPass(inputs, weight, bias):
	w_sum = np.dot(inputs, weight) + bias # 4x1 dot 1x1 plus bias each of dot result

	# Linear Activation f(x) = x
	act = w_sum

	return act

# Pre-Trained Weights & Biases after Training
W = np.array([[2.99999928]])
b = np.array([1.99999976])

# Initialize Input Data
inputs = np.array([[7], 
				   [8], 
				   [9], 
				   [10]])

# Output of Output Layer
o_out = forwardPass(inputs, W, b)

print('Output Layer Output (Linear)')
print('================================')
print(o_out, "\n")

"""
[[ 22.99999472]
 [ 25.999994  ]
 [ 28.99999328]
 [ 31.99999256]]
"""
```

Pada percobaan kali ini kita akan melakukan perdiksi nilai dari 7, 8, 9 dan 10. 
Output yang dihasilkan seharusnya adalah 23, 26, 29, 32 dan hasil prediksi adalah 22.99999472, 25.999994, 28.99999328 dan 31.99999256. 

Jika dilihat dari hasil prediksi, masih terdapat error tapi dengan nilai yang sangat kecil.

Pada part selanjutnya kita akan sama-sama mencoba `forward pass` `menggunakan` `activation function` `yang lain` `dan mencoba menambahkan hidden layer`. 

Semoga post ini bermanfaat untuk kita semua yang penasaran dengan deep learning :D
Dibawah ini adalah series Pengenalan Deep Learning yang bisa kamu ikuti :

Part 1 : Artificial Neural Network

Part 2 : `Multilayer Perceptron`

Part 3 : BackPropagation Algorithm

Part 4 : Deep Learning Framework Introduction (TensorFlow & Keras)

Part 5 : Dota 2 Heroes Classification (Multiclass Classification)

Part 6 : Deep Autoencoder

Part 7 : Convolutional Neural Network (CNN)

Part 8 : Gender Classification using Pre-Trained Network (Transfer Learning)

Deep Learning
Neural Networks
Machine Learning
Artificial Intelligence
1K claps



Samuel Sena
WRITTEN BY

Samuel Sena
Follow
Deep Reinforcement Learning Student

See responses (12)
Discover Medium
Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch
Make Medium yours
Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore
Become a member
Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade
About
Help
Legal


```python
import numpy as np 
# numpy ndarray fundamental

np.array([[7,7], 
		  [8,7], 
		  [9,7], 
		  [10,7]]) # shape 4x2

np.array([[7,8,9,10],
    	  [7,8,9,10]]) # shape 2x4

np.array([[[7], 
		   [8], 
		   [9], 
		   [10]],
		  
		  [[7], 
		   [8], 
		   [9], 
		   [10]],

		  [[7], 
		   [8], 
		   [9], 
		   [10]]]) # shape 3x4x1  ( means matrix 4x1 has depth 3 `imagining stacks each of it`)

np.zeros((2,4,2))
np.array([[[0, 0], 
		   [0, 0], 
		   [0, 0], 
		   [0, 0]],
		  
		  [[0, 0], 
		   [0, 0], 
		   [0, 0], 
		   [0, 0]]]) # shape 2x4x2  ( means matrix 4x2 has depth 2 `imagining stacks each of it`)
```