Upgrade
YurioWindiatmoko
Pengenalan Deep Learning Part 3 : BackPropagation Algorithm
Samuel Sena
Samuel Sena
Follow
Nov 3, 2017 · 9 min read



`Pada Part 1` kita sudah sedikit `disinggung` tentang `cara melakukan training` pada `neural network`. 
`Proses training` `terdiri dari` `2 bagian utama` yaitu `Forward Pass` dan `Backward Pass`. 

- Panah biru dibawah ini adalah Forward Pass dan 
- panah merah adalah backward pass.

											  output data (truth)
				blue                   blue        |
	input data ------> neural network ------> output pred
						    |					   |			
						red	|					   |
						     -- back pro <---------- 
						     weight update      red


Neural Network Training

Dalam Supervised Learning, `training data` `terdiri dari` `input` dan `output/target`. 

Pada `saat forward pass`, input akan `di-”propagate”` `menuju output layer` dan `hasil prediksi output` `akan dibandingakan dengan target` `dengan menggunakan sebuah fungsi` `yang biasa disebut dengan Loss Function`.

Lalu `untuk apa loss function itu?` `Secara simple` `loss function` `digunakan` `untuk mengukur` `seberapa bagus` `performa dari neural network` kita `dalam melakukan prediksi` `terhadap target`.

	Loss = (Target - Prediction)^2

Ada berbagai macam `loss function`, namun yang paling sering digunakan adalah `Squared Error (L2 Loss)` `untuk regresi`.

Sedangkan `untuk klasifikasi` `yang biasa digunakan` adalah `Cross Entropy`.

`Backward Pass (Back-Propagation)`
Simplenya `proses ini` `bermaksud` `untuk menyesuaikan` `kembali` `tiap weight` `dan bias` `berdasarkan error` `yang didapat` `pada saat forward pass`. 

Tahapan dari backprop adalah sebagai berikut :
- `Hitung gradient` `dari loss function` `terhadap semua parameter` `yang ada` `dengan cara` `mencari partial derivative` `(turunan parsial)` `dari` `fungsi tersebut`. 
`Disini` `kita bisa menggunakan` `metode` `Chain Rule` `(Kalkulus…:D)`. 

Untuk yang masih bingung apa itu gradient, mungkin ilustrasi dibawah ini bisa membantu.

```
Contoh 1
Tentukan turunan dari y = sin 4x !
     Penyelesaian :
     Misalkan :
     u = 4x   ⇒   u' = 4

     y' = cos u . u'
     y' = cos 4x . 4
     y' = 4cos 4x

Tentukan turunan dari y = sin x^2
     Jawab :  
     y' = cos x^2 . 2x
     y' = 2x cos x^2

Tentukan turunan dari y = x^2 cos 2x
     Jawab :
     Misalkan :
     u = x^2  ⇒  u' = 2x
     v = cos 2x ⇒ v' = −2 sin 2x

     y' = u'.v + u.v'
     y' = 2x . cos 2x + x^2 . −2 sin 2x
     y' = 2x cos 2x − 2x^2 sin 2x
     y' = 2x(cos 2x − x sin 2x)
```
f(x) = x sin(x^2) + 1
misal : 
 u = x -> u' = 1
 v = sin(x^2) -> v' = 2x cos x^2

 y' = u'.v + u.v'
 y' = 1 . sin(x^2) + x . 2x cos x^2
 
 y' = sin(x^2) + 2x^2 cos x^2


A = (2.65 , 2.79)
f'(2.65) = 11.05
# koreksi manual
= 0.67 + 2 . 7.0225 . 0.73
= 0.67 + 10.252 = 10.92

A = (-0.6 , 0.79)
f'(-0.6) = 1.03
# koreksi manual
= sin(0.36) + 2 . 0.36 . cos(0.36)
<!-- = 0.67 + 10.252 = 10.92 -->

A = (0.95 , 1.75)
f'(0.95) = 1.9
# koreksi manual
= sin(0.95^2) + 2 . (0.95^2) . cos(0.95^2)
<!-- = 0.67 + 10.252 = 10.92 -->

gradien --> positif naik (makin besar makin nanjak) , negatif turun (nilai negatif nya semakin besar semakin curam) , 0 datar

By en:User:Dino, User:Lfahlberg — English Wikipedia, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=27661406

- `Update semua parameter` `(weight dan bias)` `menggunakan` `Stochastic Gradient Descent (SGD)` `dengan mengurangi` `atau` `menambahkan` `nilai weight lama` `dengan` `“sebagian”` `(learning rate)` (gambar diatas lr = 0.05) `dari nilai gradient yang sudah kita dapat`??.

Siapkan kertas dan calculator. Yuk kita lihat contoh dibawah ini biar lebih jelas….


1 i layer --> 4 h layer ReLU --> 2 h layer Sigmoid --> 1 o layer Linear

## input to h1 1x4 `weight` + 4 `bias` = 8 param
Wij1 + bj1
Wij2 + bj2
Wij3 + bj3
Wij4 + bj4

## h1 to h2 4x2 `weight` + 2 `bias` = 10 param
Wj1k1 + bk1
Wj1k2 + bk2

Wj2k1 + bk1
Wj2k2 + bk2

Wj3k1 + bk1
Wj3k2 + bk2

Wj4k1 + bk1
Wj4k2 + bk2

## h2 to o 2x1 `weight` + 1 `bias` = 3 param
Wk1o + bo
Wk2o + bo

total param 8 param + 10 param + 3 param = 21 param (need to be update when doing backpro)


`Neural network` diatas `terdiri dari` `2 hidden layer`. 
`Hidden layer pertama menggunakan ReLU`, `hidden layer kedua` `menggunakan sigmoid` dan `terakhir` `output layer` `menggunakan linear` sebagai `activation function`. 

Bias pada diagram diatas sebenarnya ada tetapi tidak digambarkan.
Terdapat 
4 weight dan 4 bias diantara input layer dan hidden layer pertama,
8 weight dan 2 bias diantara hidden layer pertama dan kedua, 
2 weight dan 1 bias diantara hidden layer kedua dan output layer.

Sehingga total ada 21 parameter yang harus diupdate.

Kita akan `mencoba untuk melakukan prediksi` `terhadap suatu nilai`. 
Untuk `initial weight` dan `bias`, `saya tentukan sendiri` `dengan nilai` `angka yang lebih enak` `dilihat :D`

input = [[2.0]] ; output = [[3.0]]
1x1

wij = [[Wij1, Wij2, Wij3, Wij4]] = [[0.25 0.5 0.75 1]]
1x4

wjk = [[Wj1k1, Wj1k2],   
	   [Wj2k1, Wj2k2], 
	   [Wj3k1, Wj3k2], 
	   [Wj4k1, Wj4k2]]
   
    = [[1.0, 0],   
	   [0.75, 0.25], 
	   [0.5, 0.5], 
	   [0.25, 0.75]]
4x2

wko = [[Wk1o],   
	   [Wk2o]]

    = [[1.0],   
	   [0.5]] 
2x1

1x1 . 1x4 = 1x4 . 4x2 = 1x2 . 2x1 = 1x1

bij = [bij1 bij2 bij3 bij4] = [1.0 1.0 1.0 1.0]
bjk = [bjk1 bjk2] = [1.0 1.0]
bo = [1.0]

Forward Pass (Input -> Hidden Layer 1)

Disini kita akan melakukan `forward pass` `data input` `menuju` `hidden layer 1`. 
Yang dilakukan adalah `melakukan perkalian` `(dot product)` `dan` `penjumlahan matrix` `antara input`, `weight` (melakukan perkalian dot product) dan `bias`(penjumlahan matrix).



Nilai diatas adalah input dari tiap node pada hidden layer 1. Semua nilai tersebut akan dikeluarkan setelah melalui activation function. 
Pada hidden layer 1 activation function yang kita gunakan adalah ReLU => f(x) = max(0, x). 
Sehingga output dari hidden layer 1 adalah sebagai berikut :


Forward Pass (Hidden Layer 1 -> Hidden Layer 2)

Sama seperti forward pass pada layer sebelumnya, output dari tiap neuron pada ReLU layer akan mengalir ke semua neuron pada Sigmoid layer.



Setelah activation function :

sigmoid => f(x) = 1 / 1 + e^x

sigmoid([k1in k2in]) = [ 1 / 1 + e^k1in   1 / 1 + e^k2in ]

sigmoid([k1in k2in]) = [ 1 / 1 + e^-6   1 / 1 + e^-5 ]

sigmoid([k1in k2in]) = [ 0.9975   0.9933 ]

[k1out k2out] = [ 0.9975   0.9933 ]


Forward Pass (Hidden Layer 2 -> Output)

Sama seperti forward pass pada layer sebelumnya, output dari tiap neuron pada Sigmoid layer akan mengalir ke neuron pada Linear layer (Output).


Setelah activation function :


Kita sudah sampai pada output layer dan sudah mendapatkan nilai prediksi output. 
Selanjutnya kita akan mencari loss dengan menggunakan squared error (L2 Loss).

Loss = 1/2 (Prediction - Target)^2
Loss = 1/2 (Oout - output)^2
Loss = 1/2 (2.494 - 3)^2
Loss = 1/2 (-0.506)^2
Loss = 1/2 (0.256)
Loss = 0.128

Kenapa kok dikali 1/2? Nanti akan dijelaskan lebih lanjut

Activation Function Derivatives
`Sebelum membahas` `tentang` `backward pass` `ada baiknya` `kalau kita` `mencari dulu` `turunun` `tiap activation function` `yang kita pakai`.


- ReLU Derivatives
y = max(0,x)

dy/dx = { 1 untuk x > 0 , 0 untuk x <= 0 }


- Sigmoid Derivatives
y = 1 / 1 + e^(-x)

dy/dx = 1 / 1 + e^(-x) x  (1 - 1 / 1 + e^(-x)


- Linear Derivatives
y = x

dy/dx = 1



Backward Pass (Output -> Hidden Layer 2)

Hampir `sama` `seperti` pada `forward pass`, 
pada `backward pass`, `loss` `akan mengalir` `menuju` `semua node` `pada hidden layer` `untuk dicari gradient nya` `terhadap` `parameter` `yang akan diupdate`. 
Misalkan kita ingin `mengupdate` `parameter` `Wk1o`, `maka kita` `bisa gunakan` `chain rule` `seperti dibawah ini`.

dLoss / dwk1o = dLoss / dOout x dOout / doin x doin / dwk1o

	           1 loss , pred.   2 pred 
Loss = value loss func
Oout = value pred after activation function
Oin = Value (np.dot plus bias)
wk1o = weight in hidden layer k1 to o

Chain Rule

Pertama kita akan mencari berapa besar perubahan Loss berdasarkan output. 
Sehingga kita harus mencari `turunan parsial` `(partial derivative)` `dari loss function` `terhadap` `output`, kita juga `bisa menyebutnya` `sebagai` `gradient loss function` `terhadap output`.

Pada persamaan dibawah, `loss akan dikalikan dengan 1/2`, sebenarnya `tujuannya` `agar saat diturunkan`, 
`fungsi loss` `akan menjadi 1 kali Loss` `(menetralisir turunan fungsi kuadrat)`.


Contoh 1. Jika z = sin^2(x^2y) tentukan a. d z/ d x 
Jawab : a. Misal  z = u^2 , maka d z / d u = 2u = 2sin(x^2y)
	
	  u = sin(x^2y) , maka d u / d x = 2xy cos(x^2y)

	  sehingga : d z/ d x = d z/ d u x d u / d x
	            = 2sin(x^2y) x 2xy cos(x^2y)
	            = 4xy sin(x^2y)cos(x^2y)


LOSS = 1/2 (output - Oout)^2
misal z = 1/2 u^2 , maka d z / d u = 1/2 2u = u  = (output - Oout)
      u = (output - Oout) , maka d u / d Oout = - 1

      sehingga : d z/ d Oout = d z/ d u x d u / d Oout
	            = (output - Oout) x - 1
	            = Oout - output



d Loss/d Oout = d (1/2 (output - Oout)^2) / d Oout

d Loss/d Oout = -1 x 2 x 1/2 (output - Oout)
			  = Oout - output
			  = 2.494 - 3
			  = - 0.506
				  |
				  v
Gradient Loss terhadap Oout


Selanjutnya kita akan mencari `gradient` `dari Oout` `terhadap Oin`. 
Karena `activation function` yang digunakan adalah `Linear`, maka `turunannya` sangat mudah dicari.

Oout = Oin  # Oout adl hasil stlh hitung activ func , Oin adl 
			# hasil hitung jumlah K out , weight dan bias

d Oout/d Oin = d Oin/d Oin
d Oout/d Oin = 1
			   |
			   v
Gradient Oout terhadap Oin


Setelah itu kita akan cari `gradient` `dari Oin` `terhadap Wk1o`, `Wk2o` dan `bias` (bo). Perhatikan persamaan dibawah ini:

Oin = wk1o k1out + wk2o k2out + bo

d Oin/d wk1o = d (wk1o k1out + wk2o k2out + bo) / d wk1o
    	     = k1out

d Oin/d wk2o = d (wk1o k1out + wk2o k2out + bo) / d wk2o
	         = k2out

[k1out k2out] = [0.9975 0.9933]

d Oin/d bo = d (wk1o k1out + wk2o k2out + bo) / d bo
	         = 1


Terakhir kita akan menerapkan chain rule untuk mencari gradient loss terhadap weight dan bias.

dLoss / dwk1o = dLoss / dOout x dOout / doin x doin / dwk1o
			  = - 0.506 x 1 x 0.9975
			  = - 0.50474 

dLoss / dwk2o = dLoss / dOout x dOout / doin x doin / dwk2o
			  = - 0.506 x 1 x 0.9933
			  = - 0.50261

			  	 ^
			     |
Gradient Loss terhadap Weight Hidden Layer 2


dLoss / d bo = dLoss / dOout x dOout / doin x doin / d bo
			  = - 0.506 x 1 x 1
			  = - 0.506
			  		|
			  		v
Gradient Loss terhadap Bias O


Stochastic Gradient Descent (SGD) Update
`SGD` adalah `algoritma` `yang digunakan` `untuk mengupdate` `parameter` `dalam hal ini` `weight` dan `bias`. 
`Algoritmanya` `cukup sederhana`, `pada dasarnya` `kita hanya` `mengurangi` `initial weight` `dengan “sebagian”` `dari` `nilai gradient` `yang sudah kita dapat`.

`Sebagian` `disini` `diwakili` oleh `hyper-parameter` `bernama` `learning rate (alpha)`. 
Sebagai contoh saja, `kita gunakan` `0.25` `sebagai` `learning rate` `meskipun pada prakteknya` `learning rate` `0.25` `itu` `tidak ideal`. 
(Nanti akan dibahas tentang `setting hyper-parameter`).

w'k1o = wk1o - alpha (d Loss /d wk1o) = 1 - 0.25(-0.50474) = 1.1262
w'k2o = wk2o - alpha (d Loss /d wk2o) = 0.5 - 0.25(-0.50261) = 0.6256

b'o = bo - alpha (d Loss /d bo) = 1 - 0.25(-0.506) = 1.1265


wko = [[wk1o],
	   [wk2o]]
    = [[1.1262],
	   [0.6256]]
 bo = 1.1265
	     |
	     v
Parameter baru setelah diupdate


Cukup panjang memang kalau diruntut satu-satu. Tapi gpp biar lebih jelas aja :D. Kita lanjutkan lagi backprop untuk layer selanjutnya.


Backward Pass (Hidden Layer 2 -> Hidden Layer 1)

Kita bisa ulangi setiap step yang kita lakukan pada backward pass pada layer sebelumnya. 
Hanya saja kita harus lebih hati-hati karena relatif lebih rumit daripada backward pass pada layer sebelumnya. 
Hang on guys… :D

dLoss / dwj1k1 = dLoss / dk1out x dk1out / dk1in x dk1in / dwj1k1

Chain Rule again :D


Untuk mencari `gradient loss` terhadap `Wj1k1`, lagi-lagi kita akan gunakan `chain rule`. 
Pertama kita akan mencari `gradient loss` terhadap `K1out` dan `k2out` 

dLoss / dk1out = dLoss / dOout x dOout / dOin x dOin / dwk1o x 	
				 dwk1o / dk1out
			   = -0.506 x 1 x 0.9975 x wk1o lama
			   = -0.506 x 1 x 0.9975 x 1.0
			   = -0.504

dLoss / dk2out = dLoss / dOout x dOout / dOin x dOin / dwk2o x 	
				 dwk2o / dk2out
			   = -0.506 x 1 x 0.9933 x wk2o lama
			   = -0.506 x 1 x 0.9933 x 0.5
			   = -0.25130

Lalu kita akan cari gradient K1out terhadap K1in. Kali ini kita menggunakan turunan dari sigmoid yang sudah kita cari diawal tadi.

Rumus 5 : Turunan Pembagian Fungsi
jika y = f(x) / g(x) maka dy/dx = f'(x)g(x) - g'(x)f(x) / [g(x)]^2

contoh = y = x / x^2 + 1
f(x) = x maka f'(x) = 1
g(x) = x^2 + 1 maka g'(x) = 2x

dy/dx = 1.(x^2 + 1) - 2x(x) / (x^2+1)^2 =  1-x^2 / (x^2+1)^2

Rumus 8 : e^f(x) maka dy/dx = e^f(x).f'(x)
contoh :
y = e^2x+1
f(x) = 2x+1
f'(x) = 2
maka f’ = e^2x+1 . 2 = 2e^2x+1

f(x) = 1 maka f'(x) = 0
  g(x) = 1 + e^-x maka g'(x)= -e^-x 	
  dy/dx = 0.(1 + e^-x) - -e^-x(1) / (1 + e^-x)^2 
        =  e^-x / (1 + e^-x)^2  
        = 1 - 1 + e^-x / (1 + e^-x)^2  
        = 1 + e^-x / (1 + e^-x)^2  -  1 / (1 + e^-x)^2
        = 1 / (1 + e^-x)  -  1 / (1 + e^-x)^2 
        = 1 / (1 + e^-x) (1 -  1 / (1 + e^-x)^2 )

k1out =  1 / 1 + e^-k1in

d k1out / d k1in =  d (1 / 1 + e^-k1in) / d k1in

d k1out / d k1in =  1 / 1 + e^-k1in x ( 1  - 1 / 1 + e^-k1in )

d k1out / d k1in =  1 / 1 + e^-6 x ( 1  - 1 / 1 + e^-6 )
				 =  0.99752737684 x (1-0.99752737684)
				 =  0.00246650929
d k2out / d k2in =  1 / 1 + e^-5 x ( 1  - 1 / 1 + e^-5 )
				 =  0.99330714907 x (1-0.99330714907)
				 =  0.00664805667
[k1in k2in]=[6 5] 

Selanjutnya kita akan cari gradient K1in terhadap Wj1k1.


k1in = wj1k1.j1out + wj2k1.j2out + wj3k1.j3out + wj4k1.j4out + bk1

d k1in / d wj1k1 = d (wj1k1.j1out + wj2k1.j2out + wj3k1.j3out + wj4k1.j4out + bk1) / d wj1k1
				 = j1out

[d k1in / d wj1k1  d k1in / d wj2k1  d k1in / d wj3k1  d k1in / d wj4k1]  = [j1out  j2out  j3out  j4out]
		= [1.5 2.0 2.5 3.0]

[d k1in / d wj1k2  d k1in / d wj2k2  d k1in / d wj3k2  d k1in / d wj4k2]  = [j1out  j2out  j3out  j4out]
		= [1.5 2.0 2.5 3.0]

[bk1 bk2] = [1 1]
          = [-0.0012568026 ....]

Sekarang kita bisa menghitung gradient loss terhadap Wj1k1 dengan menerapkan chain rule yang tadi.

dLoss / dwj1k1 = dLoss / dk1out x dk1out / dk1in x dk1in / dwj1k1
     	       = -0.50474 x 0.00249 x 1.5
     	       = -0.0018852039

dLoss / wj2k1 = dLoss / dk1out x dk1out / dk1in x dk1in / wj2k1
     	       = -0.50474 x 0.00249 x 2.0
     	       = -0.0025136052

dLoss / wj3k1 = dLoss / dk1out x dk1out / dk1in x dk1in / wj3k1
     	       = -0.50474 x 0.00249 x 2.5
     	       = -0.0031420065

dLoss / wj4k1 = dLoss / dk1out x dk1out / dk1in x dk1in / wj4k1
     	       = -0.50474 x 0.00249 x 3.0
     	       = -0.0037704078

Akhirnya kita mendapatkan gradientnya, perhatikan yang `warna merah`. 
`Gradient dari sigmoid` `sudah cukup kecil` yaitu `0.00249` dan `setelah chain rule` `hasilnya tambah kecil` lagi yaitu `-0.00188`
`Fenomena inilah` `yang disebut` dengan `Vanishing Gradient` dan `merupakan alasan` `mengapa sigmoid` `sudah jarang` `digunakan lagi`.

`Perhitungan yang barusan` `kita lakukan tadi` `setelah diterapkan untuk semua parameter` `maka akan didapat` `semua gradient` `yang dibutuhkan untuk melakukan update`.

Kita bisa lihat dibawah ini kalau `gradientnya sangat kecil` `(vanish)`, sehingga 
`semakin dekat` `sebuah node` `dengan input layer`, `maka semakin` `lama pula` `waktu yang dibutuhkan` `untuk melakukan training`, `karena gradient` `yang digunakan` `untuk melakukan update` `sangat kecil` `dan akan bertambah kecil` `lagi setelah` `dikalikan` dengan `learning rate` :)

[[dLoss / dwj1k1, dLoss / dwj1k2],
 [dLoss / wj2k1, dLoss / wj2k2],
 [dLoss / wj3k1, dLoss / wj3k2],
 [dLoss / wj4k1, dLoss / wj4k2]]
= [[-0.0018852039, -0.00252],
  [-0.0025136052, -0.00334],
  [-0.0031420065, -0.00417],
  [-0.0037704078, -0.00501]]

[[dLoss / dbk1, dLoss / dbk2]] 
= [[-0.0012568026, -0.00167]]


SGD Update (Hidden Layer 2 -> Hidden Layer 1)

Weight dan bias yang baru `sangat mudah dicari` `setelah kita menemukan gradientnya`. 
Tetap dengan `learning rate sebesar 0.25` `kita akan mendapatkan weight dan bias yang baru`.

= [[1.00047, 0.00062],
  [0.75062, 0.25083],
  [0.50078, 0.50104],
  [0.25094, 0.75125]]

= [[1.00031, 1.00042]]

`Perubahan weight dan bias sangat kecil`


Backward Pass (Hidden Layer 1 -> Input Layer)

Kita akan lakukan lagi langkah-langkah yang sudah kita pelajari tadi. 
Kali ini kita akan melakukan update terhadap weight dan bias diantara input layer dan hidden layer 1.

Chain Rule
Pertama kita akan mencari gradient loss terhadap J1out. 
Kali Ini lebih rumit daripada perhitungan K1out. 
Karena `J1out` dipengaruhi oleh `gradient` `yang berasal` dari `K2` dan `K1`. Sehingga kita harus melihat `Layer K` `sebagai satu kesatuan`, `bukan lagi` `K1` dan `K2`.

dLoss / dwj1out = dLoss / dkout x dkout / dkin x dkin/ dwj1k x dwj1k/ dwj1out


dLoss / dkout = dLoss / dk1out + dLoss / dk2out = -0.50474 + -0.25130 = -0.75604


dkout / dkin = dk1out / dkin + dk2out / dkin = 0.00249 + 0.00665 = 0.00914


dkin / dwj1k = dk1in / dwj1k1 + dk2in / dwj1k2 = 1.5 + 1.5 = 3.0
dkin / dwj2k = dk1in / dwj2k1 + dk2in / dwj2k2 = 2 + 2 = 4.0
dkin / dwj3k = dk1in / dwj3k1 + dk2in / dwj3k2 = 2.5 + 2.5 = 5.0
dkin / dwj4k = dk1in / dwj4k1 + dk2in / dwj4k2 = 3.0 + 3.0 = 6.0


dwj1k / dwj1out = dwj1k1 / dj1out + dwj1k2 / dj1out
dwj2k / dwj2out = dwj2k1 / dj2out + dwj2k2 / dj2out 

= d (wj1k1.j1out + wj2k1.j2out + wj3k1.j3out + wj4k1.j4out + bk1) / d dj1out + d (wj1k2.j1out + wj2k2.j2out + wj3k2.j3out + wj4k2.j4out + bk1) / d dj1out
= wj1k1 + wj1k2 = 1.0 + 0 = 1.0

= wj2k1 + wj2k2 = 0.75 + 0.25 = 1.0
= wj3k1 + wj3k2 = 0.5 + 0.5 = 1.0
= wj4k1 + wj4k2 = 0.25 + 0.75 = 1.0


dLoss / dwj1out = dLoss / dkout x dkout / dkin x dkin/ dwj1k x dwj1k/ dwj1out
			 = -0.75604 x 0.00914 x 3.0 x 1.0 = -0.02073

			 = -0.75604 x 0.00914 x 4.0 x 1.0 = -0.0276408224
			 = -0.75604 x 0.00914 x 5.0 x 1.0 = -0.02073
			 = -0.75604 x 0.00914 x 6.0 x 1.0 = -0.02073

Lanjut dengan gradient J1out terhadap J1in.

j1out = max(0,j1in)
j1out = max(0,1.5)

d j1out/d j1in  = d (ReLU)/d j1in = {1 utk j1in>0, 0 utk j1in=0} 
			= 1

Selanjutnya kita akan cari gradient J1in terhadap Wij1.

j1in = wij1 i + bj1
d j1in/d wij1 =  d(wij1 i + bj1) / d wij1
d j1in/d wij1 = i
			  = 2.0

Pada akhirnya kita bisa menghitung gradient loss terhadap Wij1 dengan menerapkan chain rule.

dLoss / dwij1 = dLoss / dj1out x dj1out / dj1in x dj1in/ dwij1 
		= -0.02073 x 1 x 2
		= -0.04146

		= -0.0552816448




Perhitungan yang baru saja kita lakukan tadi akan diterapkan untuk semua parameter. 
Maka didapat semua gradient yang dibutuhkan untuk melakukan update.

[d Loss/d wij1, d Loss/d wij2, d Loss/d wij3, d Loss/d wij4] = [-0.04146, -0.05528, -0.06910, -0.08292] 

[d Loss/d bij1, d Loss/d bij2, d Loss/d bij3, d Loss/d bij4] = 
[-0.02073, -0.02764, -0.03455, -0.04146]

SGD Update (Hidden Layer 1 -> Input Layer)

[w'ij1 w'ij2 w'ij3 w'ij4] = [wij1-alpha(dLoss / dwij1) wij2-alpha(dLoss / dwij2) wij3-alpha(dLoss / dwij3) wij4-alpha(dLoss / dwij4)]
= [0.2603, 0.51382, 0.76728, 1.02073]

[b'ij1 b'ij2 b'ij3 b'ij4] = [bij1-alpha(dLoss / dbij1) bij2-alpha(dLoss / dbij2) bij3-alpha(dLoss / dbij3) bij4-alpha(dLoss / dbij4)]
= [1.02073, 1.02764, 1.03455, 1.04146]

Weight dan bias setelah parameter update
Akhirnya selesai juga. Kita sudah mendapatkan semua parameter yang baru. Nanti proses ini (forward pass dan backward pass) akan diulang terus menerus sampai kita dapatkan nilai Loss yang paling kecil.
Old Parameter vs New Parameter

Old Parameters

New Parameters
Contoh diatas hanya menggunakan satu buah data pada saat forward dan backward. 
Secara umum Gradient Descent ini terdiri dari `3 tipe`, `SGD` `yang kita pakai diatas`, `Batch Gradient Descent` dan `Mini-batch Gradient Descent`.

`Pada Batch Gradient Descent (BGD)`, `model` `akan diupdate` `setelah semua data` `seleasai` `di-”propagate”`. 

Sedangkan `Mini-batch` `berada ditengah-tengah` `SGD dan BGD`.

`Mini-batch gradient descent` `melakukan forward` `dan backward pass` `pada sekelompok kecil` `training data`. 
Misalnya `melakukan update` `untuk setiap 32/64 buah data` dan `error yang dihitung` adalah `mean dari sekelompok training data tersebut`.

Pada part ini kita sudah sama-sama belajar tentang backpropagation, gradient dan SGD. 

Nanti di part berikutnya kita akan coba belajar tentang Deep Learning Framework. 

So, stay tune guys… :D
Yes you should understand backprop
When we offered CS231n (Deep Learning class) at Stanford, we intentionally designed the programming assignments to…
medium.com

Mungkin untuk bacaan selanjutnya bisa disimak tulisan `Mas Karpathy` `tentang pentingnya` `kita memahami backprop`. 

Disitu juga `dibahas tentang “dying ReLU”` juga.

Semoga post ini bermanfaat untuk kita semua yang ingin memahami tentang backpropagation.

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
519 claps



Samuel Sena
WRITTEN BY

Samuel Sena
Follow
Deep Reinforcement Learning Student
See responses (9)
More From Medium
Related reads
How Islamophobia Was Ingrained in America’s Legal System Long Before the War on Terror
The Intercept
The Intercept in The Intercept
May 6, 2018 · 13 min read
443

Related reads
Why Big Data And Machine Learning Are Important In Our Society
Terence Mills
Terence Mills in AI.io
Jul 3, 2019 · 5 min read
126

Also tagged Neural Networks
A Beginners Guide to Neural Nets
Shane De Silva
Shane De Silva in Towards Data Science
Mar 20 · 12 min read
51

Discover Medium
Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch
Make Medium yours
Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore
Become a member
Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade
About
Help
Legal