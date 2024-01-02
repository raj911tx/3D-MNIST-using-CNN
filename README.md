# 3D-MNIST-using-CNN
Classification of 3D digits of MNIST dataset
A 3D Convolution can be used to find patterns across 3 spatial dimensions; i.e. depth, height and width. One effective use of 3D Convolutions is object segmentation in 3D medical imaging. Since a 3D model is constructed from the medical image slices, this is a natural fit. And action detection in video is another popular research area, where multiple image frames are concatenated across a temporal dimension to give a 3D spatial input, and patterns are found across frames too.

![](https://th.bing.com/th/id/OIP.C-Hh9BBJ1uNp8Redvyku9gHaEj?pid=Api&rs=1)

We get 3d point data from self driving cars as well.Which are called point clouds from lidar sensor.Which can be trained using various methods such as CNN,pointnet,Voxelnet etc.Pointnets has been seen as a better performance over other architecture.But due to its complex architecture and requiring custom layers to construct we are going for CNN in this project.

![](http://spectrum.ieee.org/image/MjgxMTk1OQ.jpeg)
In 2 dimentional image Segmentation we use 2d convolution network here in this project we are trying to use a 3d CNN model for classification.For classifying 3d datasets pointnets are better alternative but it requires creating complex architecture.MNIST dataset consists of digits from 0-9.


```
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    


```
import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"
```


```
%cd /content/gdrive/My Drive/Kaggle
```

    /content/gdrive/My Drive/Kaggle
    

First we are downloading the dataset from kaggle.


```
!kaggle datasets download -d daavoo/3d-mnist
```

    Downloading 3d-mnist.zip to /content/gdrive/My Drive/Kaggle
     90% 138M/153M [00:00<00:00, 168MB/s]
    100% 153M/153M [00:00<00:00, 184MB/s]
    


```
!unzip \*.zip  && rm *.zip
```

    Archive:  3d-mnist.zip
    replace full_dataset_vectors.h5? [y]es, [n]o, [A]ll, [N]one, [r]ename: A
      inflating: full_dataset_vectors.h5  
      inflating: plot3D.py               
      inflating: test_point_clouds.h5    
      inflating: train_point_clouds.h5   
      inflating: voxelgrid.py            
    


```
import numpy as np
import h5py
```

Then we are seperating training and testing data from the dataset.


```
with h5py.File('full_dataset_vectors.h5', 'r') as dataset:
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]
    y_train = dataset["y_train"][:]
    y_test = dataset["y_test"][:]
```


```
print ("x_train shape: ", x_train.shape)
print ("y_train shape: ", y_train.shape)

print ("x_test shape:  ", x_test.shape)
print ("y_test shape:  ", y_test.shape)
```

    x_train shape:  (10000, 4096)
    y_train shape:  (10000,)
    x_test shape:   (2000, 4096)
    y_test shape:   (2000,)
    

We add 3 color channels to data data similar to as we do 3 channels to image set


```
xtrain = np.ndarray((x_train.shape[0], 4096, 3))
xtest = np.ndarray((x_test.shape[0], 4096, 3))
```


```
from matplotlib.pyplot import cm
```


```
def add_rgb_dimention(array):
    scalar_map = cm.ScalarMappable(cmap="Oranges")
    return scalar_map.to_rgba(array)[:, : -1]

for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimention(x_train[i])
for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimention(x_test[i])
```


```
xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)
xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)
```


```
xtrain.shape
```




    (10000, 16, 16, 16, 3)



Now set the lables to one-hot matrix for classification.


```
import keras.utils
```


```
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
```


```
y_train.shape
```




    (10000, 10)




```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout,BatchNormalization
model=Sequential()
model.add(Conv3D(32,(3,3,3),activation='relu',padding='same',input_shape=xtrain.shape[1:]))
model.add(MaxPool3D((2,2,2)))
model.add(Conv3D(64,(3,3,3),activation='relu',padding='same'))
model.add(MaxPool3D((2,2,2)))
model.add(Conv3D(128,(3,3,3),activation='relu',padding='same'))
model.add(MaxPool3D((2,2,2)))
model.add(Conv3D(256,(3,3,3),activation='relu',padding='same'))
model.add(MaxPool3D((2,2,2)))

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv3d (Conv3D)              (None, 16, 16, 16, 32)    2624      
    _________________________________________________________________
    max_pooling3d (MaxPooling3D) (None, 8, 8, 8, 32)       0         
    _________________________________________________________________
    conv3d_1 (Conv3D)            (None, 8, 8, 8, 64)       55360     
    _________________________________________________________________
    max_pooling3d_1 (MaxPooling3 (None, 4, 4, 4, 64)       0         
    _________________________________________________________________
    conv3d_2 (Conv3D)            (None, 4, 4, 4, 128)      221312    
    _________________________________________________________________
    max_pooling3d_2 (MaxPooling3 (None, 2, 2, 2, 128)      0         
    _________________________________________________________________
    conv3d_3 (Conv3D)            (None, 2, 2, 2, 256)      884992    
    _________________________________________________________________
    max_pooling3d_3 (MaxPooling3 (None, 1, 1, 1, 256)      0         
    =================================================================
    Total params: 1,164,288
    Trainable params: 1,164,288
    Non-trainable params: 0
    _________________________________________________________________
    


```
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax")) 
```


```
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv3d (Conv3D)              (None, 16, 16, 16, 32)    2624      
    _________________________________________________________________
    max_pooling3d (MaxPooling3D) (None, 8, 8, 8, 32)       0         
    _________________________________________________________________
    conv3d_1 (Conv3D)            (None, 8, 8, 8, 64)       55360     
    _________________________________________________________________
    max_pooling3d_1 (MaxPooling3 (None, 4, 4, 4, 64)       0         
    _________________________________________________________________
    conv3d_2 (Conv3D)            (None, 4, 4, 4, 128)      221312    
    _________________________________________________________________
    max_pooling3d_2 (MaxPooling3 (None, 2, 2, 2, 128)      0         
    _________________________________________________________________
    conv3d_3 (Conv3D)            (None, 2, 2, 2, 256)      884992    
    _________________________________________________________________
    max_pooling3d_3 (MaxPooling3 (None, 1, 1, 1, 256)      0         
    _________________________________________________________________
    flatten (Flatten)            (None, 256)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               32896     
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 1,198,474
    Trainable params: 1,198,474
    Non-trainable params: 0
    _________________________________________________________________
    


```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(x=xtrain, y=y_train, batch_size=128, epochs=50, validation_split=0.15)
```

    Epoch 1/50
    67/67 [==============================] - 2s 33ms/step - loss: 0.6842 - accuracy: 0.7698 - val_loss: 0.8486 - val_accuracy: 0.7200
    Epoch 2/50
    67/67 [==============================] - 2s 29ms/step - loss: 0.6029 - accuracy: 0.8002 - val_loss: 0.8193 - val_accuracy: 0.7333
    Epoch 3/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.5363 - accuracy: 0.8160 - val_loss: 0.9086 - val_accuracy: 0.7240
    Epoch 4/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.4752 - accuracy: 0.8436 - val_loss: 0.8902 - val_accuracy: 0.7280
    Epoch 5/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.4569 - accuracy: 0.8425 - val_loss: 0.9162 - val_accuracy: 0.7293
    Epoch 6/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.3987 - accuracy: 0.8618 - val_loss: 0.8817 - val_accuracy: 0.7393
    Epoch 7/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.3399 - accuracy: 0.8864 - val_loss: 0.8479 - val_accuracy: 0.7533
    Epoch 8/50
    67/67 [==============================] - 2s 29ms/step - loss: 0.3193 - accuracy: 0.8928 - val_loss: 0.9403 - val_accuracy: 0.7467
    Epoch 9/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.2483 - accuracy: 0.9179 - val_loss: 0.9793 - val_accuracy: 0.7413
    Epoch 10/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.2121 - accuracy: 0.9322 - val_loss: 1.1048 - val_accuracy: 0.7527
    Epoch 11/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.2114 - accuracy: 0.9299 - val_loss: 1.1233 - val_accuracy: 0.7347
    Epoch 12/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.1986 - accuracy: 0.9345 - val_loss: 1.1661 - val_accuracy: 0.7447
    Epoch 13/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.1570 - accuracy: 0.9504 - val_loss: 1.1477 - val_accuracy: 0.7587
    Epoch 14/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.1436 - accuracy: 0.9531 - val_loss: 1.2016 - val_accuracy: 0.7433
    Epoch 15/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.1179 - accuracy: 0.9620 - val_loss: 1.2846 - val_accuracy: 0.7420
    Epoch 16/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.1209 - accuracy: 0.9631 - val_loss: 1.2649 - val_accuracy: 0.7487
    Epoch 17/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0989 - accuracy: 0.9686 - val_loss: 1.4211 - val_accuracy: 0.7453
    Epoch 18/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0840 - accuracy: 0.9720 - val_loss: 1.4027 - val_accuracy: 0.7420
    Epoch 19/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0866 - accuracy: 0.9729 - val_loss: 1.4096 - val_accuracy: 0.7513
    Epoch 20/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0671 - accuracy: 0.9779 - val_loss: 1.3836 - val_accuracy: 0.7473
    Epoch 21/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0790 - accuracy: 0.9718 - val_loss: 1.4367 - val_accuracy: 0.7353
    Epoch 22/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0845 - accuracy: 0.9725 - val_loss: 1.6759 - val_accuracy: 0.7387
    Epoch 23/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0800 - accuracy: 0.9776 - val_loss: 1.5912 - val_accuracy: 0.7367
    Epoch 24/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0866 - accuracy: 0.9716 - val_loss: 1.5191 - val_accuracy: 0.7380
    Epoch 25/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0792 - accuracy: 0.9752 - val_loss: 1.5620 - val_accuracy: 0.7453
    Epoch 26/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0628 - accuracy: 0.9796 - val_loss: 1.5169 - val_accuracy: 0.7433
    Epoch 27/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0494 - accuracy: 0.9851 - val_loss: 1.6465 - val_accuracy: 0.7473
    Epoch 28/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0353 - accuracy: 0.9884 - val_loss: 1.6982 - val_accuracy: 0.7453
    Epoch 29/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0503 - accuracy: 0.9835 - val_loss: 1.7029 - val_accuracy: 0.7487
    Epoch 30/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0421 - accuracy: 0.9869 - val_loss: 1.7695 - val_accuracy: 0.7420
    Epoch 31/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0495 - accuracy: 0.9849 - val_loss: 1.6764 - val_accuracy: 0.7420
    Epoch 32/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0546 - accuracy: 0.9806 - val_loss: 1.5567 - val_accuracy: 0.7373
    Epoch 33/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0461 - accuracy: 0.9854 - val_loss: 1.8278 - val_accuracy: 0.7327
    Epoch 34/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0681 - accuracy: 0.9793 - val_loss: 1.6963 - val_accuracy: 0.7400
    Epoch 35/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0494 - accuracy: 0.9839 - val_loss: 1.8021 - val_accuracy: 0.7513
    Epoch 36/50
    67/67 [==============================] - 2s 31ms/step - loss: 0.0390 - accuracy: 0.9900 - val_loss: 1.6994 - val_accuracy: 0.7447
    Epoch 37/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0380 - accuracy: 0.9898 - val_loss: 1.6630 - val_accuracy: 0.7367
    Epoch 38/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0437 - accuracy: 0.9862 - val_loss: 1.7814 - val_accuracy: 0.7427
    Epoch 39/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0405 - accuracy: 0.9871 - val_loss: 1.7267 - val_accuracy: 0.7440
    Epoch 40/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0226 - accuracy: 0.9935 - val_loss: 2.0664 - val_accuracy: 0.7340
    Epoch 41/50
    67/67 [==============================] - 2s 31ms/step - loss: 0.0733 - accuracy: 0.9779 - val_loss: 1.6446 - val_accuracy: 0.7420
    Epoch 42/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0680 - accuracy: 0.9798 - val_loss: 1.6985 - val_accuracy: 0.7367
    Epoch 43/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0381 - accuracy: 0.9879 - val_loss: 2.0047 - val_accuracy: 0.7273
    Epoch 44/50
    67/67 [==============================] - 2s 31ms/step - loss: 0.0569 - accuracy: 0.9833 - val_loss: 1.7345 - val_accuracy: 0.7413
    Epoch 45/50
    67/67 [==============================] - 2s 31ms/step - loss: 0.0479 - accuracy: 0.9851 - val_loss: 1.7010 - val_accuracy: 0.7447
    Epoch 46/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0383 - accuracy: 0.9889 - val_loss: 1.8370 - val_accuracy: 0.7367
    Epoch 47/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0356 - accuracy: 0.9891 - val_loss: 1.9692 - val_accuracy: 0.7507
    Epoch 48/50
    67/67 [==============================] - 2s 31ms/step - loss: 0.0304 - accuracy: 0.9902 - val_loss: 2.0966 - val_accuracy: 0.7347
    Epoch 49/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0337 - accuracy: 0.9902 - val_loss: 1.8929 - val_accuracy: 0.7293
    Epoch 50/50
    67/67 [==============================] - 2s 30ms/step - loss: 0.0237 - accuracy: 0.9925 - val_loss: 2.1200 - val_accuracy: 0.7513
    


```
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

```




    <matplotlib.legend.Legend at 0x7f9a7e0e09b0>




    
![png](3D_MNIST_using_CNN%20%281%29_files/3D_MNIST_using_CNN%20%281%29_25_1.png)
    



```
model.save('model2.h5')
from google.colab import files
files.download("model2.h5")
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>

