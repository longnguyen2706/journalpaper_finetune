# AlexNet + GoogleNet finetune

## Getting started
### Instalation
#### Install anaconda
#### Create new environment
```
   conda create --name keras_theano
```
#### Install Theano
```commandline
    pip install --user git+https://github.com/Theano/Theano.git
```

#### Install Keras v1.2 (not work with v2)
```
pip install keras==1.2
```

#### Switch to theano backend 
```
Edit YOUR_USER_FOLDER/.keras/keras.json 

{
    "epsilon": 1e-07,
    "image_data_format": "channels_last",
    "floatx": "float32",
    "backend": "theano",
    "image_dim_ordering": "th"
}
```

## Reference 
### Alexnet 
https://github.com/duggalrahul/AlexNet-Experiments-Keras/tree/master/convnets-keras

### GoogleNet
https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14