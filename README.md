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
## Run 
### Local 
```commandline
PYTHONPATH='.' python3 finetune_master.py \
    --pool_dir  '/home/long/Desktop/Hela_split_30_2018-12-04.pickle' \
    --image_dir  '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG' \
    --architecture 'alexnet' \
    --start_pool  0 \
    --end_pool 0 \
    --log_dir '/home/long/journal_paper_finetune/log' \
    --save_model_dir  '/home/long/journal_paper_finetune/saved_models' \
    --result_dir '/home/long/journal_paper_finetune/results' \
    --train_batch  16 \
    --test_batch  32 \
    --is_augmented True >>/home/long/Desktop/finetune_log.txt
```
### Lab computer
PYTHONPATH='.' python3 finetune_master.py \
    --pool_dir  '/home/duclong002/Desktop/Hela_split_30_2018-12-04.pickle' \
    --image_dir  '/home/duclong002/Dataset/JPEG_data/Hela_JPEG' \
    --architecture 'googlenet' \
    --start_pool  0 \
    --end_pool 0 \
    --log_dir '/home/duclong002/journal_paper_finetune/log' \
    --save_model_dir  '/home/duclong002/journal_paper_finetune/saved_models' \
    --result_dir '/home/duclong002/journal_paper_finetune/results' \
    --train_batch  16 \
    --test_batch  32 \
    --is_augmented True >>/home/duclong002/Desktop/finetune_log.txt

## Reference 
### Alexnet 
https://github.com/duggalrahul/AlexNet-Experiments-Keras/tree/master/convnets-keras

### GoogleNet
https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14