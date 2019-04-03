# Photo-realistic Monocular Gaze Redirection using Generative Adversarial Networks

<img src="https://github.com/HzDmS/gaze_redirection/blob/master/imgs/circle.gif" width="100" height="100" /> &ensp; <img src="https://github.com/HzDmS/gaze_redirection/blob/master/imgs/zed.gif" width="100" height="100" /> &ensp; <img src="https://github.com/HzDmS/gaze_redirection/blob/master/imgs/horizontal.gif" width="100" height="100" /> &ensp; <img src="https://github.com/HzDmS/gaze_redirection/blob/master/imgs/vertical.gif" width="100" height="100" />

Repository of arxiv paper [Photo-realistic Monocular Gaze Redirection using Generative Adversarial Networks](https://arxiv.org/abs/1903.12530)

## Dependencies
 tensorflow == 1.7  
 numpy == 1.13.1  
 scipy == 0.19.1  
 
## Dataset
```Bash
tar -xvf dataset.tar .
```

## VGG-16 pretrained weights
```Bash
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz .
tar -xvf vgg_16_2016_08_28.tar.gz .
```

## Training
```Bash
python main.py --mode train --data_path ./dataset/ --log_dir ./log/ --batch_size 32 --vgg_path ./vgg_16.ckpt
```

## Testing
```Bash
python main.py --mode test --data_path ./dataset/ --log_dir ./log/ --batch_size 32
```
