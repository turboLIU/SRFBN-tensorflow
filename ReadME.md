# SRFBN based on TensorFlow(基于TF的超分模型)
## 算法论文： Feedback Network for Image Super-Resolution [[arXiv]](https://arxiv.org/abs/1903.09814) [[CVF]](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.html) [[Poster]](https://drive.google.com/open?id=1TcDk1RvUCIjr6KvQplaen8yq15LOBJwb)

##### *"With two time steps and each contains 7 RDBs, the proposed GMFN achieves better reconstruction performance compared to state-of-the-art image SR methods including RDN which contains 16 RDBs."*

This repository is TensorFlow code for my proposed SRFBN.

```

## Contents
1. [Requirements](#Requirements)
2. [Test](#test)
3. [Train](#train)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

## Requirements
- Python 3
- skimage
- tensorflow==1.14 (tested)
- cv2 (pip install opencv-python)

## Test

#### Quick start
```

2. Download pre-trained models:
   ```
   Sorry~~~~~~~~~~~  no pre-trained model!
   But you can train network from zero, easy trained

   ```
    
3. Train:

   ```shell
   # SRFBN
   python train.py
   
   # SRFBN-S
   if you want train SRFBN-S , change relate params in config.py
   set num_steps=3, num_groups=4, num_features=32 and so on
   then python train.py
    自行修改config的参数，运行train.py就行了
   ```

4. Datas：
   ```shell
   Put image_path in txt file
   e.g.
       a/b/c/img1.jpg
       a/b/c/img2.jpg
       ...
       a/b/c/imgn.jpg
   change imgtxts in train.py
   把图片路径保存成txt文件中，格式如上描述，修改train.py中的imgtxts参数即可
   ```

### Performance

<p align="center">
    <img src="https://github.com/turboLIU/SRFBN-tensorflow/blob/master/city.gif" width="720">
    <br>
</p>

<p align="center">
    <img src="https://github.com/turboLIU/SRFBN-tensorflow/blob/master/country.gif" width="720">
    <br>
</p>

#### Test on standard SR benchmark

1. Test
    ```shell
    test file not ready, you can code by yourself, it`s very easy!
    测试程序暂时没写，你们自己写吧，很简单的~ 
    或者等我吧模型训练好了，和模型一起上传！
    ```
   
# Notice Important！
## star my code if it helps you 
## 记得点赞哦亲~

# Most Important！！！
## if it helps you a lot, please support me





