# SpineNet

by@xsy, @fxb

CV Project. My implement of SpineNet, with PyTorch

[项目链接🔗](https://github.com/41xu/SpineNet)

⚠️warning: ATTENTION PLEASE⚠️

由于mmdetection过于难用，各种版本的问题一大堆，就算用conda create env制定版本，或者下载重新编译都还是会有这样或者那样的问题。而且----by real助教"用mmdetection改改配置文件跑通怎么能叫复现呢！"

所以本项目其实是在阅读了mmdetection(更具体的该说是open-mmlab)的源码之后根据mmdetection的项目结构和流程，加上各种部件的build流程自己复现了一套差不多的东西出来。

这个项目不需要你cp config, mmdet到git clone的mmdetection中，也不需要管理各种mmcv,mmcv-full, mmdet, balabala。使用流程如下（对！没错！就是这么简单的一个东西！）


## Install

pip3 install -r requirements.txt（可能还有一些其他依赖要装，出来什么提示装什么提示就好了）

## Usage


### train model
```
python3 train.py [optional arguments] # TODO，之后再加argument说明，也可以直接看parse里args都有啥，看着填
```
### cal FLOPs
```
python3 flops.py [optional arguments]
```
### evaluation
```
python3 test.py [optional arguments] # TODO, test AP以及paper里的一些之后有时间加一下。
```

## Preparation

### data preparation

数据准备上，可以使用`COCO, Cityscapes, Pascal VOC2007/VOC2012`进行训练。由于我们按照mmdetection的模型配置和launch的流程进行的代码编写，所以dataset的准备和mmdetection支持的是一样的。

[COCO Dataset Download bash](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)

cityscapes,Pascal数据集使用时，先自己到官网上下载，之后执行`utils/dataset/cityscapes.py, utils/dataset/pascal_voc.py`

```
pip3 install cityscapesscripts # 我记得这里还要isntall mmcocodataset之类的一个东西
python3 utils/dataset/cityscapes.py CITYSCAPES_FOLDER --nproc 8 -o CITYSCAPES_FOLDER/annotations
python3 utils/dataset/pascal_voc.py VOC_FOLDER
```

之后的data文件目录应该是这样的(train, test里面分好了class，每个subfolder里就是image和说明)

```
.── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
```

### configuration 

在`config/`下对应不同结构的spinenet的文件中进行修改，以`dict()`的形式读入，注意data之类的自己看着修改路径。btw由于时间仓促有些normlization, optimizer之类的可选择性非常少，这里都是按照default的那个方式写的，而且基本都写死了，想用其他模型/组件自己修改吧。

