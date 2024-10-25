# YOLOv8目标检测在RK3588的部署与加速

1. Pytorch转ONNX
2. ONNX转RK3588
3. YOLOv8网络改动
4. 优化加速

## YOLOv8测试

### 一、YOLOv8环境配置
主要是RKNN-Toolkit2环境，用来转换RKNN
1. 常规安装yolov8环境
2. 安装RKNN-Toolkit2
下载RKNN-Toolkit2
```bash
#github地址,我这里用的是2.1.0版本，我的系统是ubuntu18.04
https://github.com/airockchip/rknn-toolkit2/releases/tag/v2.1.0
```

```bash
# 解压，进入packages目录
cd rknn-toolkit2/packages

# 安装依赖环境,根据自己的Python版本选择requirements
pip install requirements_cp38-2.1.0.txt

# 安装 RKNN-Toolkit2 根据自己的Python版本选择
pip install rknn_toolkit2-2.1.0+708089d1-cp38-cp38-linux_x86_64.whl
```
### 二、YOLOv8安装与部署
1. 安装YOLOv8
```bash
# 创建工程文件夹
mkdir yolo
cd yolo
# 克隆并安装
git clone https://github.com/ultralytics/ultralytics.git

# 进入目录
cd ultralytics/

# 切换指定分支：YOLOv8
git checkout a05edfbc27d74e6dce9d0f169036282042aadb97

# 应用patch. 
git apply enpei.modify.patch

```
新建权重文件夹weights,下载coco数据的原始权重并复制到该文件夹,这里用yolov8s.pt、yolov8n.pt

2. 导出ONNX模型

修改```export_onnx.py```中的模型为自己需要用来导出onnx的模型
```
# 加载自己的模型路径，这里以yolov8s.pt为例
model = YOLO('weights/yolov8s.pt')
# 加载模型配置文件，注意需要匹配
model = YOLO('yolov8s.yaml')
```
运行```export_onnx.py```导出ONNX
```
python export_onnx.py
```
这样会得到```yolov8.dict.onnx```文件

3. 将ONNX转化为RKNN模型及量化

量化流程在onnx转为rknn模型的过程中。模型量化后将使用更低的精度（如int8/int16）保存模型的权重，部署后可减少模型的内存占用空间，加快推理速度，我经过测试量化后推理速度可以快一倍。
**NPU浮点算力比较弱，针对量化模型NPU有更好的优化，所以量化模型性能会比浮点模型好很多。**

```bash
 #  Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', optimization_level = 2, target_platform='rk3588')


    # Load ONNX model
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=["reg1","cls1","reg2","cls2","reg3","cls3"])
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    # Build model
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    # Export RKNN model
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)

#   release rknn
    rknn.release()
```
具体步骤为：
- 初始化RKNN对象
```
# verbose 参数指定是否要在屏幕上打印详细的日志信息
rknn = RKNN(verbose=False)
```
- RKNN模型配置（在构建RKNN模型之前，需要对模型进行通道均值、量化图片RGB2BGR转换、量化类型等的配置）
```
# mean_values:输入的均值，表示一个输入的三个通道的值减去0
# std_values:输入的归一化值，表示设置一个输入的三个通道的值减去均值以后再除以255
# target_platform: 指定目标芯片

rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', optimization_level = 2, target_platform='rk3588')
```
- 加载ONNX模型
```bash
# model 模型文件路径
ret = rknn.load_onnx(model=ONNX_MODEL)
```
- 构建RKNN模型
```bash
# do_quantization：是否对模型进行量化，默认为True
# dataset: 用于量化校正的数据集，量化图片建议选择和预测图片较吻合的图片
ret = rknn.build(do_quantization=True, dataset='./dataste.txt')
```
RKNN-Toolkit2在量化过程中，会根据量化数据集，计算推理结果所需要的量化参数。基于该原因，校准数据集里的数据最好是从训练集或验证集中取一个有代表性的子集（该子集足够覆盖验证集的数据的分布范围即可），建议数量在50~200张之间。如果校准数据集选择的不合适，会导致量化后的数据分布范围与验证集相差较多，那模型的量化精度就会比较差。
量化数据集```'./dataste.txt'```要求：

**每一行一张图片路径**
```
./bus.jpg
./bus1.jpg
./bus2.jpg
...
./bus50.jpg
```
**图片格式：** jpg或png格式

**是否需要缩放到固定尺寸？** 不需要，RKNN-Toolkit2会自动对这些图片进行缩放处理。但是缩放操作也可能会使图片信息发生改变，对量化精度产生一定影响，所以最好使用尺寸相近的图片。

**量化图片建议选择和预测图片较吻合的图片**

**建议数量在50~200张之间**
- 导出RKNN模型

```bash
# Export RKNN model
ret = rknn.export_rknn(RKNN_MODEL)
```
- 其他
```bash
# 如果自定义数据集，记得更换CLASS类别
CLASSES = ['pedestrians','riders',...,'crowd']
```
量化步骤：
```bash
# 生成量化图片
# 在视频中随机挑选100帧画面作为校准图片，并保存进datasets.txt
ffmpeg -i bj_full.mp4 ./img/frame_%d.jpg
ls ./img/*.jpg | shuf -n 100 > datasets.txt

```


进入```convert_rknn```文件夹，
```
cd ../convert_rknn/
```
将onnx导出为rknn模型 
```
python convert_rknn.py 
```
把转化得到的```yolov8s.dict.rknn```,拷到RK3588开发板的```yolov8_letterbox```,
在RK3588开发板，编译yolov8_letterbox,运行测试代码
```bash
# 浮点版本
./build/yolov8_img ./weights/yolov8s.float.rknn ./media/000057.jpg
```
一切正常的话，可以看到对应的检测结果result.jpg。

NPU资源监控指令
```
watch -n 1 "cat /syskernel/debug/rknpu/load"
```
RK3588有三个NPU核心，支持三核合作、双核合作以及独立工作。


### 更换激活函数
原版YOLOv8使用的及激活函数是SiLU,我们换成计算更高效的ReLU(可能会影响一点精度)
- 注意更改激活函数后，原有的pytorch模型需要重新训练再导出ONNX。
激活函数修改流程：
```bash
#下载ultralytics-8.0.01.tar.gz(其他版本会有问题)
#解压，进入目录
cd ultralytics/
#修改激活函数
```
改动细节：激活函数```ultralytics-8.0.101/ultralytics/nn/modules/conv.py```第28行 ```default_act=nn.SiLU()```修改为```default_act=nn.ReLU()```,然后再按照yolov8标准的训练步骤训练即可，**注意训练时不需要指定预训练模型，而是从头训练**
### 三、YOLOv8训练自己的数据集
- 数据集制作

1、使用```tool/yolo_data.py```，将标注好的xml文件,转成标签文件

2、使用```tool/spilit_data.py```，划分数据集

- 训练模型

1、准备好ymal配置文件和```train_s.py```放在同一个目录
```bash
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /home/shorwin/Demo/yolo/datasets/Military_Target # dataset root dir，数据集根目录，使用绝对路径
train: images/train  # train images (relative to 'path') ，训练集图片目录（相对于path）
val: images/val  # val images (relative to 'path') ，测试集图片目录（相对于path）
test:  # test images (optional

# Classes，类别: 0:pedestrians, 1:riders, 2:partially-visible persons, 3:ignore regions, 4:crowd
names:
    0: human
    1: truck
    2: vehicle
    3: house
    4: tank
```

2、使用```train_s.py```训练模型
```
python train_s.py
```
**注意如果修改了激活函数，就不需要加载预训练权重，需要从头训练**

训练完成以后，在按照上述的方法导出RKNN模型：
pt模型导出ONNX
- Git clone yolov8
- 应用patch，扔掉decode,修改模型
- 修改ReLU激活函数
- 导出onnx
```bash
# 如果是自定义数据集，需要修改./ultralytics/ultralytics/models/v8/yolov8.yaml
nc:5 #number of classes
```
- 检查一下onnx激活函数是否变化，使用netron工具,安装
```
pip install netron
```
ONNX导出RKNN
- 设定校准集
- 导出量化的RKNN
```bash
# 如果自定义数据集，记得更换CLASS类别
CLASSES = ['pedestrians','riders',...,'crowd']
```
然后在板子上测试

- 更改数据集对应的参数：
```bash
# 位置1：src/process/postprocess,cpp 第59行
static int class_num=5;

# 位置2： src/task/yolov8_custom.coo 第8行
static std::vector<std::string> g_classes = {"pedestrians", "riders", "partially-visible-person", "ignore-regions", "crowd"};
```
- 验证
```bash
#ReLU激活函数
./build/yolov8_img ./weight/xxx.init.rknn ./media/000057.jpg
```
如果可以检测图片，说明工作正常

