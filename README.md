# Surface-Defect-Detection-Based-Yolov8
基于yolov8、pyqt5与opencv的缺陷检测系统

使用yolo进行训练与推理的时候，均采用默认img_size，并区分远景与近景两部分，分别进行检测。近景的图片是通过yolo识别之后，再根据label数据标签编写脚本自动裁剪得到。

## 系统界面

![](.\pic1.png)



## 检测结果

远景：

![./pic2.png](.\pic3.png)

近景：

![](.\pic2.png)