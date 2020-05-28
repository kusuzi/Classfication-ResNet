# Classfication-ResNet
Using RESNET model for transfer learning to classify data sets on Pytorch
##### 相关文件下载
数据集：链接：https://pan.baidu.com/s/1F7ioRh1lMfeOcVxD8mS8OA 
提取码：2t1w 

预训练文件和我训练好的文件：链接：https://pan.baidu.com/s/1qZEj0369652ffM4sD6TUYA 
提取码：9hen

我只用了一个epoch，也可以多用几个epoch效果会更好
##### 步骤  
- 下载文件或者git clone  
- 新建data文件夹并放入数据集  
- 新建weights文件夹并放入下载的权重文件
- 新建test_images文件夹并放入测试图片  
- 运行train.py文件利用resnet进行迁移训练  
- 运行predict.py文件进行测试模型  


##### 文件信息

  |—— classfication-ResNet   
  |———— data                  （下载的cifar数据集，也可以是其他数据集）    
  |———— test_images          （测试图片，按需求添加文件夹和图片）    
  |———— weights             （权重文件：-pre表示预训练文件）  
  |————model               （ResNet的网络结构）  
  |————Predict_cifar       (测试文件)  
  |————Train_Cifar        (训练文件)  
