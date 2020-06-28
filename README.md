# AdaDB: An Adaptive Gradient Method with Data-Dependent Bound

## Requirements 
* Python 3.6.8
* PyTorch 1.2.0 
* CUDA Version 10.0.130 
* GPU: GTX 1080Ti

## Usage
You can use AdaDB just like any other PyTorch optimizers.
```
optimizer = optimizers.AdaDB(model.parameters(), lr=1e-3, final_lr=0.1)
```


## Run Demos on CIFAR-100 

Train model ResNet-18 on CIFAR-100 dataset using optimizer AdaDB. 
```
cd AdaDB
python cifar.py --model=resnet18 --optim=adadb --lr=1e-3 --final_lr=0.1 --gamma=1e-5
```



