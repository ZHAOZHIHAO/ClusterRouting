# Capsule networks with non-iterative cluster routing

> A pytorch implementation.

[**Capsule networks with non-iterative cluster routing**](https://arxiv.org/abs/2109.09213)  
Zhihao Zhao, [Samuel Cheng](https://samuelcheng.info/)
Elsevier Neural Networks, Volume 143, November 2021. 

Use this bibtex to cite this repository:
```tex
@article{zhao2021capsule,
  title={Capsule networks with non-iterative cluster routing},
  author={Zhao, Zhihao and Cheng, Samuel},
  journal={Neural Networks},
  volume={143},
  pages={690--697},
  year={2021},
  publisher={Elsevier}
}
```
### Prerequisites
This code is tested with the following packages
- Python 3.7
- Pytorch 1.7.0
- torchvision  0.8.0
- CUDA 10.2

#### Training CIFAR10
```bash
python3 main.py --dataset cifar10 --C 4 --K 8 --D 24 --batch_size 64 --save_dst ./checkpoint
```
#### Training CIFAR10 quickly (less parameters)
```bash
python3 main.py --dataset cifar10 --C 4 --K 5 --D 6 --batch_size 64 --save_dst ./checkpoint
```
### Performances
model|CIFAR-10|SVHN|smallNORB
:---:|:---:|:---:|:---:
Dynamic routing|10.6%|4.3%|2.7%
EM routing|11.9%|-|1.8%
Cluster routing|7.37%|3.37%|1.57%
