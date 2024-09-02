# SSLArtScanpath
---


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

----

![CVPR 2022](./static/CVPR-22png.png)
![CVF](./static/CVF.png)


## Abstract 
In our paper, we propose a novel strategy to learn distortion invariant latent representation from painting pictures for visual attention modelling downstream task. In further detail, we design an unsupervised framework that jointly maximises the mutual information over different painting styles. To show the effectiveness of our approach, we firstly propose a lightweight scanpath baseline model and compare its performance to some state-of-the-art methods. Secondly, we train the encoder of our baseline model on large-scale painting images to study the efficiency of the proposed self-supervised strategy. The lightweight decoder proves effective in learning from the self-supervised pre-trained encoder with better performances than the end-to-end fine-tuned supervised baseline on two painting datasets, including a proposed new visual attention modelling dataset.


## Model 
![Model](./static/model.jpg)


## Reuslts
### MIT1003 Dataset
 ![qual-mit1003](./static/qual-mit1003.png)

### Le Meur Painting Dataset
 ![qual-lemeur](./static/qual-lemeur.png)

### AVAtt Painting Dataset (Ours)
 ![qual-avatt](./static/qual-avatt.png)


# Citation

Please cite the following papers for this project: 

```bibtex
@inproceedings{10.1145/3549555.3549597,
author = {Kerkouri, Mohamed Amine and Tliba, Marouane and Chetouani, Aladine and Bruno, Alessandro},
title = {A domain adaptive deep learning solution for scanpath prediction of paintings},
year = {2022},
isbn = {9781450397209},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3549555.3549597},
doi = {10.1145/3549555.3549597},
booktitle = {Proceedings of the 19th International Conference on Content-Based Multimedia Indexing},
pages = {57–63},
numpages = {7},
keywords = {Paintings., Scanpath Prediction, Unsupervised Domain Adaptation},
location = {Graz, Austria},
series = {CBMI '22}
}
```

# Intructions 

To run the model it is preffered to : 
1.  create a virtual envirement using (```venv``` or ```conda```)

2. follow the instruction to install the appropriate <a href="https://pytorch.org/get-started/locally/"> ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) </a> and Torchvision verions on you station.

3. clone the repository: 
```bash
git clone https://github.com/kmamine/SP_Gen.git
```

4. install dependencies : 
```bash 
cd ./SP_Gen/
pip install -r requirements.txt
```

The repo relies on <a href="https://pytorch.org/get-started/locally/">Pytorch</a>, <a href="https://pytorch.org/get-started/locally/">torchvision</a>, and <a href="https://kornia.github.io/">kornia</a> libraries.




