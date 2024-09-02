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


## Model Architecture
![Model](./static/model.jpg)

## Self-Supervised Learning Training Approach 
![training](./static/SSL.png)


## Reuslts

### AVAtt Painting Dataset (Ours)
 ![qual-avatt](./static/visualization.jpg)


# Citation

Please cite the following papers for this project: 

```bibtex
@InProceedings{Tliba_2022_CVPR,
    author    = {Tliba, Marouane and Kerkouri, Mohamed Amine and Chetouani, Aladine and Bruno, Alessandro},
    title     = {Self Supervised Scanpath Prediction Framework for Painting Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1539-1548}
}
```

# Intructions 


The repo relies on <a href="https://pytorch.org/get-started/locally/">Pytorch</a>, and <a href="https://pytorch.org/get-started/locally/">torchvision</a> libraries.




