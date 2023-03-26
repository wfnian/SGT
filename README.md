## 论文 <i>SGT: A Generalized Processing Model for 1-D Remote Sensing Signal Classification</i>

本库是对论文[SGT: A Generalized Processing Model for 1-D Remote Sensing Signal Classification](https://ieeexplore.ieee.org/document/9964184)的 pytorch 版本的原始代码。

<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/226499603-ac678075-5cdc-4fdd-a15c-102759936322.svg" height = "360" alt="" align=center />
<br><br>
<b>图 1.</b> SGT论文框架.
</p> 
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/226499695-8eaf134b-71a9-4ca5-885f-a0e3cf8e7374.svg" height = "360" alt="" align=center />
<br><br>
<b>图 2.</b> shift模块计算过程.
</p> 
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/226499776-fe035d6b-f8a4-44b9-9ba9-62a7fcae187b.svg" height = "360" alt="" align=center />
<br><br>
<b>图 3.</b> grad 模块计算过程.
</p> 
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/226499853-8367b409-dd12-4dfc-9fa6-3042754b0279.svg" height = "360" alt="" align=center />
<br><br>
<b>图 4.</b> patch split的改进.
</p> 
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/226499927-9ae9b01e-80b9-4427-af1f-32f3429a14bf.svg" height = "360" alt="" align=center />
<br><br>
<b>图 5.</b> 区别非连续性数据，音频数据不可用.
</p> 
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/226500054-55d6c44d-b2ec-400f-8fcf-e941d8fea560.svg" height = "360" alt="" align=center />
<br><br>
<b>图 6.</b> 在磁异常信号数据上的效果.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/182008386-d1eaa3e8-cf89-49fc-89b9-01e4f6b37b42.png" height = "360" alt="" align=center />
<br><br>
<b>图 7.</b> 高光谱不连续数据集上的效果.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/182008397-229d0327-4f76-4559-ab9c-a353b18fb8e7.png" height = "360" alt="" align=center />
<br><br>
<b>图 8.</b> 心电图MiT-BiH数据集上的效果.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/182008405-009b230f-87b4-425a-b311-d98b1b98acc3.png" height = "360" alt="" align=center />
<br><br>
<b>图 9.</b> shift和grad设置的消融实验.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/182008414-be5c0c92-8998-4f9d-a139-d2c4f68dc79a.png" height = "360" alt="" align=center />
<br><br>
<b>图 10.</b> 消融实验.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/182008422-3aaad689-c184-4ef4-bf6c-89e7d82aea18.png" height = "360" alt="" align=center />
<br><br>
<b>图 11.</b> 高光谱分类效果.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/18214955/182008433-b1e40fcc-c376-42a1-8084-f2409d7cd020.png" height = "360" alt="" align=center />
<br><br>
<b>图 12.</b> 磁异常检测的对比试验.
</p>
      

## 论文错误更正

1. Section II.D

$$ Embedding\_{grad}(g) = g \cdot tanh(V^T)$$

$$ Embedding\_{grad}(g) = g \times tanh(V^T) $$

## Citation
如果您发现此存储库对您的研究有用，请考虑引用以下论文

```bibtex
@ARTICLE{9964184,
  author={anonymity},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={SGT: A Generalized Processing Model for 1-D Remote Sensing Signal Classification},
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3224933}}
```
