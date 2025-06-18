# **Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting**



We propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under
single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100k FPS and uses the
lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel
spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal
strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that
SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal
quality.


![Overview of SwiftWRF.](./img/a.png)

![Visual comparison of SwiftWRF.](./img/b.png)
## **Requirements**

Before running the code, install the required dependencies:

```
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

The `gsplat` module is implemented based on [GaussianImage](https://github.com/Xinjie-Q/GaussianImage). Please ensure it is correctly installed before proceeding.

## **Dataset**
Before running the experiments, you need to download the **NeRF2-s23** and **Our customized** datasets.  



- **NeRF2-s23**: See [NeRF2](https://github.com/XPengZhao/NeRF2?tab=readme-ov-file)

After downloading, place the dataset inside the `datasets/` directory before proceeding with training or evaluation.

## **Startup**
To run **SwiftWRF**, execute the following command:

```
./scripts/D2GV/train.sh <yourpath>/datasets/nerf2
```

To run **SwiftWRF** with loaded models, add the following command to ./scripts/D2GV/train.sh:

```
--model_path "./checkpoints/"
```
## **TODO**
-Release customized datasets.


-Release training code for RSSI prediction.

## **Citation**
If you find our work helpful, please consider citing our work:
```
@misc{liu2025rasterizingwirelessradiancefield,
      title={Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting}, 
      author={Mufan Liu and Cixiao Zhang and Qi Yang and Yujie Cao and Yiling Xu and Yin Xu and Shu Sun and Mingzeng Dai and Yunfeng Guan},
      year={2025},
      eprint={2506.12787},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.12787}, 
}
```


## Acknowledgement
Some source code of ours is borrowed from [GaussianImage](https://github.com/Xinjie-Q/GaussianImage) and [D-3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians). We sincerely appreciate the excellent works of these authors.
