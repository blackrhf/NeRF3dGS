# 基于 NeRF、TensoRF 和3D Gaussian Splatting的物体重建和新视图合成

TensoRF和NeRF运行环境统一说明：

conda create -n TensoRF python=3.8

conda activate TensoRF

pip install torch torchvision

pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard

-----------------------------------------------
NeRF:

训练：python run_nerf.py --config configs/yourowndata.txt --expname my_experiment

查看过程曲线：tensorboard --logdir=./logs/my_experiment

只渲染：python run_nerf.py --config configs/youowndata.txt --ft_path ./logs/my_experiment/030000.tar --render_only

--------------------------------------------------
TensoRF：

训练：python train.py --expname my_experiment

查看过程曲线：tensorboard --logdir=./log/my_experiment

只渲染：python train.py --config configs/your_own_data.txt --ckpt log/my_experiment/my_experiment.th --render_train 1 --render_only 1

render_train可以换成render_test
