# EKS: Affine-Equivariant Kernel Space Encoding for NeRF Editing

<p align="center">
  <a href="https://arxiv.org/abs/2508.02831"><img src="https://img.shields.io/badge/arXiv-2508.02831-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/MikolajZ/eks_demo_data"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Demo%20Data-yellow" alt="Demo Data"></a>
  <a href="https://huggingface.co/datasets/MikolajZ/eks_data"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Full%20Datasets-yellow" alt="Full Datasets"></a>
  <a href="https://mikolajzielinski.github.io/eks.github.io/"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue" alt="Project Page"></a>
</p>

👋 Hi there!  
If you’ve landed here looking to **edit NeRFs** (Neural Radiance Fields) you’re in the right place. This project lets you 🛠️ interactively modify NeRF scenes with ease.  
Feel free to explore 🔍, clone the repo 📦, and start experimenting 🚀. Contributions, issues, and ideas are always welcome! 💡✨

<p align="center">
  <img src="images/teaser.png" alt="Description" width="800"/>
</p>

<p align="center">
  <img src="images/Ficus_leaf_wind.gif" width="22%" />
  <img src="images/Hotdog_fly.gif" width="22%" />
  <img src="images/pillow_duck.gif" width="22%" />
  <img src="images/Rug_cup.gif" width="22%" />
</p>

# ⚙️ Installation

This project is developed as an extension for Nerfstudio. To get started, please install [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/2adcc380c6c846fe032b1fe55ad2c960e170a215) along with its dependencies. <br>

<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/_images/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nerf.studio/_images/logo.png">
    <!-- /pypi-strip -->
    <img alt="nerfstudio" src="https://docs.nerf.studio/_images/logo.png" width="400">
    <!-- pypi-strip -->
    </picture>
    <!-- /pypi-strip -->
</p>

Simply, install this repo with:
``` bash
pip install -e .
ns-install-cli
```
🚀 This will install the package in editable mode and kick off the Nerfstudio CLI installer to get you all set up and ready to go! 🎉

# Running the demo
To test if everything was installed properly, you can run the `ficus` demo.
First, download the demo dataset.
Install [Git LFS](https://git-lfs.com/) 
```bash
git lfs install
git clone https://huggingface.co/datasets/MikolajZ/eks_demo_data
```

Place the `data` and `blender` folder into `eks`.

```bash
# First train the model with
ns-train eks --data data/ficus --timestamp eks_demo

# Prepare the animation with blender
blender -b blender/ficus/Ficus.blend -P blender/ficus/script.py

# Downscale and reformat animation
eks-export ply-from-obj --batch-folder blender/ficus/plys --gausses-per-face 1 --output-folder outputs/ficus/eks/eks_demo/reference_meshes --ply-mode True

# Export tetrahedron soup
eks-export tetrahedrons --load-config outputs/ficus/eks/eks_demo/config.yml

# Bind triangle soup with animation and get final edits
eks-export ply-from-edits --load-config outputs/ficus/eks/eks_demo/config.yml

# Now you can render the animation
eks-render dataset --load-config outputs/ficus/eks/eks_demo/config.yml --rendered-output-names rgb --output-path edits/eks_demo --selected-camera-idx 50

# Render the video
ffmpeg -framerate 24 -i %05d.jpg -c:v libx264 -pix_fmt yuv420p ficus_wind.mp4
```

# Data preparation
### Synthetic data
By default, our method supports the NeRF Synthetic format. If you want to use your own data you need to put `sparse_pc.ply` in the dataset folder. For synthetic data we used [LagHash](https://github.com/theialab/lagrangian_hashes) to generate the initial point cloud or you can use Gaussian Splats.

### Real data
For real data please follow [Nerfstudio data format](https://docs.nerf.studio/quickstart/data_conventions.html). Like with synthetic data, you'll also need to place `sparse_pc.ply` in the dataset folder to initialize the network with initial Gaussians. We have used [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repository to generate them.

### Replicating results from the paper
If you want to replicate the results that we showed in the paper you can download all our datasets with:
```bash
git lfs install
git clone https://huggingface.co/datasets/MikolajZ/eks_data
```
You will also find there all the config files we have used for training so you can use the same hyperparameters as we did. In [dataset `README.md`](https://huggingface.co/datasets/MikolajZ/eks_data) you will find the description of the dataset content and tips for training the models.

# Training the network

Example train commands:
``` bash
# For nerf synthetic
ns-train eks --data <path_to_dataset>

# For MiP-NeRF but also other real data
ns-train eks-real --data <path_to_dataset>
```

# Rendering results
We use [Blender](https://www.blender.org) for generating our animations. It is important to generate for each frame of your animation an `*.ply` file containing modified Gaussians obtained from the training. You can find them in the output folder of your training under the name `step-<num_steps>_means.ply`. 

> ⚠️  It is very important to use only `*.ply` files since they don't change the order of vertices upon the save.

In the output folder of your trained model (usually named with the timestamp) create `camera_path` folder and put your `*.ply` file there. It is important to name them `00000.ply`, `00001.ply`, `00002.ply`, etc.

Now you are ready to go and you can start rednering. You have two options right here:
### Dataset Render
``` bash
eks-render dataset \
  --load-config outputs/<path_to_your_config.yml> \
  --output-path edits/<output_folder> \
  --rendered-output-names rgb \
  --selected-camera-idx <num_camera_from_test_data>
```
 - If you specify a camera index, all frames will be rendered from that viewpoint.
 - If not, the tool renders from all test-time cameras.

### Camera Path Render
``` bash
eks-render camera-path \
  --load-config outputs/<path_to_your_config.yml> \
  --camera-path-filename <camera_paths_folder> \
  --output-path edits/<output_folder> \
  --output-format images
```
-  The camera path here refers to what’s generated by the Nerfstudio viewer not your `*.ply` animation folder!

# Driving Gausses with a mesh
First thing that you need to do is to export the tertahedron soup from the model. We got you right there and we have prepared this command:
``` bash
eks-export tetrahedrons --load-config outputs/<path_to_your_config.yml>
```
This will create a tetrahedron soup in the project output folder. In this output folder create another folder called `reference_meshes` put the edits of your Gausses in this folder in the same format as for rendering. Now this is important. Make sure that `triangle_soup.ply` and `00000.ply` represent your model in the exact same state (no modifications) since we will use those two as a reference. Other `*.ply` files can be in any configuration.

Now run the script:
``` bash
eks-export ply-from-edits --load-config outputs/<path_to_your_config.yml>
```
This will drive the Gausses and will create `camera_path` folder. Now you can proceed to normal rendering mentioned above.

# Citations
If you found this work usefull, please consider citing:

``` bibtex
@misc{zielinski2025eks,
  title     = {Affine-Equivariant Kernel Space Encoding for NeRF Editing},
  author    = {Miko\l{}aj Zieli\'{n}ski and Krzysztof Byrski and Tomasz Szczepanik and Dominik Belter and Przemys\l{}aw Spurek},
  year      = {2025},
  eprint    = {2508.02831},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2508.02831}
}
```