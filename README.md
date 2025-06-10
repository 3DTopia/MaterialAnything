# Material Anything: Generating Materials for Any 3D Object via Diffusion



### [Project Page](https://xhuangcv.github.io/MaterialAnything/) |   [Paper](https://arxiv.org/abs/2411.15138)

https://github.com/user-attachments/assets/a3cb8213-e767-4972-861a-e15d08e14823

Material Anything:</b> A PBR material generation model for various 3D meshes, including <span style="color: #4E95D9;">texture-less</span>, <span style="color: #F2AA84;">albedo-only</span>, <span style="color: #8ED973;">generated</span>, and <span style="color: #D86ECC;">scanned</span> objects.

---
## News

- [2024.12] Release [Material3D Dataset](dataset/) and the [rendering script](rendering_scripts/).
- [2025.06] Release [Material Estimator](https://huggingface.co/xanderhuang/material_estimator) and the [Material Refiner](https://huggingface.co/xanderhuang/material_refiner).
## Inference Setup  (Newly Updated) 
The setup is the same as the original [Text2Tex](https://github.com/daveredrum/Text2Tex). After install PyTorch and PyTorch3D, You can install dependencies:
```bash
pip install -r requirements.txt
```
## Download Models
You can download our models on HuggingFace: [material_estimator](https://huggingface.co/xanderhuang/material_estimator) and [material_refiner](https://huggingface.co/xanderhuang/material_refiner). We provide the script to download load these models.
```sh
bash ./download_models.sh
```
After downloading, the `pretrained_models/` is structured like:
```
./pretrained_models
├── material_estimator/
└── material_refiner/
```

## Quickstart

The `bash` directory contains tools for generating PBR materials for a texture-less mesh. The workflow involves three main steps:
1. Generating **albedo**, **roughness**, **metallic**, and **bump** maps
2. Rendering videos using Blender
You can run the entire pipeline with a single command. For example:
```sh
bash ./bash/test.sh
```

## Material3D Dataset

**Material3D** consists of 80K+ curated 3D objects with high-quality material maps and annotated text descriptions.  
- [Object List](dataset/material3d.json) gives IDs for curated 3D objects in Objaverse.   
- [Caption List](https://github.com/3DTopia/3DTopia/releases/download/data/3DTopia-objaverse-caption-361k.json) gives corresponding text descriptions of the 3D objects.   
- [Render Script](rendering_scripts) gives Blender scripts that render 3D objects into material view-maps and material UV-maps. Refer to the following instructions for the rendering.

---

### Install Blender  

Our Blender scripts are based on **Blender 3.2.2**. While newer Blender versions have been tested, some node names have changed, which may cause compatibility issues. It is recommended to install Blender 3.2.2.  

```bash
# Download Blender 3.2.2
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
rm blender-3.2.2-linux-x64.tar.xz

# Add Blender to PATH in ~/.bashrc
export PATH="/path/to/blender-folder:$PATH"

# Refresh the environment variables
source ~/.bashrc
```

---

### Material Rendering  

The following commands can be used to render material maps:  

#### Render Multi-view Material Maps  
```bash
blender -b -P ./rendering_scripts/blender_script_material.py -- \
    --object_path "./my_object.glb" \
    --output_dir './dataset/outputs' \
    --render_space 'VIEW'
```

#### Render UV-space Material Maps  
```bash
blender -b -P ./rendering_scripts/blender_script_material.py -- \
    --object_path "./my_object.glb" \
    --output_dir './dataset/outputs' \
    --render_space 'UV'
```

---

### Distributed Material Rendering  

For large-scale rendering, we provide a distributed rendering script. You can modify it based on your dataset and system configuration.  

#### Example Usage:  
```bash
python rendering_scripts/distributed_render.py \
    --timeout 3600 \
    --num_gpus 8 \
    --workers_per_gpu 12 \
    --input_models_path './models_path_all.json' \
    --resolution 512 \
    --render_space 'VIEW'
```

---

### Notes  

- Ensure that the Blender binary path is correctly added to your `PATH` environment variable before executing any scripts.  
- For distributed rendering, adjust the `--num_gpus`, `--workers_per_gpu`, and `--timeout` parameters based on your hardware setup.  
- Use the `--render_space` flag to specify whether to render in `VIEW` or `UV` space.  

---


## Abstract
>We present <b>Material Anything</b>, a fully-automated, 
unified diffusion framework designed to generate physically-based materials for 3D objects. 
Unlike existing methods that rely on complex pipelines or case-specific optimizations, 
Material Anything offers a robust, end-to-end solution adaptable to objects under diverse lighting conditions. 
Our approach leverages a pre-trained image diffusion model, 
enhanced with a triple-head architecture and rendering loss to improve stability and material quality. 
Additionally, we introduce confidence masks as a dynamic switcher within the diffusion model, 
enabling it to effectively handle both textured and texture-less objects across varying lighting conditions. 
By employing a progressive material generation strategy guided by these confidence masks, 
along with a UV-space material refiner, our method ensures consistent, UV-ready material outputs. 
Extensive experiments demonstrate our approach outperforms existing methods across a wide range of object categories 
and lighting conditions.

## Overview
<div class="half">
    <img src="assets/pipeline.jpg" width="1080">
</div>
<b>Overview of Material Anything.</b> For texture-less objects, 
we first generate coarse textures using image diffusion models. 
For objects with pre-existing textures, we directly process them. 
Next, a material estimator progressively estimates materials for each view from a rendered image, 
normal, and confidence mask. The confidence mask serves as additional guidance for illuminance uncertainty, 
addressing lighting variations in the input image and enhancing consistency across generated multi-view materials. 
These materials are then unwrapped into UV space and refined by a material refiner.

## Acknowledgement
We would like to thank [Text2Tex](https://github.com/daveredrum/Text2Tex) or providing an excellent and powerful codebase for texture generation. A significant portion of our code is based on their repository.

## Citation
If you find this work helpful for your research, please cite:
```
@article{huang2024materialanything,
  author = {Huang, Xin and Wang, Tengfei and Liu, Ziwei and Wang, Qing},
  title = {Material Anything: Generating Materials for Any 3D Object via Diffusion},
  journal = {arXiv preprint arXiv:2411.15138},
  year = {2024}
  }
```
