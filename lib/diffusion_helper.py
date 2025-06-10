import torch

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

# Stable Diffusion 2
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline, 
    EulerDiscreteScheduler
)
import sys
# sys.path.append("..")  # Add the parent directory to the Python path
# from MVPBR.pipelines.pipeline_mvdiffusion_albedo_unet import MVDiffusionAlbedoPipeline

# customized
sys.path.append(".")

from pipelines.pipeline_stable_diffusion_switcher import StableDiffusionPipeline as MVDiffusionAlbedoPipeline
from pipelines.pipeline_stable_diffusion_uv import StableDiffusionPipeline as UVDiffusionPipeline
from models.scheduling_ddpm import DDPMScheduler
from models.ControlNet.gradio_depth2image import init_model, process


def get_controlnet_depth():
    print("=> initializing ControlNet Depth...")
    model, ddim_sampler = init_model()

    return model, ddim_sampler


def get_inpainting(device):
    print("=> initializing Inpainting...")

    model = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to(device)

    return model

def get_image2materials(model_id, device):
    print("=> initializing material estimation...")

    model = MVDiffusionAlbedoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

    model.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return model

def get_uvrefiner(model_id, device):
    print("=> initializing UV refinement...")

    model = UVDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

    # model.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return model

def get_text2image(device):
    print("=> initializing Inpainting...")

    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to(device)

    return model


@torch.no_grad()
def apply_controlnet_depth(model, ddim_sampler, 
    init_image, prompt, strength, ddim_steps,
    generate_mask_image, keep_mask_image, depth_map_np, 
    a_prompt, n_prompt, guidance_scale, seed, eta, num_samples,
    device, blend=0, save_memory=False):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_np: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating ControlNet Depth RePaint image...")


    # Stable Diffusion 2 receives PIL.Image
    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image_np = process(
        model, ddim_sampler,
        np.array(init_image), prompt, a_prompt, n_prompt, num_samples,
        ddim_steps, guidance_scale, seed, eta, 
        strength=strength, detected_map=depth_map_np, unknown_mask=np.array(generate_mask_image), save_memory=save_memory
    )[0]

    init_image = init_image.convert("RGB")
    diffused_image = Image.fromarray(diffused_image_np).convert("RGB")

    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(init_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = np.array(keep_mask_image).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)

        overlap_mask_np = np.array(generate_mask_image).astype(np.uint8)
        overlap_mask_np *= keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]

        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)

        generate_image_np[overlap_mask_np == 1] = overlap_np

        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

    init_image_masked = init_image
    diffused_image_masked = diffused_image

    return diffused_image, init_image_masked, diffused_image_masked


@torch.no_grad()
def apply_inpainting(model, 
    init_image, mask_image_tensor, prompt, height, width, device):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating Inpainting image...")

    mask_image = mask_image_tensor[0].cpu()
    mask_image = mask_image.permute(2, 0, 1)
    mask_image = transforms.ToPILImage()(mask_image).convert("L")

    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image = model(
        prompt=prompt, 
        image=init_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)), 
        height=512, 
        width=512
    ).images[0].resize((height, width))

    return diffused_image


@torch.no_grad()
def apply_inpainting_postprocess(model, 
    init_image, mask_image_tensor, prompt, height, width, device):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating Inpainting image...")

    mask_image = mask_image_tensor[0].cpu()
    mask_image = mask_image.permute(2, 0, 1)
    mask_image = transforms.ToPILImage()(mask_image).convert("L")

    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image = model(
        prompt=prompt, 
        image=init_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)), 
        height=512, 
        width=512
    ).images[0].resize((height, width))

    diffused_image_tensor = torch.from_numpy(np.array(diffused_image)).to(device)

    init_images_tensor = torch.from_numpy(np.array(init_image)).to(device)
    
    init_images_tensor = diffused_image_tensor * mask_image_tensor[0] + init_images_tensor * (1 - mask_image_tensor[0])
    init_image = Image.fromarray(init_images_tensor.cpu().numpy().astype(np.uint8)).convert("RGB")

    return init_image


@torch.no_grad()
def apply_material_estimation(model, prompt, input_image, input_normal, init_materials, masks, height, width, device):

    print("=> generating material images...")

    generator = torch.Generator("cuda").manual_seed(0)

    output_images_pil = model(
        prompt=[''],
        cond_image=[input_image],
        normal_image=[input_normal], 
        init_materials=init_materials, 
        masks=masks,
        num_inference_steps=50, 
        guidance_scale=1.0,
        # image_guidance_scale=-1,
        generator=generator,
        height=height,
        width=width,
    ).images

    albedo_image, rm_image, normal_image = output_images_pil

    return albedo_image.resize((height, width)), rm_image.resize((height, width)), normal_image.resize((height, width))

@torch.no_grad()
def apply_uv_refinement(model, albedo, rm, bump, ccm, mask, height, width, device):

    print("=> generating refined materials...")

    generator = torch.Generator("cuda").manual_seed(0)

    output_images_pil = model(
        prompt='',
        ccm_image=[ccm],
        albedo_image=[albedo],
        rm_image=[rm],
        bump_image=[bump],
        # mask_image=[mask],
        num_inference_steps=50, 
        guidance_scale=1.0,
        # image_guidance_scale=-1,
        generator=generator,
        height=height,
        width=width,
    ).images

    albedo_image, rm_image, bump_image = output_images_pil

    return albedo_image, rm_image, bump_image

