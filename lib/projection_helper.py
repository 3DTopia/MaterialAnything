import os
import torch

import cv2
import random

import numpy as np

from torchvision import transforms

from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import interpolate_face_attributes

from PIL import Image

from tqdm import tqdm

# customized
import sys
sys.path.append(".")
from lib.voronoi import voronoi_solve

from lib.camera_helper import init_camera
from lib.render_helper import init_renderer, render
from lib.shading_helper import (
    BlendParams,
    init_soft_phong_shader, 
    init_flat_texel_shader,
)
from lib.vis_helper import visualize_outputs, visualize_quad_mask
from lib.constants import *


def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()


def compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device):
    """
        compose quad mask:
            -> 0: background
            -> 1: old
            -> 2: update
            -> 3: new
    """

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    update_mask_tensor = transforms.ToTensor()(update_mask_image).to(device)
    old_mask_tensor = transforms.ToTensor()(old_mask_image).to(device)

    all_mask_tensor = new_mask_tensor + update_mask_tensor + old_mask_tensor
    all_mask_tensor[all_mask_tensor>0] = 1

    quad_mask_tensor = torch.zeros_like(all_mask_tensor)
    quad_mask_tensor[old_mask_tensor == 1] = 1
    quad_mask_tensor[new_mask_tensor == 1] = 3
    quad_mask_tensor[update_mask_tensor == 1] = 2 # update_mask_tensor is dilated

    return old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor


def compute_view_heat(similarity_tensor, quad_mask_tensor):
    num_total_pixels = quad_mask_tensor.reshape(-1).shape[0]
    heat = 0
    for idx in QUAD_WEIGHTS:
        heat += (quad_mask_tensor == idx).sum() * QUAD_WEIGHTS[idx] / num_total_pixels

    return heat


def select_viewpoint(selected_view_ids, view_punishments,
    mode, dist_list, elev_list, azim_list, sector_list, view_idx,
    similarity_texture_cache, exist_texture,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    device, use_principle=False
):
    if mode == "sequential":
        
        num_views = len(dist_list)

        dist = dist_list[view_idx % num_views]
        elev = elev_list[view_idx % num_views]
        azim = azim_list[view_idx % num_views]
        sector = sector_list[view_idx % num_views]
        
        selected_view_ids.append(view_idx % num_views)

    elif mode == "heuristic":

        if use_principle and view_idx < 6:

            selected_view_idx = view_idx

        else:

            selected_view_idx = None
            max_heat = 0

            print("=> selecting next view...")
            view_heat_list = []
            for sample_idx in tqdm(range(len(dist_list))):

                view_heat, *_ = render_one_view_and_build_masks(dist_list[sample_idx], elev_list[sample_idx], azim_list[sample_idx], 
                    sample_idx, sample_idx, view_punishments,
                    similarity_texture_cache, exist_texture,
                    mesh, faces, verts_uvs,
                    image_size, faces_per_pixel,
                    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                    device)

                if view_heat > max_heat:
                    selected_view_idx = sample_idx
                    max_heat = view_heat

                view_heat_list.append(view_heat.item())

            print(view_heat_list)
            print("select view {} with heat {}".format(selected_view_idx, max_heat))

 
        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

        view_punishments[selected_view_idx] *= 0.01

    elif mode == "random":

        selected_view_idx = random.choice(range(len(dist_list)))

        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]
        
        selected_view_ids.append(selected_view_idx)

    else:
        raise NotImplementedError()

    return dist, elev, azim, sector, selected_view_ids, view_punishments


@torch.no_grad()
def build_backproject_mask(mesh, faces, verts_uvs, 
    cameras, reference_image, faces_per_pixel, 
    image_size, uv_size, device):
    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(-1, 2)

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs[:, 1]).reshape(-1) * (uv_size - 1),
        pixel_uvs[:, 0].reshape(-1) * (uv_size - 1)
    )

    K = faces_per_pixel

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size)))).float() / 255.
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])

    # texture
    texture_tensor = torch.zeros(uv_size, uv_size, 3).to(device)
    texture_tensor[texture_locations_y, texture_locations_x, :] = texture_values.reshape(-1, 3)

    return texture_tensor[:, :, 0]

@torch.no_grad()
def erode_exist_mask(mask_tensor, valid_mask_tensor, device, iterations=5):
    not_exist_mask= (1 - mask_tensor) * valid_mask_tensor
    not_exist_mask = (not_exist_mask.cpu().numpy()*255).astype(np.uint8) 
    # cv2.imwrite("exist_texture_before.png", (exist_mask_tensor[0].cpu().numpy()*255).astype(np.uint8))
    not_exist_mask = cv2.dilate(not_exist_mask[0], kernel=np.ones((3, 3), np.uint8), iterations=iterations)
    not_exist_mask = torch.from_numpy(not_exist_mask / 255.).unsqueeze(0).to(device)
    mask_tensor = (1 - not_exist_mask) * valid_mask_tensor
    # cv2.imwrite("exist_texture_after.png", (exist_mask_tensor[0].cpu().numpy()*255).astype(np.uint8))
    return mask_tensor

def dilate_update_mask(mask_tensor, valid_mask_tensor, device, iterations=5):
    mask_tensor = (mask_tensor.cpu().numpy()*255).astype(np.uint8) 
    # cv2.imwrite("exist_texture_before.png", (exist_mask_tensor[0].cpu().numpy()*255).astype(np.uint8))
    mask_tensor = cv2.dilate(mask_tensor[0], kernel=np.ones((3, 3), np.uint8), iterations=iterations)
    mask_tensor = torch.from_numpy(mask_tensor / 255.).unsqueeze(0).to(device)
    mask_tensor = mask_tensor * valid_mask_tensor
    # cv2.imwrite("exist_texture_after.png", (exist_mask_tensor[0].cpu().numpy()*255).astype(np.uint8))
    return mask_tensor

from scipy.ndimage import label, binary_opening
from skimage.morphology import remove_small_objects
def remove_isolated_regions(mask_tensor, device):
    mask_np = mask_tensor.squeeze(0).cpu().numpy()
    single_channel_mask = mask_np[..., 0]

    labeled_mask, num_features = label(single_channel_mask.astype(bool))

    if num_features > 1:
        size_threshold = 50
        cleaned_single_channel = remove_small_objects(labeled_mask, min_size=size_threshold)
    else:
        cleaned_single_channel = single_channel_mask

    cleaned_mask_np = np.repeat(cleaned_single_channel[:, :, np.newaxis], 3, axis=2)

    cleaned_mask_tensor = torch.from_numpy(cleaned_mask_np).unsqueeze(0).to(device, torch.float32)
    return cleaned_mask_tensor

def remove_isolated_regions_cv(mask_tensor, visible_mask, device):
    mask_tensor = (mask_tensor.cpu().numpy()*255).astype(np.uint8) 
    mask_tensor = cv2.dilate(mask_tensor[0], kernel=np.ones((3, 3), np.uint8), iterations=3)
    mask_tensor = cv2.erode(mask_tensor, kernel=np.ones((3, 3), np.uint8), iterations=3)
    mask_tensor = torch.from_numpy(mask_tensor / 255.).unsqueeze(0).to(device)
    mask_tensor = mask_tensor * visible_mask
    return mask_tensor


@torch.no_grad()
def build_diffusion_mask(mesh_stuff, 
    renderer, exist_texture, similarity_texture_cache, target_value, device, image_size, 
    smooth_mask=False, view_threshold=0.01):

    mesh, faces, verts_uvs = mesh_stuff
    mask_mesh = mesh.clone() # NOTE in-place operation - DANGER!!!

    # visible mask => the whole region
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=torch.ones_like(exist_texture_expand),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    # visible_mask_tensor, *_ = render(mask_mesh, renderer)
    visible_mask_tensor, _, similarity_map_tensor, *_ = render(mask_mesh, renderer)
    visible_mask_tensor_src = visible_mask_tensor.clone()
    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= view_threshold).float()
    visible_mask_tensor *= valid_mask_tensor

    # nonexist mask <=> new mask
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=1 - exist_texture_expand,
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    new_mask_tensor, *_ = render(mask_mesh, renderer)
    new_mask_tensor_src = new_mask_tensor.clone()
    new_mask_tensor *= valid_mask_tensor

    # exist mask => visible mask - new mask
    exist_mask_tensor = visible_mask_tensor - new_mask_tensor
    exist_mask_tensor[exist_mask_tensor < 0] = 0 # NOTE dilate can lead to overflow
    if (exist_mask_tensor[...,0].sum() / valid_mask_tensor.sum()) < 0.1:
        update_mask_tensor = torch.zeros_like(exist_mask_tensor)
    else:
        exist_mask_tensor_src = visible_mask_tensor_src - new_mask_tensor_src
        exist_mask_tensor_src[exist_mask_tensor_src < 0] = 0 # NOTE dilate can lead to overflow

        exist_mask_tensor_src = remove_isolated_regions(exist_mask_tensor_src, device)
        exist_mask_tensor_src = remove_isolated_regions_cv(exist_mask_tensor_src, visible_mask_tensor_src, device)
        new_mask_tensor_src = visible_mask_tensor_src - exist_mask_tensor_src

        exist_mask_tensor_src = erode_exist_mask(exist_mask_tensor_src, visible_mask_tensor_src, device)
        new_mask_tensor_src = erode_exist_mask(new_mask_tensor_src, visible_mask_tensor_src, device)
    
        update_mask_tensor = visible_mask_tensor_src - exist_mask_tensor_src - new_mask_tensor_src
        update_mask_tensor = dilate_update_mask(update_mask_tensor, visible_mask_tensor_src, device, iterations=2)
        update_mask_tensor = visible_mask_tensor * update_mask_tensor
        update_mask_tensor = torch.clamp(update_mask_tensor, 0, 1)

    # all update mask
    mask_mesh.textures = TexturesUV(
        maps=(
            similarity_texture_cache.argmax(0) == target_value
            # # only consider the views that have already appeared before
            # similarity_texture_cache[0:target_value+1].argmax(0) == target_value
        ).float().unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    all_update_mask_tensor, *_ = render(mask_mesh, renderer)

    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor + update_mask_tensor
    update_mask_tensor = (update_mask_tensor > 0).float()

    # keep mask => exist mask - update mask
    old_mask_tensor = (exist_mask_tensor - update_mask_tensor) * exist_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert("L")

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, update_mask, old_mask, exist_mask

@torch.no_grad()
def build_diffusion_materials(mesh_stuff, renderer, init_albedo, init_roughness_metallic, init_bump, device):

    mesh, faces, verts_uvs = mesh_stuff
    material_mesh = mesh.clone() # NOTE in-place operation - DANGER!!!

    material_mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_albedo)[None, ...].permute(0, 2, 3, 1).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...]
    )
    init_albedo_view, *_ = render(material_mesh, renderer)

    material_mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_roughness_metallic)[None, ...].permute(0, 2, 3, 1).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...]
    )
    init_roughness_metallic_view, *_ = render(material_mesh, renderer)

    material_mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_bump)[None, ...].permute(0, 2, 3, 1).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...]
    )
    init_bump_view, *_ = render(material_mesh, renderer)

    # convert
    init_albedo_view_pil = init_albedo_view[0].cpu().float().permute(2, 0, 1)
    init_albedo_view_pil = transforms.ToPILImage()(init_albedo_view_pil).convert("RGB")

    init_roughness_metallic_view_pil = init_roughness_metallic_view[0].cpu().float().permute(2, 0, 1)
    init_roughness_metallic_view_pil = transforms.ToPILImage()(init_roughness_metallic_view_pil).convert("RGB")

    init_bump_view_pil = init_bump_view[0].cpu().float().permute(2, 0, 1)
    init_bump_view_pil = transforms.ToPILImage()(init_bump_view_pil).convert("RGB")

    init_materials = {
        "albedo_pil": init_albedo_view_pil,
        "roughness_metallic_pil": init_roughness_metallic_view_pil,
        "bump_pil": init_bump_view_pil,
        "albedo": init_albedo_view,
        "roughness_metallic": init_roughness_metallic_view,
        "bump": init_bump_view
    }

    return init_materials


@torch.no_grad()
def render_one_view(mesh,
    dist, elev, azim,
    image_size, faces_per_pixel,
    device):

    # render the view
    cameras = init_camera(
        dist, elev, azim,
        image_size, device
    )
    renderer = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )

    init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments = render(mesh, renderer)
    
    return (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    )


@torch.no_grad()
def build_similarity_texture_cache_for_all_views(mesh, faces, verts_uvs,
    dist_list, elev_list, azim_list,
    image_size, image_size_scaled, uv_size, faces_per_pixel,
    device):

    num_candidate_views = len(dist_list)
    similarity_texture_cache = torch.zeros(num_candidate_views, uv_size, uv_size).to(device)
    # similarity_texture_cache_masked = torch.zeros(num_candidate_views, uv_size, uv_size).to(device)

    print("=> building similarity texture cache for all views...")
    for i in tqdm(range(num_candidate_views)):
        cameras, _, visible_tensor, _, similarity_tensor, _, _ = render_one_view(mesh,
            dist_list[i], elev_list[i], azim_list[i],
            image_size, faces_per_pixel, device)

        similarity_texture_cache[i] = build_backproject_mask(mesh, faces, verts_uvs, 
            cameras, transforms.ToPILImage()(similarity_tensor[0, :, :, 0]).convert("RGB"), faces_per_pixel,
            image_size_scaled, uv_size, device)
        
        # similarity_tensor = similarity_tensor * (similarity_tensor > 0.15).float() * visible_tensor
        # similarity_texture_cache_masked[i] = build_backproject_mask(mesh, faces, verts_uvs, 
        #     cameras, transforms.ToPILImage()(similarity_tensor[0, :, :, 0]).convert("RGB"), faces_per_pixel,
        #     image_size_scaled, uv_size, device)

    return similarity_texture_cache


@torch.no_grad()
def render_one_view_and_build_masks(dist, elev, azim, 
    selected_view_idx, view_idx, view_punishments,
    similarity_texture_cache, exist_texture,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    device, save_intermediate=False, smooth_mask=False, view_threshold=0.01):
    
    # render the view
    (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    ) = render_one_view(mesh,
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )
    
    init_image = init_images_tensor[0].cpu()
    init_image = init_image.permute(2, 0, 1)
    init_image = transforms.ToPILImage()(init_image).convert("RGB")

    normal_map = normal_maps_tensor[0].cpu()
    normal_map = normal_map.permute(2, 0, 1)
    # normal_map = (normal_map + 1.) / 2.


    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")


    flat_renderer = init_renderer(cameras,
        shader=init_flat_texel_shader(
            camera=cameras,
            device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel
    )
    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask(
        (mesh, faces, verts_uvs), 
        flat_renderer, exist_texture, similarity_texture_cache, selected_view_idx, device, image_size, 
        smooth_mask=smooth_mask, view_threshold=view_threshold
    )
    # NOTE the view idx is the absolute idx in the sample space (i.e. `selected_view_idx`)
    # it should match with `similarity_texture_cache`

    (
        old_mask_tensor, 
        update_mask_tensor, 
        new_mask_tensor, 
        all_mask_tensor, 
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    LUB2RUF = torch.tensor([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ], dtype=normal_map.dtype)
    # covert normal map to camera space
    normal_map = normal_map.permute(1, 2, 0).view(-1, 3)
    normal_map = (cameras.R[0].cpu().T @ normal_map.T).T @ LUB2RUF
    normal_map = normal_map.reshape(image_size, image_size, 3).permute(2, 0, 1)
    normal_map = (normal_map + 1.) / 2.

    normal_map = normal_map*all_mask_tensor[0].cpu() + (1 - all_mask_tensor[0].cpu())
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_update.png".format(view_idx)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_old.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        renderer, cameras, fragments,
        init_image, normal_map, depth_map, 
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
        old_mask_image, update_mask_image, new_mask_image, 
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor
    )

@torch.no_grad()
def render_one_view_and_build_masks_materials(dist, elev, azim, 
    selected_view_idx, view_idx, view_punishments,
    similarity_texture_cache, exist_texture, init_albedo, init_roughness_metallic, init_bump,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    device, save_intermediate=False, smooth_mask=False, view_threshold=0.01):
    
    # render the view
    (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    ) = render_one_view(mesh,
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )
    
    init_image = init_images_tensor[0].cpu()
    init_image = init_image.permute(2, 0, 1)
    init_image = transforms.ToPILImage()(init_image).convert("RGB")

    normal_map = normal_maps_tensor[0].cpu()
    normal_map = normal_map.permute(2, 0, 1)
    valid_mask_tensor = (abs(normal_maps_tensor[0]).sum(-1) > 0).float()
    # normal_map = (normal_map + 1.) / 2.


    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")


    flat_renderer = init_renderer(cameras,
        shader=init_flat_texel_shader(
            camera=cameras,
            device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel
    )
    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask(
        (mesh, faces, verts_uvs), 
        flat_renderer, exist_texture, similarity_texture_cache, selected_view_idx, device, image_size, 
        smooth_mask=smooth_mask, view_threshold=view_threshold
    )

    init_materials = build_diffusion_materials(
        (mesh, faces, verts_uvs), 
        renderer, 
        init_albedo, 
        init_roughness_metallic, 
        init_bump, 
        device
    )

    # NOTE the view idx is the absolute idx in the sample space (i.e. `selected_view_idx`)
    # it should match with `similarity_texture_cache`

    (
        old_mask_tensor, 
        update_mask_tensor, 
        new_mask_tensor, 
        all_mask_tensor, 
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    LUB2RUF = torch.tensor([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ], dtype=normal_map.dtype)
    # covert normal map to camera space
    normal_map = normal_map.permute(1, 2, 0).view(-1, 3)
    normal_map = (cameras.R[0].cpu().T @ normal_map.T).T @ LUB2RUF
    normal_map = normal_map.reshape(image_size, image_size, 3).permute(2, 0, 1)
    normal_map = (normal_map + 1.) / 2.

    normal_map = normal_map*valid_mask_tensor.cpu() + (1 - valid_mask_tensor.cpu())
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        init_materials['albedo_pil'].save(os.path.join(init_image_dir, "{}_{}.png".format(view_idx, "albedo")))
        init_materials['roughness_metallic_pil'].save(os.path.join(init_image_dir, "{}_{}.png".format(view_idx, "roughness_metallic")))
        init_materials['bump_pil'].save(os.path.join(init_image_dir, "{}_{}.png".format(view_idx, "bump")))

        normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_update.png".format(view_idx)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_old.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

        background = Image.new("L", (image_size, image_size), (255))
        bg_mask = (all_mask_tensor != 0)
        bg_mask_image = Image.fromarray(bg_mask[0].cpu().numpy().astype(np.uint8)*255)
        exist_mask_image_white_bg = Image.composite(exist_mask_image, background, bg_mask_image)
        exist_mask_image_white_bg.save(os.path.join(mask_image_dir, "{}_exist_white_bg.png".format(view_idx)))
        

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        renderer, cameras, fragments,
        init_image, init_materials,  normal_map, depth_map, 
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
        old_mask_image, update_mask_image, new_mask_image, 
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor, valid_mask_tensor
    )



@torch.no_grad()
def backproject_from_image_src(mesh, faces, verts_uvs, cameras, 
    reference_image, new_mask_image, update_mask_image, 
    init_texture, exist_texture,
    image_size, uv_size, faces_per_pixel,
    device):

    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(pixel_uvs.shape[-2], pixel_uvs.shape[1], pixel_uvs.shape[2], 2)

    # the update mask has to be on top of the diffusion mask
    new_mask_image_tensor = transforms.ToTensor()(new_mask_image).to(device).unsqueeze(-1)
    update_mask_image_tensor = transforms.ToTensor()(update_mask_image).to(device).unsqueeze(-1)
    
    project_mask_image_tensor = torch.logical_or(update_mask_image_tensor, new_mask_image_tensor).float()
    project_mask_image = project_mask_image_tensor * 255.
    project_mask_image = Image.fromarray(project_mask_image[0, :, :, 0].cpu().numpy().astype(np.uint8))
    
    project_mask_image_scaled = project_mask_image.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST
    )
    project_mask_image_tensor_scaled = transforms.ToTensor()(project_mask_image_scaled).to(device)

    pixel_uvs_masked = pixel_uvs[project_mask_image_tensor_scaled == 1]

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs_masked[:, 1]).reshape(-1) * (uv_size - 1), 
        pixel_uvs_masked[:, 0].reshape(-1) * (uv_size - 1)
    )
    
    K = pixel_uvs.shape[0]
    project_mask_image_tensor_scaled = project_mask_image_tensor_scaled[:, None, :, :, None].repeat(1, 4, 1, 1, 3)

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size))))
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])
    
    texture_values_masked = texture_values.reshape(-1, 3)[project_mask_image_tensor_scaled.reshape(-1, 3) == 1].reshape(-1, 3)

    # texture
    texture_tensor = torch.from_numpy(np.array(init_texture)).to(device)
    texture_tensor[texture_locations_y, texture_locations_x, :] = texture_values_masked
    
    init_texture = Image.fromarray(texture_tensor.cpu().numpy().astype(np.uint8))

    # update texture cache
    exist_texture[texture_locations_y, texture_locations_x] = 1

    return init_texture, project_mask_image, exist_texture

import torchvision.transforms as transforms

def bake_texture(views, mesh, pre_dist_list, pre_elev_list, pre_azim_list, weights_masked, image_size = None, uv_size=None, exp=None, noisy=False, generator=None, return_mask=False, device='cuda'):
    if not exp:
        exp=1
    to_tensor = transforms.ToTensor()
    views = [to_tensor(view).permute(1, 2, 0).to(device=device) for view in views] # CxHxW -> HxWxC


    tmp_mesh = mesh.clone()
    gradient_maps = [torch.zeros((uv_size, uv_size) + (views[0].shape[2],), device=device, requires_grad=True) for view in views]
    optimizer = torch.optim.SGD(gradient_maps, lr=1, momentum=0)
    optimizer.zero_grad()
    loss = 0
    for i in range(len(pre_dist_list)):
        
        cameras = init_camera(
            pre_dist_list[i], pre_elev_list[i], pre_azim_list[i],
            image_size, device
        )
        renderer = init_renderer(cameras,
            shader=init_soft_phong_shader(
                camera=cameras,
                blend_params=BlendParams(),
                device=device),
            image_size=image_size, 
            faces_per_pixel=1
        )

        zero_map = gradient_maps[i]

        zero_tex = TexturesUV([zero_map], mesh.textures.faces_uvs_padded(), mesh.textures.verts_uvs_padded())
        tmp_mesh.textures = zero_tex
        images_predicted,_ = renderer(tmp_mesh)
        loss += torch.sum((1 - images_predicted)**2)
    loss.backward()
    optimizer.step()


    # render the view
    tmp_mesh = mesh.clone()
    bake_maps = [torch.zeros((uv_size, uv_size) + (views[0].shape[2],), device=device, requires_grad=True) for view in views]
    optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
    optimizer.zero_grad()
    loss = 0
    for i in range(len(views)):    

        cameras = init_camera(
            pre_dist_list[i], pre_elev_list[i], pre_azim_list[i],
            image_size, device
        )
        renderer = init_renderer(cameras,
            shader=init_soft_phong_shader(
                camera=cameras,
                blend_params=BlendParams(),
                device=device),
            image_size=image_size, 
            faces_per_pixel=1
        )

        bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded())
        tmp_mesh.textures = bake_tex
        images_predicted,_ = renderer(tmp_mesh)
        predicted_rgb = images_predicted[..., :-1]
        loss += (((predicted_rgb[0] - views[i]))**2).sum()
    loss.backward(retain_graph=False)
    optimizer.step()

    total_weights = 0
    baked = 0
    # mask = torch.zeros_like(self.visible_triangles[0], dtype=torch.bool)
    for i in range(len(bake_maps)):
        normalized_baked_map = bake_maps[i].detach() / (gradient_maps[i].detach() + 1E-16)
        bake_map = voronoi_solve(normalized_baked_map, gradient_maps[i][...,0])
        weight = weights_masked[i][:,:,None] ** exp 
        total_weights += weight * ((bake_map).sum(-1) > 0)[..., None]
        baked += bake_map * weight
        # bake_map = bake_map * (weight > 0)
        # bake_map = bake_map.cpu().numpy()
        # bake_map = (bake_map * 255).astype(np.uint8)
        # bake_map = Image.fromarray(bake_map)
        # bake_map.save('bake_map_debug.png')
        # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    baked /= (total_weights + 1E-16)
    baked = voronoi_solve(baked, total_weights[...,0])
    baked_valid = (total_weights > 0 ).float()
    baked = baked * baked_valid + (1 - baked_valid)
    # import pdb; pdb.set_trace()
    # bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded())
    # tmp_mesh.textures = bake_tex
    # extended_mesh = tmp_mesh.extend(len(pre_dist_list))
    # images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
    # learned_views = [image.permute(2, 0, 1) for image in images_predicted]
    # if return_mask:
    #     return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1), mask

    # save baked
    baked = baked.cpu().numpy()
    baked = (baked * 255).astype(np.uint8)
    baked = Image.fromarray(baked)
    # baked.save('uv_debug.png')

    return baked


@torch.no_grad()
def backproject_from_image(mesh, faces, verts_uvs, cameras, 
    reference_image, new_mask_image, update_mask_image, 
    init_texture, exist_texture,
    image_size, uv_size, faces_per_pixel,
    device, blending_weights=0.0):

    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(pixel_uvs.shape[-2], pixel_uvs.shape[1], pixel_uvs.shape[2], 2)

    # the update mask has to be on top of the diffusion mask
    new_mask_image_tensor = transforms.ToTensor()(new_mask_image).to(device).unsqueeze(-1)
    update_mask_image_tensor = transforms.ToTensor()(update_mask_image).to(device).unsqueeze(-1)
    
    project_mask_image_tensor = torch.logical_or(update_mask_image_tensor, new_mask_image_tensor).float()
    project_mask_image = project_mask_image_tensor * 255.
    project_mask_image = Image.fromarray(project_mask_image[0, :, :, 0].cpu().numpy().astype(np.uint8))
    
    project_mask_image_scaled = project_mask_image.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST
    )
    project_mask_image_tensor_scaled = transforms.ToTensor()(project_mask_image_scaled).to(device)

    pixel_uvs_masked = pixel_uvs[project_mask_image_tensor_scaled == 1]

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs_masked[:, 1]).reshape(-1) * (uv_size - 1), 
        pixel_uvs_masked[:, 0].reshape(-1) * (uv_size - 1)
    )

    new_exist_texture = torch.zeros_like(exist_texture)
    new_exist_texture[texture_locations_y, texture_locations_x] = 1
    overlap_regison = exist_texture * new_exist_texture

    
    K = pixel_uvs.shape[0]
    project_mask_image_tensor_scaled = project_mask_image_tensor_scaled[:, None, :, :, None].repeat(1, 4, 1, 1, 3)

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size))))
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])
    
    texture_values_masked = texture_values.reshape(-1, 3)[project_mask_image_tensor_scaled.reshape(-1, 3) == 1].reshape(-1, 3)

    # texture
    blending_weights = torch.ones_like(exist_texture) - new_exist_texture + blending_weights * overlap_regison
    blending_weights = blending_weights.unsqueeze(-1)
    texture_tensor = torch.from_numpy(np.array(init_texture)).to(device)
    new_texture_tensor = torch.ones_like(texture_tensor)*255
    new_texture_tensor[texture_locations_y, texture_locations_x, :] = texture_values_masked

    texture_tensor = texture_tensor * blending_weights + (1-blending_weights) * new_texture_tensor
    
    init_texture = Image.fromarray(texture_tensor.cpu().numpy().astype(np.uint8))

    # update texture cache
    exist_texture[texture_locations_y, texture_locations_x] = 1

    return init_texture, project_mask_image, exist_texture

@torch.no_grad()
def backproject_from_image_weights(mesh, faces, verts_uvs, cameras, 
    reference_image, new_mask_image, update_mask_image, 
    init_texture, exist_texture,
    image_size, uv_size, faces_per_pixel, init_weight, new_weight,
    device):

    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(pixel_uvs.shape[-2], pixel_uvs.shape[1], pixel_uvs.shape[2], 2)

    # the update mask has to be on top of the diffusion mask
    new_mask_image_tensor = transforms.ToTensor()(new_mask_image).to(device).unsqueeze(-1)
    update_mask_image_tensor = transforms.ToTensor()(update_mask_image).to(device).unsqueeze(-1)
    
    project_mask_image_tensor = torch.logical_or(update_mask_image_tensor, new_mask_image_tensor).float()
    project_mask_image = project_mask_image_tensor * 255.
    project_mask_image = Image.fromarray(project_mask_image[0, :, :, 0].cpu().numpy().astype(np.uint8))
    
    project_mask_image_scaled = project_mask_image.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST
    )
    project_mask_image_tensor_scaled = transforms.ToTensor()(project_mask_image_scaled).to(device)

    pixel_uvs_masked = pixel_uvs[project_mask_image_tensor_scaled == 1]

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs_masked[:, 1]).reshape(-1) * (uv_size - 1), 
        pixel_uvs_masked[:, 0].reshape(-1) * (uv_size - 1)
    )
    
    K = pixel_uvs.shape[0]
    project_mask_image_tensor_scaled = project_mask_image_tensor_scaled[:, None, :, :, None].repeat(1, 4, 1, 1, 3)

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size))))
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])
    
    texture_values_masked = texture_values.reshape(-1, 3)[project_mask_image_tensor_scaled.reshape(-1, 3) == 1].reshape(-1, 3)

    # texture
    texture_tensor = torch.from_numpy(np.array(init_texture)).to(device)

    texture_tensor_new = torch.ones_like(texture_tensor) * 255
    texture_tensor_new[texture_locations_y, texture_locations_x, :] = texture_values_masked
    Image.fromarray(texture_tensor_new.cpu().numpy().astype(np.uint8)).save("texture_tensor.png")
    init_weight[exist_texture<1] = 0
    new_weight_masked = torch.zeros_like(new_weight)
    new_weight_masked[texture_locations_y, texture_locations_x] = new_weight[texture_locations_y, texture_locations_x]
    texture_tensor = (texture_tensor * init_weight.unsqueeze(-1) + texture_tensor_new * new_weight_masked.unsqueeze(-1)) / (init_weight.unsqueeze(-1) + new_weight_masked.unsqueeze(-1) + 1e-6)

    # update texture cache
    exist_texture[texture_locations_y, texture_locations_x] = 1
    texture_tensor[exist_texture<1] = 255

    init_texture = Image.fromarray(texture_tensor.cpu().numpy().astype(np.uint8))

    return init_texture, project_mask_image, exist_texture

