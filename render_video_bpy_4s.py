"""
Blender script to render images of 3D models.

Modified from https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --num_images 32 \

"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple, Literal
import numpy as np
import bpy
from mathutils import Vector, Matrix
import logging
import threading
import json
import fcntl

# file_lock = threading.Lock()

logger = logging.getLogger("bpy")

logger.setLevel(logging.WARNING)


class BlenderRendering():
    def __init__(self, args) -> None:
        
        self.args = args
        self.object_uid = args.obj_uid

        context = bpy.context
        self.scene = context.scene
        self.render = self.scene.render
        self.scene.use_nodes = True
        self.tree = self.scene.node_tree
        self.links = self.tree.links

        self.render.engine = args.engine
        self.render.image_settings.file_format = "PNG"
        self.render.image_settings.color_mode = "RGBA"
        self.render.resolution_x = args.resolution
        self.render.resolution_y = args.resolution
        self.render.resolution_percentage = 100

        self.scene.cycles.device = "GPU"
        self.scene.cycles.samples = 128
        self.scene.cycles.diffuse_bounces = 1
        self.scene.cycles.glossy_bounces = 1
        self.scene.cycles.transparent_max_bounces = 3
        self.scene.cycles.transmission_bounces = 3
        self.scene.cycles.filter_width = 0.01
        self.scene.cycles.use_denoising = True
        self.scene.render.film_transparent = True
        self.scene.render.dither_intensity = 0.0

        self.scene.render.use_persistent_data = True # use persistent data for speed up
        context.view_layer.use_pass_normal = True # for normal rendering
        context.view_layer.use_pass_z = True # for depth rendering
        context.view_layer.use_pass_position = True
        context.view_layer.pass_alpha_threshold = 0.5

        # Set the device_type
        cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
        cycles_preferences.compute_device_type = "CUDA"  # or "OPENCL"
        cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
        for device in cuda_devices:
            device.use = True

    def compose_RT(self, R, T):
        return np.hstack((R, T.reshape(-1, 1)))

    def sample_point_on_sphere(self, radius: float) -> Tuple[float, float, float]:
        theta = random.random() * 2 * math.pi
        phi = math.acos(2 * random.random() - 1)
        return (
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi),
        )


    def sample_spherical(self, radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
        correct = False
        while not correct:
            vec = np.random.uniform(-1, 1, 3)
    #         vec[2] = np.abs(vec[2])
            radius = np.random.uniform(radius_min, radius_max, 1)
            vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
            if maxz > vec[2] > minz:
                correct = True
        return vec

    def set_camera_location(self, camera, i, option: str):
        assert option in ['fixed', 'random', 'front']

        if option == 'fixed':
            locations = [
                [ 0, -1,  0],
                [-1, -1,  0],
                [-1,  0,  0],
                [-1,  1,  0],
                [ 0,  1,  0],
                [ 1,  1,  0],
                [ 1,  0,  0],
                [ 1, -1,  0]
            ]
            vec = locations[i]
            radius = 2
            vec = vec / np.linalg.norm(vec, axis=0) * radius
            x, y, z = vec
        elif option == 'random':
            # from https://blender.stackexchange.com/questions/18530/
            x, y, z = self.sample_spherical(radius_min=1.9, radius_max=2.6, maxz=1.60, minz=-0.75)
        elif option == 'front':
            x, y, z = 0, -np.random.uniform(1.9, 2.6, 1)[0], 0

        camera.location = x, y, z

        # adjust orientation
        direction = - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        return camera
    

    def _create_light(
        self,
        name: str,
        light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        energy: float,
        use_shadow: bool = False,
        use_contact_shadow: bool = False,
        specular_factor: float = 1.0,
    ):
        """Creates a light object.

        Args:
            name (str): Name of the light object.
            light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
            location (Tuple[float, float, float]): Location of the light.
            rotation (Tuple[float, float, float]): Rotation of the light.
            energy (float): Energy of the light.
            use_shadow (bool, optional): Whether to use shadows. Defaults to False.
            specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

        Returns:
            bpy.types.Object: The light object.
        """

        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_object = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = location
        light_object.rotation_euler = rotation
        light_data.use_shadow = use_shadow
        light_data.use_contact_shadow = use_contact_shadow
        light_data.specular_factor = specular_factor
        light_data.energy = energy
        if light_type=="SUN":
            light_data.angle=0.5
        return light_object


    def randomize_lighting(self):
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """
        # Add random angle offset in 0-90
        angle_offset = random.uniform(0, math.pi / 2)
        

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        # Create key light
        key_light = self._create_light(
            name="Key_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, -0.785398 + angle_offset), # 45 0 -45
            energy=random.choice([2.5, 3.25, 4]),
            use_shadow=True,
        )

        # Create rim light
        rim_light = self._create_light(
            name="Rim_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(-0.785398, 0, -3.92699 + angle_offset), # -45 0 -225
            energy=random.choice([2.5, 3.25, 4]),
            use_shadow=True,
        )

        # Create fill light
        fill_light = self._create_light(
            name="Fill_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, 2.35619 + angle_offset), # 45 0 135
            energy=random.choice([2, 3, 3.5]),
        )

        # Create small light
        small_light1 = self._create_light(
            name="S1_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(1.57079, 0, 0.785398 + angle_offset), # 90 0 45
            energy=random.choice([0.25, 0.5, 1]),
        )

        small_light2 = self._create_light(
            name="S2_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(1.57079, 0, 3.92699 + angle_offset), # 90 0 45
            energy=random.choice([0.25, 0.5, 1]),
        )

        # Create bottom light
        bottom_light = self._create_light(
            name="Bottom_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(3.14159, 0, 0), # 180 0 0
            energy=random.choice([1, 2, 3]),
        )

        return dict(
            key_light=key_light,
            fill_light=fill_light,
            rim_light=rim_light,
            bottom_light=bottom_light,
            small_light1=small_light1,
            small_light2=small_light2,
        )

    def add_lighting(self, option: str) -> None:
        assert option in ['fixed', 'random']
        
        # delete the default light
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
        
        # add a new light
        if option == 'fixed':
            #Make light just directional, disable shadows.
            bpy.ops.object.light_add(type='SUN')
            light = bpy.context.object
            light.name = 'Light'
            light.data.use_shadow = False
            # Possibly disable specular shading:
            light.data.specular_factor = 1.0
            light.data.energy = 5.0

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light2 = bpy.context.object
            light2.name = 'Light2'
            light2.data.use_shadow = False
            light2.data.specular_factor = 1.0
            light2.data.energy = 3 #0.015
            bpy.data.objects['Light2'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light2'].rotation_euler[0] += 180

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light3 = bpy.context.object
            light3.name = 'Light3'
            light3.data.use_shadow = False
            light3.data.specular_factor = 1.0
            light3.data.energy = 3 #0.015
            bpy.data.objects['Light3'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light3'].rotation_euler[0] += 90

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light4'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light4'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light4'].rotation_euler[0] += -90

            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light5'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light5'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light5'].rotation_euler[1] += -90

            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light6'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light6'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light6'].rotation_euler[1] += 90

        elif option == 'random':
            bpy.ops.object.light_add(type="AREA")
            light = bpy.data.lights["Area"]
            light.energy = random.uniform(500000, 600000)
            bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
            bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
            bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

            # set light scale
            bpy.data.objects["Area"].scale[0] = 200
            bpy.data.objects["Area"].scale[1] = 200
            bpy.data.objects["Area"].scale[2] = 200


    def reset_scene(self) -> None:
        """Resets the scene to a clean state."""
        # delete everything that isn't part of a camera or a light
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        # delete all the materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        # delete all the textures
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        # delete all the images
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)


    # load the glb model
    def load_object(self, object_path: str) -> None:
        """Loads a glb model into the scene."""
        if object_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=object_path)
        elif object_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=object_path)
        elif object_path.endswith(".obj"):
            bpy.ops.import_scene.obj(filepath=object_path)
        else:
            raise ValueError(f"Unsupported file type: {object_path}")


    def scene_bbox(self, single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        for obj in self.scene_meshes() if single_obj is None else [single_obj]:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return Vector(bbox_min), Vector(bbox_max)


    def scene_root_objects(self):
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj


    def scene_meshes(self):
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, (bpy.types.Mesh)):
                yield obj


    def normalize_scene(self, box_scale: float):
        bbox_min, bbox_max = self.scene_bbox()
        scale = box_scale / max(bbox_max - bbox_min)
        for obj in self.scene_root_objects():
            obj.scale = obj.scale * scale
        # Rotate the object so that the front is facing the camera
        for obj in self.scene_root_objects():
            obj.rotation_euler = (0, 0, 0)
            obj.rotation_euler[2] = math.pi
            obj.rotation_euler[0] = math.pi / 2

        # Apply scale to matrix_world.
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        for obj in self.scene_root_objects():
            obj.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")


    def setup_camera(self):
        cam = self.scene.objects["Camera"]
        cam.location = (0, 3.0, 0.0)
        cam.data.lens = 33
        cam.data.sensor_width = 32
        cam.data.sensor_height = 32  # affects instrinsics calculation, should be set explicitly
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        return cam, cam_constraint


    def get_bsdf_node_from_material(self, mat):
        nodes = mat.node_tree.nodes

        node_types = []
        principled_bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled_bsdf_node.name = 'New BSDF'
        principled_bsdf_node.label = 'New BSDF'
        for node in nodes:
            node_types.append(node.type)
            if "BSDF" in node.type and 'New BSDF' not in node.name:
                # Create a new Principled BSDF node and link the color input
                # FIX: the input name may not be Color
                is_BSDF = True
                assert (
                    "Color" in node.inputs or "Base Color" in node.inputs 
                ), f"BSDF node {node.type} does not have a 'Color' input"

                color_input_name = 'Base Color' if node.type == 'BSDF_PRINCIPLED' else 'Color'

                if node.inputs[color_input_name].is_linked:
                    color_link = node.inputs[color_input_name].links[0].from_socket
                    mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Base Color"])
                else:
                    if not principled_bsdf_node.inputs["Base Color"].is_linked:
                        principled_bsdf_node.inputs["Base Color"].default_value = node.inputs[
                            color_input_name
                        ].default_value

                if "Roughness" in node.inputs:
                    if node.inputs["Roughness"].is_linked:
                        color_link = node.inputs["Roughness"].links[0].from_socket
                        mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Roughness"])
                    else:
                        principled_bsdf_node.inputs["Roughness"].default_value = node.inputs[
                            "Roughness"
                        ].default_value

                if "Metallic" in node.inputs:
                    if node.inputs["Metallic"].is_linked:
                        color_link = node.inputs["Metallic"].links[0].from_socket
                        mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Metallic"])
                    else:
                        principled_bsdf_node.inputs["Metallic"].default_value = node.inputs[
                            "Metallic"
                        ].default_value

                if "Normal" in node.inputs:
                    if node.inputs["Normal"].is_linked:
                        color_link = node.inputs["Normal"].links[0].from_socket
                        mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Normal"])
                    else:
                        principled_bsdf_node.inputs["Normal"].default_value = node.inputs[
                            "Normal"
                        ].default_value

        bump_linked = nodes['New BSDF'].inputs['Normal'].is_linked
        is_scan = set(['EMISSION', 'LIGHT_PATH', 'BSDF_TRANSPARENT', 'MIX_SHADER', 'TEX_IMAGE']).issubset(set(node_types))
        return bump_linked


    def assign_material_value(
        self, node_tree, combine_node, bsdf_node, channel_name, material_name, mesh_name=None, rand=False
        ):
            """
            Assigns the specified material property to a channel of the CombineRGB node.
            param combine_node: The CombineRGB node to which the material property will be assigned.
            param bsdf_node: The BSDF node from which the material property is sourced.
            param channel_name: The channel name in the CombineRGB node (e.g., "R", "G", or "B").
            param material_name: The name of the material property in the BSDF node (e.g., "Roughness", "Metallic").
            """
            if material_name not in bsdf_node.inputs:
                rand = True


            if rand:
                # TODO unused
                # if args.mat_assign_mode == "choices":
                rand_val = random.choice([0.05, 0.2, 0.5, 0.8, 1.0])
                    # rand_val = random.choice([0, 1])
                # elif args.mat_assign_mode == "random":
                #     rand_val = random.random()
                # else:
                #     raise ValueError(f"Invalid material assign mode: {args.mat_assign_mode}")

                combine_node.inputs[channel_name].default_value = rand_val
                print(mesh_name, "Random", material_name, "value:", rand_val)

                # # save
                # if self.material_data is None:
                #     self.material_data = {}

                # if mesh_name not in self.material_data:
                #     self.material_data[mesh_name] = {}

                # self.material_data[mesh_name][material_name] = rand_val
                return

            if bsdf_node.inputs[material_name].is_linked:
                input_link = bsdf_node.inputs[material_name].links[0].from_socket
                combine_node.inputs[channel_name].default_value = 1.0  # Ensure full value if linked
                node_tree.links.new(input_link, combine_node.inputs[channel_name])
            else:
                # assign a single value
                combine_node.inputs[channel_name].default_value = bsdf_node.inputs[
                    material_name
                ].default_value


    def get_material_nodes(self, materials_dir):
        """
        Update the material nodes to either display the base color (albedo) or a combination of roughness and metallic values.
        param use_albedo: If True, sets up for albedo map; otherwise, sets up for roughness-metallic map.
        param rand: If True, randomizes the roughness and metallic values.
        """

        # Create a new material
        material = bpy.data.materials.new(name="PBR_Material")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Clear all default nodes
        for node in nodes:
            nodes.remove(node)

        # Add a Principled BSDF node
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.name = 'New BSDF'
        bsdf.location = (0, 0)

        # Add a Material Output node
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        material_output.location = (300, 0)

        # Connect the BSDF node to the Material Output node
        links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

        # Load and connect the Albedo (Base Color) texture
        tex_image_albedo = nodes.new(type="ShaderNodeTexImage")
        tex_image_albedo.image = bpy.data.images.load(os.path.join(materials_dir, "final_albedo_after.png"))
        links.new(tex_image_albedo.outputs['Color'], bsdf.inputs['Base Color'])

        # Load and connect the Roughness and Metallic texture
        tex_image_roughness_metallic = nodes.new(type="ShaderNodeTexImage")
        tex_image_roughness_metallic.image = bpy.data.images.load(os.path.join(materials_dir, "final_rm_after.png"))
        tex_image_roughness_metallic.image.colorspace_settings.name = 'Non-Color'
        separate_xyz = nodes.new(type="ShaderNodeSeparateXYZ")
        links.new(tex_image_roughness_metallic.outputs['Color'], separate_xyz.inputs['Vector'])
        links.new(separate_xyz.outputs['Y'], bsdf.inputs['Roughness'])
        links.new(separate_xyz.outputs['Z'], bsdf.inputs['Metallic'])

        # Load and connect the Bump texture (using it as a Normal map)
        tex_image_bump = nodes.new(type="ShaderNodeTexImage")
        tex_image_bump.image = bpy.data.images.load(os.path.join(materials_dir, "final_bump_after.png"))
        tex_image_bump.image.colorspace_settings.name = 'Non-Color'

        # Add a Bump node to convert the Bump map to a Normal map
        normal_node = nodes.new(type="ShaderNodeNormalMap")
        normal_node.inputs["Strength"].default_value = 4.0
        links.new(tex_image_bump.outputs['Color'], normal_node.inputs['Color'])
        links.new(normal_node.outputs['Normal'], bsdf.inputs['Normal'])
        self.original_mats = []
        for obj in bpy.context.scene.objects:
            # each obj is a sub-mesh
            if not (obj.type == "MESH"):
                continue

            # Assign the material to the active object
            obj.active_material = material

            # Apply smooth shading
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()
            self.original_mats.append(material)

        

    def update_material_nodes(self, mode, rand=False):
        """
        Update the material nodes to either display the base color (albedo) or a combination of roughness and metallic values.
        param use_albedo: If True, sets up for albedo map; otherwise, sets up for roughness-metallic map.
        param rand: If True, randomizes the roughness and metallic values.
        """

        mat_id = 0
        for obj in bpy.context.scene.objects:
            # each obj is a sub-mesh
            if not (obj.type == "MESH"):
                continue

            mesh_name = obj.name

            mat = self.original_mats[mat_id]
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            mat_id = mat_id + 1

            # process the new material
            # for mat in obj.data.materials:
            if not (mat and mat.node_tree):
                continue
            
            nodes = mat.node_tree.nodes
            principled_bsdf_node = nodes['New BSDF']
            if not principled_bsdf_node:
                continue  # Skip this mesh if no suitable BSDF node is found

            emission_node = nodes.new(type="ShaderNodeEmission")
            if mode == "albedo":
                # Link albedo to emission
                if principled_bsdf_node.inputs["Base Color"].is_linked:
                    input_link = principled_bsdf_node.inputs["Base Color"].links[0].from_socket
                    mat.node_tree.links.new(input_link, emission_node.inputs["Color"])
                else:
                    emission_node.inputs["Color"].default_value = principled_bsdf_node.inputs[
                        "Base Color"
                    ].default_value
            elif mode == "roughness_metallic":
                # mode is like "roughness_metallic"
                # Create a mix of roughness (G) and metallic (B) values
                combine_node = nodes.new(type="ShaderNodeCombineRGB")
                mat_fn = lambda ch, name: self.assign_material_value( mat.node_tree,
                    combine_node, principled_bsdf_node, ch, name, mesh_name, rand=rand
                )

                combine_node.inputs["R"].default_value = 1  # R is fixed
                # mat_fn("R", "Specular")
                mat_fn("G", "Roughness")
                mat_fn("B", "Metallic")
                mat.node_tree.links.new(
                    combine_node.outputs["Image"], emission_node.inputs["Color"]
                )
            elif mode == "bump":
                if principled_bsdf_node.inputs['Normal'].is_linked:
                    normal_map_node = principled_bsdf_node.inputs['Normal'].links[0].from_node
                    if normal_map_node.type == 'NORMAL_MAP' and normal_map_node.space == 'TANGENT':
                        material_node = normal_map_node.inputs['Color'].links[0].from_node
                        if material_node.type == 'TEX_IMAGE':
                            mat.node_tree.links.new(
                                material_node.outputs["Color"], emission_node.inputs["Color"]
                            )
                        else:
                            emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                    else:
                        emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                else:
                    emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
            elif mode == "position":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Position'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])
            elif mode == "normal":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Normal'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])
            elif mode == "mesh":
                # remove the link and set the value
                if principled_bsdf_node.inputs["Base Color"].is_linked:
                    mat.node_tree.links.remove(principled_bsdf_node.inputs["Base Color"].links[0])
                    # principled_bsdf_node.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
                    # default color 08d9d6
                    principled_bsdf_node.inputs["Base Color"].default_value = (0.03, 0.85, 0.85, 1.0)
                if principled_bsdf_node.inputs["Roughness"].is_linked:
                    mat.node_tree.links.remove(principled_bsdf_node.inputs["Roughness"].links[0])
                    principled_bsdf_node.inputs["Roughness"].default_value = 0.5
                if principled_bsdf_node.inputs["Metallic"].is_linked:
                    mat.node_tree.links.remove(principled_bsdf_node.inputs["Metallic"].links[0])
                    principled_bsdf_node.inputs["Metallic"].default_value = 0.0
                if principled_bsdf_node.inputs["Normal"].is_linked:
                    mat.node_tree.links.remove(principled_bsdf_node.inputs["Normal"].links[0])


            if mode == 'rgba_new' or mode == 'mesh':
                mat.node_tree.links.new(
                    principled_bsdf_node.outputs['BSDF'],
                    nodes["Material Output"].inputs["Surface"],
                )
            else:
                # Connect emission to material output
                mat.node_tree.links.new(
                    emission_node.outputs["Emission"],
                    nodes["Material Output"].inputs["Surface"],
                )

    def update_material_nodes_uv(self, mode, rand=False):
        """
        Update the material nodes to either display the base color (albedo) or a combination of roughness and metallic values.
        param use_albedo: If True, sets up for albedo map; otherwise, sets up for roughness-metallic map.
        param rand: If True, randomizes the roughness and metallic values.
        """

        for mat in bpy.data.materials:
            nodes = mat.node_tree.nodes
            if 'New BSDF' not in nodes.keys():
                continue

            principled_bsdf_node = nodes['New BSDF']
            if not principled_bsdf_node:
                continue  # Skip this mesh if no suitable BSDF node is found

            emission_node = nodes.new(type="ShaderNodeEmission")
            if mode == "albedo":
                # Link albedo to emission
                if principled_bsdf_node.inputs["Base Color"].is_linked:
                    input_link = principled_bsdf_node.inputs["Base Color"].links[0].from_socket
                    mat.node_tree.links.new(input_link, emission_node.inputs["Color"])
                else:
                    emission_node.inputs["Color"].default_value = principled_bsdf_node.inputs[
                        "Base Color"
                    ].default_value
            elif mode == "roughness_metallic":
                # mode is like "roughness_metallic"
                # Create a mix of roughness (G) and metallic (B) values
                combine_node = nodes.new(type="ShaderNodeCombineRGB")
                mat_fn = lambda ch, name: self.assign_material_value( mat.node_tree,
                    combine_node, principled_bsdf_node, ch, name, rand=rand
                )

                combine_node.inputs["R"].default_value = 1  # R is fixed
                # mat_fn("R", "Specular")
                mat_fn("G", "Roughness")
                mat_fn("B", "Metallic")
                mat.node_tree.links.new(
                    combine_node.outputs["Image"], emission_node.inputs["Color"]
                )
            elif mode == "bump":
                if principled_bsdf_node.inputs['Normal'].is_linked:
                    normal_map_node = principled_bsdf_node.inputs['Normal'].links[0].from_node
                    if normal_map_node.type == 'NORMAL_MAP' and normal_map_node.space == 'TANGENT':
                        material_node = normal_map_node.inputs['Color'].links[0].from_node
                        if material_node.type == 'TEX_IMAGE':
                            mat.node_tree.links.new(
                                material_node.outputs["Color"], emission_node.inputs["Color"]
                            )
                        else:
                            emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                    else:
                        emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                else:
                    emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
            elif mode == "position":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Position'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])
            elif mode == "normal":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Normal'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])


            if mode == 'rgba_new':
                mat.node_tree.links.new(
                    principled_bsdf_node.outputs['BSDF'],
                    nodes["Material Output"].inputs["Surface"],
                )
            else:
                # Connect emission to material output
                mat.node_tree.links.new(
                    emission_node.outputs["Emission"],
                    nodes["Material Output"].inputs["Surface"],
                )


    def save_rgb_images(self, extra_outs=['normal', 'depth', 'ccm'], save_camera=True) -> None:
        """Saves rendered images of the object in the scene."""

        self.scene.view_settings.view_transform = 'Raw'
        os.makedirs(self.args.output_dir, exist_ok=True)

        # prepare to save
        pose_dir = os.path.join(self.args.output_dir, self.object_uid, 'pose')
        os.makedirs(pose_dir, exist_ok=True)

        extra_dirs = {}
        for t in extra_outs:
            temp_dir = os.path.join(self.args.output_dir, self.object_uid, t)
            os.makedirs(temp_dir, exist_ok=True)
            extra_dirs[t] = temp_dir

        # create input render layer node
        for node in self.tree.nodes:
            if node.bl_idname == 'CompositorNodeRLayers' or node.bl_idname == 'CompositorNodeComposite':
                self.tree.nodes.remove(node)
        
        render_layers = self.tree.nodes.new('CompositorNodeRLayers')
        render_layers.label = 'Custom Outputs'
        render_layers.name = 'Custom Outputs'

        rgba_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
        rgba_file_output.label = 'rgba'
        rgba_file_output.name = 'rgba'
        rgba_file_output.base_path = ''
        
        self.tree.links.new(render_layers.outputs['Image'], rgba_file_output.inputs['Image'])
        # self.tree.nodes['Composite'].color_management_rules.use = True
        # rule1 = self.tree.nodes['Composite'].color_management_rules.new(name="Output1 Rule")
        # rule1.view_transform = "Standard"

        if 'depth' in extra_outs:
            # create depth output node
            depth_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
            depth_file_output.label = 'depth'
            depth_file_output.name = 'depth'
            depth_file_output.base_path = ''

            # remap as image types can not represent the full range of depth
            map_depth = self.tree.nodes.new(type="CompositorNodeMapRange")
            # size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map
            map_depth.inputs['From Min'].default_value = 0.555
            map_depth.inputs['From Max'].default_value = 2.222
            map_depth.inputs['To Min'].default_value = 0
            map_depth.inputs['To Max'].default_value = 1

            divide_node = self.tree.nodes.new(type="CompositorNodeMath")
            divide_node.operation = 'DIVIDE'
            divide_node.inputs[0].default_value = 2.0

            self.links.new(render_layers.outputs['Depth'], divide_node.inputs[1])
            self.links.new(divide_node.outputs[0], map_depth.inputs[0])

            # add alpha channel
            set_alpha_node = self.tree.nodes.new(type='CompositorNodeSetAlpha')
            set_alpha_node.mode = 'REPLACE_ALPHA'
            self.tree.links.new(map_depth.outputs[0], set_alpha_node.inputs['Image'])
            self.tree.links.new(render_layers.outputs['Alpha'], set_alpha_node.inputs['Alpha'])

            self.links.new(set_alpha_node.outputs['Image'], depth_file_output.inputs['Image'])

        if "normal" in extra_outs:
            # create normal output node
            normal_file_output = self.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
            normal_file_output.label = "normal"
            normal_file_output.name = "normal"
            normal_file_output.base_path = ""

            # Create a Separate RGB node
            separate_rgb_node = self.scene.node_tree.nodes.new(type="CompositorNodeSepRGBA")
            self.scene.node_tree.links.new(render_layers.outputs["Normal"], separate_rgb_node.inputs["Image"])

            # Create a Combine RGBA node
            combine_rgba_node = self.scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
            reversed_G = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            reversed_G.operation = 'MULTIPLY'
            reversed_G.inputs[0].default_value = -1

            # y -> -y
            self.scene.node_tree.links.new(separate_rgb_node.outputs["G"], reversed_G.inputs[1])

            # map normal range from [-1, 1] to [0, 1]
            # channel R
            bias_node_R = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            bias_node_R.operation = "ADD"
            bias_node_R.inputs[0].default_value = 1

            scale_node_R = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            scale_node_R.operation = "MULTIPLY"
            scale_node_R.inputs[0].default_value = 0.5

            self.scene.node_tree.links.new(separate_rgb_node.outputs["R"], bias_node_R.inputs[1])
            self.scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

            # channel G
            bias_node_G = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            bias_node_G.operation = "ADD"
            bias_node_G.inputs[0].default_value = 1

            scale_node_G = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            scale_node_G.operation = "MULTIPLY"
            scale_node_G.inputs[0].default_value = 0.5

            self.scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
            self.scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

            # channel B
            bias_node_B = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            bias_node_B.operation = "ADD"
            bias_node_B.inputs[0].default_value = 1

            scale_node_B = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            scale_node_B.operation = "MULTIPLY"
            scale_node_B.inputs[0].default_value = 0.5

            self.scene.node_tree.links.new(separate_rgb_node.outputs["B"], bias_node_B.inputs[1])
            self.scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])


            # Combine RGB
            self.scene.node_tree.links.new(combine_rgba_node.inputs["R"], scale_node_R.outputs[0])
            self.scene.node_tree.links.new(combine_rgba_node.inputs["G"], scale_node_B.outputs[0])
            self.scene.node_tree.links.new(combine_rgba_node.inputs["B"], scale_node_G.outputs[0])
            self.scene.node_tree.links.new(combine_rgba_node.inputs["A"], separate_rgb_node.outputs["A"])


            # add alpha channel
            set_alpha_node = self.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
            set_alpha_node.mode = "REPLACE_ALPHA"
            self.scene.node_tree.links.new(
                combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
            )
            self.scene.node_tree.links.new(
                render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
            )

            self.scene.node_tree.links.new(
                set_alpha_node.outputs["Image"], normal_file_output.inputs["Image"]
            )

        if "ccm" in extra_outs:
            # create CCM output node
            ccm_file_output = self.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
            ccm_file_output.label = "ccm"
            ccm_file_output.name = "ccm"
            ccm_file_output.base_path = ""

            # Create a Separate RGB node
            separate_xyz_node = self.scene.node_tree.nodes.new(type="CompositorNodeSeparateXYZ")
            self.scene.node_tree.links.new(render_layers.outputs["Position"], separate_xyz_node.inputs["Vector"])

            # Create a Combine RGBA node
            combine_rgba_node = self.scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
            reversed_G = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            reversed_G.operation = 'MULTIPLY'
            reversed_G.inputs[0].default_value = -1

            # y -> -y
            self.scene.node_tree.links.new(separate_xyz_node.outputs["Y"], reversed_G.inputs[1])

            # map ccm range from [-1, 1] to [0, 1]
            # channel R
            bias_node_R = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            bias_node_R.operation = "ADD"
            bias_node_R.inputs[0].default_value = 1

            scale_node_R = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            scale_node_R.operation = "MULTIPLY"
            scale_node_R.inputs[0].default_value = 0.5

            self.scene.node_tree.links.new(separate_xyz_node.outputs["X"], bias_node_R.inputs[1])
            self.scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

            # channel G
            bias_node_G = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            bias_node_G.operation = "ADD"
            bias_node_G.inputs[0].default_value = 1

            scale_node_G = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            scale_node_G.operation = "MULTIPLY"
            scale_node_G.inputs[0].default_value = 0.5

            self.scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
            self.scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

            # channel B
            bias_node_B = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            bias_node_B.operation = "ADD"
            bias_node_B.inputs[0].default_value = 1

            scale_node_B = self.scene.node_tree.nodes.new(type="CompositorNodeMath")
            scale_node_B.operation = "MULTIPLY"
            scale_node_B.inputs[0].default_value = 0.5

            self.scene.node_tree.links.new(separate_xyz_node.outputs["Z"], bias_node_B.inputs[1])
            self.scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])


            # Combine RGB
            self.scene.node_tree.links.new(combine_rgba_node.inputs["R"], scale_node_R.outputs[0])
            self.scene.node_tree.links.new(combine_rgba_node.inputs["G"], scale_node_B.outputs[0])
            self.scene.node_tree.links.new(combine_rgba_node.inputs["B"], scale_node_G.outputs[0])


            # add alpha channel
            set_alpha_node = self.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
            set_alpha_node.mode = "REPLACE_ALPHA"
            self.scene.node_tree.links.new(
                combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
            )
            self.scene.node_tree.links.new(
                render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
            )

            self.scene.node_tree.links.new(
                set_alpha_node.outputs["Image"], ccm_file_output.inputs["Image"]
            )

        for i in range(self.args.num_images):
            # set the camera position
            self.camera.location = self.cam_locations[i]
            self.camera.rotation_euler = self.cam_rotations[i]

            # render the image
            # render_path = os.path.join(img_dir, f"{i:03d}_0001.png")
            # self.render.filepath = render_path

            for out in extra_outs:
                render_path = os.path.join(extra_dirs[out], f"{i:03d}_")
                # self.tree.nodes[out].file_slots[0].format.color_management = 'OVERRIDE'
                # # self.tree.nodes[out].file_slots[0].format.has_linear_colorspace = True
                # self.tree.nodes[out].file_slots[0].format.view_settings.view_transform = 'Raw'
                # self.tree.nodes[out].file_slots[0].format.display_settings.display_device = 'sRGB'
                self.tree.nodes[out].file_slots[0].path = render_path
                if out == 'rgba':
                    self.tree.nodes[out].file_slots[0].save_as_render = False
                else:
                    self.tree.nodes[out].file_slots[0].save_as_render = True
            bpy.ops.render.render(write_still=False)
                
            # save camera RT matrix (W2C)
            RT_path = os.path.join(pose_dir, f"{i:03d}_0001.npy")
            if save_camera:
                # location, rotation = self.camera.matrix_world.decompose()[0:2]
                # RT = self.compose_RT(rotation.to_matrix(), np.array(location))
                RT = self.get_3x4_RT_matrix_from_blender(self.camera)
                np.save(RT_path, RT)
        
        # save the camera intrinsics
        if save_camera:
            intrinsics = self.get_calibration_matrix_K_from_blender(self.camera.data, return_principles=True)
            with open(os.path.join(self.args.output_dir, self.object_uid,'intrinsics.npy'), 'wb') as f_intrinsics:
                np.save(f_intrinsics, intrinsics)

        for node in self.tree.nodes:
            if 'Compositor' in node.bl_idname:
                self.tree.nodes.remove(node)
        self.scene.use_nodes = False


    def render_views(self, mode='albedo', env_map=None, save_camera=False) -> None:
        """Saves rendered images of the object in the scene."""
        if mode != 'rgba':
            self.update_material_nodes(mode)
        os.makedirs(self.args.output_dir, exist_ok=True)

        # prepare to save
        if env_map is None:
            img_dir = os.path.join(self.args.output_dir, self.object_uid, mode)
        else:
            img_dir = os.path.join(self.args.output_dir, self.object_uid, mode, env_map)
        pose_dir = os.path.join(self.args.output_dir, self.object_uid, 'pose')

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        CIRCLE_FIXED_START = (0,0,0)
        CIRCLE_FIXED_END = (0,0,0)

        stepsize = 360.0 / self.args.num_images
        vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]

        self.empty.rotation_euler = CIRCLE_FIXED_START
        self.empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff

        if mode == 'albedo':
            self.scene.view_settings.view_transform = 'Standard'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 128
        elif mode == 'roughness_metallic' or mode == 'bump':
            self.scene.view_settings.view_transform = 'Raw'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 128
        elif mode == 'rgba' or mode == 'mesh':
            self.scene.view_settings.view_transform = 'Standard'
            self.scene.cycles.use_denoising = True
            self.scene.cycles.samples = 128

        if env_map is not None:
            env_map_path = None
            for hdr_file in self.HDR_files:
                if env_map in hdr_file:
                    env_map_path = hdr_file
                    break

            if env_map_path is None:
                raise ValueError(f"Environment map {env_map} not found in the HDR files.")
            
            # set the environment map
            bpy.context.scene.world.node_tree.nodes['HDRTex'].image = bpy.data.images.load(env_map_path)
            # set the strength of the environment map
            bpy.context.scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 2.0


        from math import radians
        for i in range(self.args.num_images):
            # set the camera position
            # self.camera.location = self.cam_locations[i]
            # self.camera.rotation_euler = self.cam_rotations[i]

            # render the image
            render_path = os.path.join(img_dir, f"{i:03d}_0001.png")
            self.render.filepath = render_path
            
            # set HDR rotation
            bpy.context.scene.world.node_tree.nodes['Mapping'].inputs["Rotation"].default_value[2] = radians(stepsize*i)

            # Apply smooth shading and Subdivision Surface Modifier
            for obj in bpy.context.scene.objects:
                if obj.type == "MESH":
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)  # Ensure the object is selected

                    # Apply smooth shading
                    bpy.ops.object.shade_smooth()

                    # # Add Subdivision Surface Modifier
                    # if not obj.modifiers.get("Subdivision"):
                    #     subsurf_mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
                    #     subsurf_mod.levels = 2  # Viewport subdivision level
                    #     subsurf_mod.render_levels = 3  # Render subdivision level

                    # # Apply the modifier
                    # bpy.ops.object.modifier_apply(modifier="Subdivision")

                    # Enable Auto Smooth
                    # obj.data.use_auto_smooth = True
                    # obj.data.auto_smooth_angle = radians(30)  # Set the angle threshold

                    # obj.select_set(False)  # Deselect the object after applying smooth shading

            bpy.ops.render.render(write_still=True)

            # save camera RT matrix (C2W)
            RT_path = os.path.join(pose_dir, f"{i:03d}.npy")
            if save_camera:
                location, rotation = self.camera.matrix_world.decompose()[0:2]
                RT = self.compose_RT(rotation.to_matrix(), np.array(location))
                np.save(RT_path, RT)

            self.empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
            self.empty.rotation_euler[2] += radians(2*stepsize)
        
        # save the camera intrinsics
        if save_camera:
            intrinsics = self.get_calibration_matrix_K_from_blender(self.camera.data, return_principles=True)
            with open(os.path.join(self.args.output_dir, self.object_uid,'intrinsics.npy'), 'wb') as f_intrinsics:
                np.save(f_intrinsics, intrinsics)

        # assemble the video
        import imageio
        all_images = []
        for i in range(self.args.num_images):
            image = imageio.v3.imread(os.path.join(img_dir, f"{i:03d}_0001.png"))
            mask = image[..., 3:4] > 0
            image = image * mask + (1. - mask) * 255
            image = image.astype(np.uint8)
            if mode == 'roughness_metallic':
                image_roughness = image[..., 1:2].repeat(3, axis=-1)
                image_metallic = image[..., 2:3].repeat(3, axis=-1)
                imageio.v3.imwrite(os.path.join(img_dir, f"{i:03d}_roughness.png"), image_roughness)
                imageio.v3.imwrite(os.path.join(img_dir, f"{i:03d}_metallic.png"), image_metallic)
            all_images.append(image)
        all_images = np.stack(all_images, axis=0)
        if mode != 'roughness_metallic':
            if env_map is not None:
                out_video_path = os.path.join(self.args.output_dir, f"{self.object_uid}_{mode}_{env_map}.mp4")
            else:
                out_video_path = os.path.join(self.args.output_dir, f"{self.object_uid}_{mode}.mp4")
            imageio.v3.imwrite(out_video_path, all_images, fps=30, quality=9)
        else:
            all_images_roughness = all_images[..., 1:2].repeat(3, axis=-1)
            all_images_metallic = all_images[..., 2:3].repeat(3, axis=-1)
            out_video_path_roughness = os.path.join(self.args.output_dir, f"{self.object_uid}_roughness.mp4")
            out_video_path_metallic = os.path.join(self.args.output_dir, f"{self.object_uid}_metallic.mp4")
            imageio.v3.imwrite(out_video_path_roughness, all_images_roughness, fps=30, quality=9)
            imageio.v3.imwrite(out_video_path_metallic, all_images_metallic, fps=30, quality=9)

    
    def combine_objects(self):
        bpy.ops.object.select_all(action='DESELECT')
        # import pdb; pdb.set_trace()
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.select_set(True)

        bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
        if len(bpy.context.selected_objects) > 1:
            bpy.ops.object.join()

            bpy.ops.object.mode_set(mode='EDIT')
            obj = bpy.context.active_object
            obj.data.uv_layers.new(name="NewUV")
            obj.data.uv_layers.active_index = len(obj.data.uv_layers) - 1
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)
            # bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')
    

    
    def bake_material_images(self, image_name, bake_type, color_space='sRGB', width=1024, height=1024):
        """Saves baked images of the object in the scene."""

        self.update_material_nodes_uv(mode=image_name)

        # define a new baking image
        bpy.ops.image.new(name=image_name, width=width, height=height)
        bake_image = bpy.data.images[image_name]
        bake_image.colorspace_settings.name = color_space

        # prepare to save
        os.makedirs(self.args.output_dir, exist_ok=True)
        img_dir = os.path.join(self.args.output_dir, self.object_uid, 'material000')
        os.makedirs(img_dir, exist_ok=True)

        # add the bake image for each material
        for mat in bpy.data.materials:

            if 'New BSDF' not in mat.node_tree.nodes.keys():
                continue
            # add texture node into tree
            texture_node = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
            texture_node.image = bake_image
            mat.node_tree.nodes.active = texture_node

            # obj.data.materials.clear()
            # obj.data.materials.append(mat)

        # set bake type
        bpy.context.scene.cycles.bake_type = bake_type
        
        # select object and bake
        obj = bpy.context.view_layer.objects.active
        obj.select_set(True)

        if image_name == 'rgba_new':
            for i, HDR_file in enumerate(self.HDR_files):
                # set texture environment
                bpy.context.scene.world.node_tree.nodes['HDRTex'].image = bpy.data.images.load(HDR_file)
                bpy.ops.object.bake(type=bake_type)
                bake_image.filepath_raw = os.path.join(img_dir, f"{image_name.lower()}_test_{i:03d}.png")
                bake_image.file_format = 'PNG'
                bake_image.save()
        else:
            bpy.ops.object.bake(type=bake_type)
            bake_image.filepath_raw = os.path.join(img_dir, f"{image_name.lower()}.png")
            bake_image.file_format = 'PNG'
            bake_image.save()

        


    def download_object(self, object_url: str) -> str:
        """Download the object and return the path."""
        # uid = uuid.uuid4()
        uid = object_url.split("/")[-1].split(".")[0]
        tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
        local_path = os.path.join("tmp-objects", f"{uid}.glb")
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
        urllib.request.urlretrieve(object_url, tmp_local_path)
        os.rename(tmp_local_path, local_path)
        # get the absolute path
        local_path = os.path.abspath(local_path)
        return local_path


    def get_calibration_matrix_K_from_blender(self, camera, return_principles=False):
        """
            Get the camera intrinsic matrix from Blender camera.
            Return also numpy array of principle parameters if specified.
            
            Intrinsic matrix K has the following structure in pixels:
                [fx  0 cx]
                [0  fy cy]
                [0   0  1]
            
            Specified principle parameters are:
                [fx, fy] - focal lengths in pixels
                [cx, cy] - optical centers in pixels
                [width, height] - image resolution in pixels
            
        """
        # Render resolution
        render = bpy.context.scene.render
        width = render.resolution_x * render.pixel_aspect_x
        height = render.resolution_y * render.pixel_aspect_y

        # Camera parameters
        focal_length = camera.lens  # Focal length in millimeters
        sensor_width = camera.sensor_width  # Sensor width in millimeters
        sensor_height = camera.sensor_height  # Sensor height in millimeters

        # Calculate the focal length in pixel units
        focal_length_x = width * (focal_length / sensor_width)
        focal_length_y = height * (focal_length / sensor_height)

        # Assuming the optical center is at the center of the sensor
        optical_center_x = width / 2
        optical_center_y = height / 2

        # Constructing the intrinsic matrix
        K = np.array([[focal_length_x, 0, optical_center_x],
                    [0, focal_length_y, optical_center_y],
                    [0, 0, 1]])
        
        if return_principles:
            return np.array([
                [focal_length_x, focal_length_y],
                [optical_center_x, optical_center_y],
                [width, height],
            ])
        else:
            return K
        
    # function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
    def get_3x4_RT_matrix_from_blender(self, cam):
        # bcam stands for blender camera
        # R_bcam2cv = Matrix(
        #     ((1, 0,  0),
        #     (0, 1, 0),
        #     (0, 0, 1)))

        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam @ location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam @ cam.location
        # Use location from matrix_world to account for constraints:     
        T_world2bcam = -1*R_world2bcam @ location

        # # Build the coordinate transform matrix from world to computer vision camera
        # R_world2cv = R_bcam2cv@R_world2bcam
        # T_world2cv = R_bcam2cv@T_world2bcam

        # put into 3x4 matrix
        RT = Matrix((
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],)
            ))
        return RT
        
    
    def generate_pose(self) -> None:
        '''generate camera poses for rendering'''
        self.cam_locations = []
        self.cam_rotations = []
        radius = np.random.uniform(1.8, 2.2, 1)[0]
        for i in range(self.args.num_images):
            # camera_option = 'random' if i > 0 else 'front'
            camera_option = 'fixed'

            if camera_option == 'fixed':
                locations = [
                    [ 0, -1,  0],
                    [-1,  0,  0],
                    [ 0,  1,  0],
                    [ 1,  0,  0],
                    [-1, -1,  1],
                    [-1,  1,  1],
                    [ 1,  1,  1],
                    [ 1, -1,  1],
                    [ 0,  0,  1]
                ]
                vec = locations[i]
                vec = vec / np.linalg.norm(vec, axis=0) * radius
                x, y, z = vec
            elif camera_option == 'random':
                # from https://blender.stackexchange.com/questions/18530/
                x, y, z = self.sample_spherical(radius_min=1.6, radius_max=2.2, maxz=2.0, minz=-0.75)
            elif camera_option == 'front':
                x, y, z = 0, -np.random.uniform(1.6, 2.2, 1)[0], 0

            self.cam_locations.append((x, y, z))

            # adjust orientation
            direction = - Vector((x, y, z))
            rot_quat = direction.to_track_quat('-Z', 'Y')
            self.cam_rotations.append(rot_quat.to_euler())

        
    def init_scene(self, object_file: str) -> None:
        """Load object into the scene."""
        self.reset_scene()

        # load the object
        self.load_object(object_file)
        # self.object_uid = os.path.basename(object_file).split(".")[0]
        self.normalize_scene(box_scale=2)
        # self.add_lighting(option='fixed')
        # self.randomize_lighting()
        self.camera, self.cam_constraint = self.setup_camera()

        # create an empty object to track
        # self.empty = bpy.data.objects.new("Empty", None)
        def parent_obj_to_camera(b_camera):
            origin = (0, 0, 0)
            b_empty = bpy.data.objects.new("Empty", None)
            b_empty.location = origin
            b_camera.parent = b_empty  # setup parenting

            scn = bpy.context.scene
            scn.collection.objects.link(b_empty)
            bpy.context.view_layer.objects.active = b_empty
            # scn.objects.active = b_empty
            return b_empty
        self.empty = parent_obj_to_camera(self.camera)
        self.cam_constraint.target = self.empty

        # generate camera pose
        # self.generate_pose()

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        # set the world
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")

        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes.clear()
        background_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeBackground")
        env_texture_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        env_texture_node.name = "HDRTex"
        output_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeOutputWorld")

        bpy.context.scene.world.node_tree.links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
        bpy.context.scene.world.node_tree.links.new(background_node.outputs["Background"], output_node.inputs["Surface"])

        # add environment texture
        HDR_path = './hdr_env'
        HDR_files = [f for f in os.listdir(HDR_path) if f.endswith('.exr')]
        self.HDR_files = [os.path.join(HDR_path, f) for f in HDR_files]

        bpy.context.scene.world.node_tree.nodes['HDRTex'].image = bpy.data.images.load(self.HDR_files[0])


        # Add Texture Coordinate and Mapping nodes for rotation
        coord_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeTexCoord")
        mapping_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeMapping")
        mapping_node.name = "Mapping"

        # Connect the coordinate node to the mapping node
        bpy.context.scene.world.node_tree.links.new(coord_node.outputs["Generated"], mapping_node.inputs["Vector"])

        # Connect the mapping node to the environment texture node for rotation control
        bpy.context.scene.world.node_tree.links.new(mapping_node.outputs["Vector"], env_texture_node.inputs["Vector"])

        # Set rotation on the Z-axis (in radians), adjust as needed
        # Example: Rotate by 90 degrees (π/2 radians)
        mapping_node.inputs["Rotation"].default_value[2] = 0  # 90 degrees in radians



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--materials_dir",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/views")
    parser.add_argument("--obj_uid", type=str, default="0")
    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
    )
    parser.add_argument("--num_images", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--part_id", type=int, default=0)
        
    # argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args()

    print('===================', args.engine, '===================')

    render = BlenderRendering(args)


    start_i = time.time()

    local_path = args.object_path

    render.init_scene(local_path)
    render.get_material_nodes(args.materials_dir)
    render.render_views(mode='rgba', env_map='forest')
    # render.render_views(mode='rgba', env_map='city')
    # render.render_views(mode='rgba', env_map='interior')
    render.render_views(mode='albedo')
    render.render_views(mode='roughness_metallic')
    render.render_views(mode='bump')
    render.render_views(mode='mesh', env_map='forest')

    end_i = time.time()
    print("Finished " + local_path + " in " + f"{end_i - start_i}" + " seconds")

