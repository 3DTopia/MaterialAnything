
#!/bin/bash

mesh_dir="./demo"  # Replace with the actual dir containing the mesh files
uid=20250609
video_dir=./outputs/a_baseline_ours_texture # The directory where the video will be saved
output_dir=./outputs/$uid

# Gnerate PBR texture for the mesh
python scripts/generate_texture_pbr_3d.py \
    --image2materials_model ./pretrained_models/material_estimator \
    --uvrefine_model ./pretrained_models/material_refiner \
    --input_dir $mesh_path \
    --output_dir $output_dir \
    --obj_name mesh \
    --obj_file mesh.obj \
    --prompt "a rusty orange robot, ultra realistic" \
    --add_view_to_prompt \
    --ddim_steps 50 \
    --new_strength 1 \
    --update_strength 0.5 \
    --view_threshold 0.1 \
    --blend 0 \
    --dist 1.2 \
    --num_viewpoints 36 \
    --viewpoint_mode predefined \
    --use_principle \
    --update_steps 0 \
    --update_mode heuristic \
    --seed 42 \
    --post_process \
    --device 2080 \
    --use_objaverse # assume the mesh is normalized with y-axis as up
    
# Render the video using Blender
python render_video_bpy_4s.py \
    --object_path $output_dir/generate/mesh/0.obj \
    --materials_dir $output_dir/generate/material \
    --output_dir $video_dir \
    --obj_uid $uid


