#! /bin/bash

# Runs generate_algebraic_contours with various input meshes and cameras
output_dir=output
bin_dir=../build/bin
data_dir=../data/meshes
camera_dir=../data/cameras

model_list=($(ls ${data_dir}))

# ${output_dir}/fig-timing

# Loop through meshes
# for file in ${model_list[@]}
# do
#   if [[ "$file" = *".obj" ]]; then
#     model=${file%.*}
#     ${bin_dir}/generate_timing_data \
#       -i ${data_dir}/${model}.obj \
#       -o ${camera_dir}/
#   fi
# done

${bin_dir}/generate_algebraic_contours \
    -i ${data_dir}/tent_tri_clean_conf_simplified_with_uv.obj \
    -c ${camera_dir}/camera_matrix_identity.csv
