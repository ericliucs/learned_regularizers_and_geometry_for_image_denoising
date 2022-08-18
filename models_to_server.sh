#!/bin/bash
#echo Attempting to back up files
#echo Make sure to change old version!!!
#ssh collaborators@mathcs.duq.edu 'bash'< backup_script.sh
rsync -r -av --progress denoising/models/saved_models collaborators@mathcs.duq.edu:/work/collaborators/Ryan/learned_regularizers_and_geometry_for_image_denoising/denoising/models
echo Code has been updated