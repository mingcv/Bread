#######################################
# Breaking Down the Darkness : Testing
#######################################

# Using gamma correction for evaluation with parameter --gc
# Show extra intermediate outputs with parameter --save_extra
CUDA_VISIBLE_DEVICES=0 python eval_Bread.py -m1 IAN -m2 ANSN -m3 FuseNet -m4 FuseNet --mef --comment Bread+NFM+ME[eval] --batch_size 1 -m1w ./checkpoints/IAN_335.pth -m2w ./checkpoints/ANSN_422.pth -m3w ./checkpoints/FuseNet_MECAN_251.pth -m4w ./checkpoints/FuseNet_NFM_297.pth
CUDA_VISIBLE_DEVICES=0 python test_Bread.py -m1 IAN -m2 ANSN -m3 FuseNet -m4 FuseNet --mef --comment Bread+NFM+ME[test] --batch_size 1 -m1w ./checkpoints/IAN_335.pth -m2w ./checkpoints/ANSN_422.pth -m3w ./checkpoints/FuseNet_MECAN_251.pth -m4w ./checkpoints/FuseNet_NFM_297.pth

# Using CAN w/o MEF data
CUDA_VISIBLE_DEVICES=0 python eval_Bread.py -m1 IAN -m2 ANSN -m3 IAN -m4 FuseNet --comment Bread+NFM[eval] --batch_size 1 -m1w ./checkpoints/IAN_335.pth -m2w ./checkpoints/ANSN_422.pth -m3w ./checkpoints/IANet_CAN_51.pth -m4w ./checkpoints/FuseNet_NFM_297.pth
CUDA_VISIBLE_DEVICES=0 python test_Bread.py -m1 IAN -m2 ANSN -m3 IAN -m4 FuseNet --comment Bread+NFM[test] --batch_size 1 -m1w ./checkpoints/IAN_335.pth -m2w ./checkpoints/ANSN_422.pth -m3w ./checkpoints/IANet_CAN_51.pth -m4w ./checkpoints/FuseNet_NFM_297.pth

# Remove NFM for much better generalization performance
# changing parameter a for an ideal denoising strength
CUDA_VISIBLE_DEVICES=0 python test_Bread_NoNFM.py -m1 IAN -m2 ANSN -m3 FuseNet --mef -a 0.10 --comment Bread+ME[test] --batch_size 1 -m1w ./checkpoints/IAN_335.pth -m2w ./checkpoints/ANSN_422.pth -m3w ./checkpoints/FuseNet_MECAN_251.pth

##############################################################
# Breaking Down the Darkness : Training
# SICE dataset and LOL dataset are need to be download online
##############################################################

# Multi-exposure data synthesis
python exposure_augment.py

# Train IAN
CUDA_VISIBLE_DEVICES=0 python train_IAN.py -m IAN --comment IAN_train --batch_size 1 --val_interval 1 --num_epochs 500 --lr 0.001 --no_sche

# Train ANSN
CUDA_VISIBLE_DEVICES=0 python train_ANSN.py -m1 IAN -m2 ANSN --comment ANSN_train --batch_size 1 --val_interval 1 --num_epochs 500 --lr 0.001 --no_sche -m1w ./checkpoints/IAN_335.pth

# Train CAN
CUDA_VISIBLE_DEVICES=0 python train_CAN.py -m1 IAN -m3 FuseNet --comment CAN_train --batch_size 1 --val_interval 1 --num_epochs 500 --lr 0.001 --no_sche -m1w ./checkpoints/IAN_335.pth

# Train MECAN on SICE
CUDA_VISIBLE_DEVICES=0 python train_MECAN.py -m FuseNet --comment MECAN_train --batch_size 1 --val_interval 1 --num_epochs 500 --lr 0.001 --no_sche

# Finetune MECAN on SICE and LOL datasets
CUDA_VISIBLE_DEVICES=0 python train_MECAN_finetune.py -m FuseNet --comment MECAN_finetune --batch_size 1 --val_interval 1 --num_epochs 500 --lr 1e-4 --no_sche -mw ./checkpoints/FuseNet_MECAN_for_Finetuning_404.pth



