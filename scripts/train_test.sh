#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps
export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug
ID=159
#CUDA_VISIBLE_DEVICES=1 python tools/mobot_trainer.py --config-file 'configs/transfiner/mask_rcnn_R_101_FPN_3x.yaml' --model 'pretrained_model/output_3x_transfiner_r101.pth' --train 'MOBOT_Train' --test 'MOBOT_Val' --max-iter 10000 --output-dir 'model_0001' --base-ir 0.001 --checkpoint-period 500 --eval-period 10000 > log.txt
CUDA_VISIBLE_DEVICES=0,1,2 python tools/mobot_trainer.py --config-file 'configs/transfiner/mask_rcnn_R_101_FPN_3x.yaml' --model 'pretrained_model/output_3x_transfiner_r101.pth' --train 'MOBOT_Train_end' --test 'MOBOT_Val_end' --max-iter 10000 --output-dir 'model_0002' --base-ir 0.001 --checkpoint-period 500 --eval-period 500 > log.txt