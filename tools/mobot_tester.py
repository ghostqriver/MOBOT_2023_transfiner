import logging
import os
from collections import OrderedDict
import warnings
import glob

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import  default_setup, default_writers, launch,hooks
from mobot.engine import mobot_argument_parser,Mobot_Dataset_Register,Mobot_Default_Setting,mobot_default_setup
from mobot.engine import Mobot_DefaultTrainer as Trainer
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    verify_results
)

from mobot.utils import (
    check_path,
    read_path,
    read_filename,
    read_finalfoldername
    )
from mobot.utils import (read_scores,plot_test)

logger = logging.getLogger("Mobot")

# If call by functions
class ARGS:
    def __init__(
            self,config_file=None,model=None,train=None,test=None,batch_size=None,
            base_ir=None,max_iter=None,checkpoint_period=None,eval_period=None,
            resume=None,eval_only=None,num_gpus=1,num_machines=1,machine_rank=0,dist_url=None
            ):
        self.config_file = config_file
        self.model = model
        self.train = train
        self.test = test
        self.batch_size=batch_size
        self.base_ir = base_ir
        self.max_iter = max_iter
        self.checkpoint_period = checkpoint_period
        self.eval_period = eval_period
        self.resume = resume
        self.eval_only = eval_only
        self.num_gpus = num_gpus
        self.num_machines = num_machines 
        self.machine_rank = machine_rank
        self.dist_url = dist_url


def read_models(args):
    if args.models_path != None:
        models = glob.glob(args.models_path + '/*.pth')
        file_name = read_finalfoldername(args.models_path)
    else:
        models = [args.model]
        file_name = read_filename(args.model)
    return models,file_name+'_test.npy'


def test_setup(cfg,model):
    cfg.MODEL.WEIGHTS = model
    

def set_output_dir(args):
    if args.output_dir == None:
        args.output_dir = args.models_path
    if args.output_dir == None:
        args.output_dir =  read_path(args.model)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    print('Merge the config file from:',args.config_file)
    
    # Modify for multiple model tests here
    # cfg.MODEL.WEIGHTS = args.model
    # print('Use model weights from:', args.model)
    
    # If args.output_dir is None, set it to model's dir
    set_output_dir(args)
    
    cfg.DATASETS.TRAIN = args.train
    cfg.DATASETS.TEST = args.test
    
    logger.info('Training set:',args.train)
    logger.info('Validation set:',args.test)
    
    mobot_default_setup(cfg,args)
    default_setup(
        cfg, args
    )

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
    cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.DATALOADER.SHUFFLE = True
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    
    cfg.SOLVER.BASE_LR = args.base_ir # pick a good LR 0.00025
    cfg.SOLVER.MAX_ITER = args.max_iter
    # cfg.SOLVER.STEPS = args.steps    
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period # Save the checkpoint each 2000 iterations 
    cfg.TEST.EVAL_PERIOD = args.eval_period # the period to run eval_function. Set to 0 to not evaluate periodically (but still evaluate after the last iteration if eval_after_train is True).                                   


    return cfg


def main(args):
    
    Mobot_Default_Setting()
    # Register Mobot's dataset
    Mobot_Dataset_Register()

    res_dict = OrderedDict()
    cfg = setup(args)

    models,file_name = read_models(args)

    for model in models:
        model_name = read_filename(model)
        test_setup(cfg,model)
    # if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg,model)

        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)

        res_dict[model_name] = res
        torch.cuda.empty_cache()

        np.save(cfg.OUTPUT_DIR+'/'+file_name, res_dict)
    
    plot_test(read_scores(cfg.OUTPUT_DIR+'/'+file_name))

    # return res_list
  


if __name__ == "__main__":
    args = mobot_argument_parser().parse_args()
    print("Command Line Args:", args)

    # Launch multi-gpu or distributed training
    # detectron2.engine.launch
    # world_size = args.num_machines * args.num_gpus
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
