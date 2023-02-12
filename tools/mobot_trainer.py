import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import  default_setup, default_writers, launch,hooks
from mobot.engine import mobot_argument_parser
import mobot.engine.Mobot_DefaultTrainer as Trainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage


logger = logging.getLogger("detectron2")

# If call by functions
# class ARGS:
    # def __init__(
#             self, MODEL_DIR, OUTPUT_DIR, BASE_LR, MAX_ITER, 
#             CHECKPOINT_PERIOD, EVAL_PERIOD, STEPS=[],
#             MODEL_CKPT=None, config_file="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
#             ):
#         self.MODEL_DIR,self.OUTPUT_DIR = MODEL_DIR,OUTPUT_DIR
#         self.BASE_LR,self.MAX_ITER = BASE_LR,MAX_ITER
#         self.STEPS = STEPS
#         self.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
#         self.EVAL_PERIOD = EVAL_PERIOD
#         self.MODEL_CKPT = MODEL_CKPT
#         self.BACKBONE = BACKBONE

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.CONFIG_FILE)
    model_path = os.path.join(args.MODEL_DIR, "{}.pth".format(args.MODEL_CKPT)) 
    cfg.MODEL.WEIGHTS = model_path
    print('Use model weights from:', model_path)
 
    default_setup(
        cfg, args
    )

    cfg.DATASETS.TRAIN = args.TRAIN
    cfg.DATASETS.TEST = args.TEST
    print('Training set:',args.TRAIN)
    print('Validation set:',args.TEST)

    cfg.SOLVER.IMS_PER_BATCH = args.BATCH_SIZE 
    
    cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.DATALOADER.SHUFFLE = True
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.OUTPUT_DIR=args.OUTPUT_DIR
    cfg.SOLVER.BASE_LR = args.BASE_LR # pick a good LR 0.00025
    cfg.SOLVER.MAX_ITER = args.MAX_ITER
    cfg.SOLVER.STEPS = args.STEPS    
    cfg.SOLVER.CHECKPOINT_PERIOD = args.CHECKPOINT_PERIOD # Save the checkpoint each 2000 iterations 
    cfg.TEST.EVAL_PERIOD = args.EVAL_PERIOD # the period to run eval_function. Set to 0 to not evaluate periodically (but still evaluate after the last iteration if eval_after_train is True).                                   
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()
  


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
