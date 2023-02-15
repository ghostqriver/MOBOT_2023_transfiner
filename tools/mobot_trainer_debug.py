# import logging
# import os
# from collections import OrderedDict
# import torch
# from torch.nn.parallel import DistributedDataParallel

# import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
# from detectron2.config import get_cfg
# from detectron2.data import (
#     MetadataCatalog,
#     build_detection_test_loader,
#     build_detection_train_loader,
# )
# from detectron2.engine import  default_setup, default_writers, launch,hooks
# from mobot.engine import mobot_argument_parser,Mobot_Dataset_Register,Mobot_Default_Setting
# from mobot.engine import Mobot_DefaultTrainer as Trainer
# from detectron2.evaluation import (
#     CityscapesInstanceEvaluator,
#     CityscapesSemSegEvaluator,
#     COCOEvaluator,
#     COCOPanopticEvaluator,
#     DatasetEvaluators,
#     LVISEvaluator,
#     PascalVOCDetectionEvaluator,
#     SemSegEvaluator,
#     inference_on_dataset,
#     print_csv_format,
#     verify_results
# )
# from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler, build_optimizer
# from detectron2.utils.events import EventStorage
# import warnings

# import logging
# import os
# from collections import OrderedDict
# import torch
# from torch.nn.parallel import DistributedDataParallel

# import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
# from detectron2.config import get_cfg
# from detectron2.data import (
#     MetadataCatalog,
#     build_detection_test_loader,
#     build_detection_train_loader,
# )
# from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
# from detectron2.evaluation import (
#     CityscapesInstanceEvaluator,
#     CityscapesSemSegEvaluator,
#     COCOEvaluator,
#     COCOPanopticEvaluator,
#     DatasetEvaluators,
#     LVISEvaluator,
#     PascalVOCDetectionEvaluator,
#     SemSegEvaluator,
#     inference_on_dataset,
#     print_csv_format,
# )
# from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler, build_optimizer
# from detectron2.utils.events import EventStorage


# logger = logging.getLogger("detectron2")

# def get_evaluator(cfg, dataset_name, output_folder=None):
#     """
#     Create evaluator(s) for a given dataset.
#     This uses the special metadata "evaluator_type" associated with each builtin dataset.
#     For your own dataset, you can simply create an evaluator manually in your
#     script and do not have to worry about the hacky if-else logic here.
#     """
#     if output_folder is None:
#         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#     evaluator_list = []
#     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#     if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
#         evaluator_list.append(
#             SemSegEvaluator(
#                 dataset_name,
#                 distributed=True,
#                 output_dir=output_folder,
#             )
#         )
#     if evaluator_type in ["coco", "coco_panoptic_seg"]:
#         evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
#     if evaluator_type == "coco_panoptic_seg":
#         evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
#     if evaluator_type == "cityscapes_instance":
#         assert (
#             torch.cuda.device_count() > comm.get_rank()
#         ), "CityscapesEvaluator currently do not work with multiple machines."
#         return CityscapesInstanceEvaluator(dataset_name)
#     if evaluator_type == "cityscapes_sem_seg":
#         assert (
#             torch.cuda.device_count() > comm.get_rank()
#         ), "CityscapesEvaluator currently do not work with multiple machines."
#         return CityscapesSemSegEvaluator(dataset_name)
#     if evaluator_type == "pascal_voc":
#         return PascalVOCDetectionEvaluator(dataset_name)
#     if evaluator_type == "lvis":
#         return LVISEvaluator(dataset_name, cfg, True, output_folder)
#     if len(evaluator_list) == 0:
#         raise NotImplementedError(
#             "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
#         )
#     if len(evaluator_list) == 1:
#         return evaluator_list[0]
#     return DatasetEvaluators(evaluator_list)

# # If call by functions
# class ARGS:
#     def __init__(
#             self,config_file=None,model=None,train=None,test=None,batch_size=None,
#             base_ir=None,max_iter=None,checkpoint_period=None,eval_period=None,
#             resume=None,eval_only=None,num_gpus=1,num_machines=1,machine_rank=0,dist_url=None
#             ):
#         self.config_file = config_file
#         self.model = model
#         self.train = train
#         self.test = test
#         self.batch_size=batch_size
#         self.base_ir = base_ir
#         self.max_iter = max_iter
#         self.checkpoint_period = checkpoint_period
#         self.eval_period = eval_period
#         self.resume = resume
#         self.eval_only = eval_only
#         self.num_gpus = num_gpus
#         self.num_machines = num_machines 
#         self.machine_rank = machine_rank
#         self.dist_url = dist_url

# def setup(args):
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     print('Merge the config file from:',args.config_file)
#     cfg.MODEL.WEIGHTS = args.model
#     print('Use model weights from:', args.model)
 
#     default_setup(
#         cfg, args
#     )

#     cfg.DATASETS.TRAIN = [args.train]
#     cfg.DATASETS.TEST = [args.test]
#     logger.info('Training set:',args.train)
#     logger.info('Validation set:',args.test)

#     cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
#     cfg.DATALOADER.NUM_WORKERS = 2 
#     cfg.DATALOADER.SHUFFLE = True
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
#     cfg.OUTPUT_DIR=args.output_dir
#     cfg.SOLVER.BASE_LR = float(args.base_ir) # pick a good LR 0.00025
#     cfg.SOLVER.MAX_ITER = int(args.max_iter)
#     # cfg.SOLVER.STEPS = args.steps    
#     cfg.SOLVER.CHECKPOINT_PERIOD = int(args.checkpoint_period) # Save the checkpoint each 2000 iterations 
#     cfg.TEST.EVAL_PERIOD = int(args.eval_period) # the period to run eval_function. Set to 0 to not evaluate periodically (but still evaluate after the last iteration if eval_after_train is True).                                   
    
#     cfg.DATALOADER.SHUFFLE = True


#     return cfg
# def do_test(cfg, model):
#     results = OrderedDict()
#     for dataset_name in cfg.DATASETS.TEST:
#         data_loader = build_detection_test_loader(cfg, dataset_name)
#         evaluator = get_evaluator(
#             cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#         )
#         results_i = inference_on_dataset(model, data_loader, evaluator)
#         results[dataset_name] = results_i
#         if comm.is_main_process():
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results_i)
#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results


# def do_train(cfg, model, resume=False):
#     model.train()
#     optimizer = build_optimizer(cfg, model)
#     scheduler = build_lr_scheduler(cfg, optimizer)

#     checkpointer = DetectionCheckpointer(
#         model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
#     )
#     start_iter = (
#         checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
#     )
#     max_iter = cfg.SOLVER.MAX_ITER

#     periodic_checkpointer = PeriodicCheckpointer(
#         checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
#     )

#     writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

#     # compared to "train_net.py", we do not support accurate timing and
#     # precise BN here, because they are not trivial to implement in a small training loop
#     data_loader = build_detection_train_loader(cfg)
#     logger.info("Starting training from iteration {}".format(start_iter))
#     with EventStorage(start_iter) as storage:
#         for data, iteration in zip(data_loader, range(start_iter, max_iter)):
#             storage.iter = iteration

#             loss_dict = model(data)
#             losses = sum(loss_dict.values())
#             assert torch.isfinite(losses).all(), loss_dict

#             loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
#             losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#             if comm.is_main_process():
#                 storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()
#             storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
#             scheduler.step()

#             if (
#                 cfg.TEST.EVAL_PERIOD > 0
#                 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
#                 and iteration != max_iter - 1
#             ):
#                 do_test(cfg, model)
#                 # Compared to "train_net.py", the test results are not dumped to EventStorage
#                 comm.synchronize()

#             if iteration - start_iter > 5 and (
#                 (iteration + 1) % 20 == 0 or iteration == max_iter - 1
#             ):
#                 for writer in writers:
#                     writer.write()
#             periodic_checkpointer.step(iteration)

# def main(args):
#     cfg = setup(args)

#     model = build_model(cfg)
#     logger.info("Model:\n{}".format(model))
#     if args.eval_only:
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume
#         )
#         return do_test(cfg, model)

#     distributed = comm.get_world_size() > 1
#     if distributed:
#         model = DistributedDataParallel(
#             model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
#         )

#     do_train(cfg, model, resume=args.resume)
#     return do_test(cfg, model)


# if __name__ == "__main__":
#     args = mobot_argument_parser().parse_args()
#     Mobot_Dataset_Register()
#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )

import glob
