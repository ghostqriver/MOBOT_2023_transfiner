import argparse
import sys
import os
import warnings
import logging
from collections import OrderedDict
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import DefaultTrainer,default_writers,launch
from detectron2.evaluation import (   
    COCOEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog,build_detection_test_loader,build_detection_train_loader
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.config import get_cfg

from mobot.utils import check_path
from mobot.utils import (read_scores,plot_test,plot)

logger = logging.getLogger("Mobot")


def mobot_default_setup(cfg,args):

    cfg.OUTPUT_DIR=args.output_dir
    check_path(cfg.OUTPUT_DIR)

    cfg.DATALOADER.SHUFFLE = True
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model


def mobot_argument_parser(epilog=None):
    """
    Create a parser with some arguments we will use.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
        Examples:

        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

        Change some config options:
            $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

        Run on multiple machines:
            (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    
    parser.add_argument("--model", default=None, metavar="FILE", help="path to model")

    parser.add_argument("--models-path", default=None, help="path of models to be tested")

    parser.add_argument("--train", default=None, help="training set")
    
    parser.add_argument("--test", default=None, help="test set")

    parser.add_argument("--batch-size", type=int, default=1, help="batch size")

    parser.add_argument("--output-dir", default=None, help="output directory")

    parser.add_argument("--base-ir", type=float,default=0.01, help="base ir")

    parser.add_argument("--max-iter",type=int,default=None, help="max iter")
    
    parser.add_argument("--checkpoint-period",type=int,default=1000, help="checkpoint period")

    parser.add_argument("--eval-period",type=int,default=1000, help="eval period")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    # For prediction
    parser.add_argument(
        "--input",
        nargs="+",
        default = None,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", default = None,help="Path to video file.")
    return parser


class Mobot_DefaultTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        if len(evaluator_list) == 1:
            return evaluator_list[0]
            
        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def test(cls,cfg,model):

        results = OrderedDict()
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
        evaluator = Mobot_DefaultTrainer.build_evaluator(cfg, cfg.DATASETS.TEST, os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST))

        results_i = inference_on_dataset(model, data_loader, evaluator) # Predict
        results[cfg.DATASETS.TEST] = results_i
        
        if comm.is_main_process(): 
            logger.info("Evaluation results for {} in csv format:".format(cfg.DATASETS.TEST))
            print_csv_format(results_i)
        
        if len(results) == 1: # Only one given test set
            results = list(results.values())[0]
        return results

    def train(self,resume):
        model = self.model
        cfg = self.cfg

        optimizer = self.optimizer
        scheduler = self.scheduler
        checkpointer = self.checkpointer
        # start_iter = self.start_iter
        max_iter = self.max_iter
        
        model.train() 
        
        # # Generate lr_list
        # use_scheduler = False
        # if lr_strategy == 1:
        #     lr_list = gen_lr_list_1(cfg, start_iter, min_lr, max_lr)
        # elif lr_strategy == 2:
        #     lr_list = gen_lr_list_2(cfg, start_iter, min_lr, max_lr)
        # elif lr_strategy == 3:
        #     lr_list = gen_lr_list_choosing_lr(cfg, start_iter, min_lr, max_lr)
        # else:
        #     use_scheduler = True
        # if resume:
        #     start_iter = (
        #     checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
        #     )
        # else:
        start_iter = 0

        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

        data_loader = build_detection_train_loader(cfg)
    
        print(data_loader)
        
        logger.info("Starting training from iteration {}".format(start_iter))
    
    # Record the losses and accuracy scores
    # train_loss = np.array([0.,0.,0.,0.,0.]) # 'loss_cls''loss_box_reg''loss_mask''loss_rpn_cls''loss_rpn_loc'
    # train_losses = 0
    # test_acc = OrderedDict()
    # train_loss_list = OrderedDict()
        lrs = []
        res_dict = OrderedDict()
        
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                lrs.append(optimizer.param_groups[0]["lr"])
                storage.iter = iteration
                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    model.eval()
                    with torch.no_grad():
                        res = Mobot_DefaultTrainer.test(cfg,model)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()

                    model_name = 'model_'+str(iteration).zfill(7)
                    res_dict[model_name] = res
                    np.save(cfg.OUTPUT_DIR+'/'+cfg.OUTPUT_DIR+'train_from'+str(start_iter), res_dict)

                    model.train()
                if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)

        plot(lrs)
        # plot_test(read_scores(res_dict))
                # if use_scheduler == False:
                #     optimizer.param_groups[0]["lr"] = lr_list[iteration] 
                
                
            # Show the images in the dataloader, aug or nonaug
    #             print(str(len(data))+" images each batch")
    #             for i in data:
    #                 plt.imshow(c(np.transpose(i['image'],(1,2,0))))
    #                 plt.title(i['file_name'])
    #                 plt.show()
                
    #             loss_dict = model(data)
    #             losses = sum(loss_dict.values())
                
    #             assert True, loss_dict
    # #             assert torch.isfinite(losses).all(), loss_dict
                
    #             loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
    #             losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    #             if comm.is_main_process():
    #                 storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

    #             optimizer.zero_grad()
    #             losses.backward()
    #             optimizer.step()
    #             storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                
    #             lr = optimizer.param_groups[0]["lr"]
                
    #             if use_scheduler == True:
    #                 scheduler.step()

    #             if (
    #                 cfg.TEST.EVAL_PERIOD > 0
    #                 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
    #                 and iteration != max_iter - 1
    #             ):
    #                 test_acc[iteration+1] = do_test(cfg, model)
    #                 test_acc[iteration+1]['lr'] = lr

    #                 # Compared to "train_net.py", the test results are not dumped to EventStorage
    #                 comm.synchronize()
                
    #             train_loss += np.array(list(loss_dict_reduced.values()))
    #             train_losses += losses_reduced
                
    #             if iteration - start_iter > 5 and (
    #                 (iteration + 1) % 20 == 0 or iteration == max_iter - 1
    #             ):
                    
    #                 train_loss_list[iteration+1] = {key:value for key,value in  zip(loss_dict_reduced.keys(),train_loss / 20)}
    #                 train_loss_list[iteration+1]['total_loss'] = train_losses / 20
    #                 train_loss_list[iteration+1]['lr'] = lr

    #                 train_loss = np.array([0.,0.,0.,0.,0.])
    #                 train_losses = 0
                    
    #                 for writer in writers:
    #                     writer.write()
                        
    #             # Save the score list
    #             if (
    #                 cfg.TEST.EVAL_PERIOD > 0
    #                 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
    #                 and iteration != max_iter - 1
    #             ):
    #                 np.save(cfg.OUTPUT_DIR+'/'+file_name+'.npy', {'train_loss':train_loss_list,'test_acc':test_acc}) 
                        
    #             periodic_checkpointer.step(iteration)
                
    #     return {'train_loss':train_loss_list,'test_acc':test_acc}

def Mobot_Dataset_Register():
    
    try:
        register_coco_instances("MOBOT_Train", {}, "Datasets/Train/Train.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Train_denoise", {}, "Datasets/Train/Train_denoise.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Val", {}, "Datasets/Val/Val.json", "Datasets/Val",)   
        
        register_coco_instances("MOBOT_Val_denoise", {}, "Datasets/Val/Val_denoise.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Test", {}, "Datasets/Test/Test.json", "Datasets/Test",)   

        register_coco_instances("MOBOT_Test_denoise", {}, "Datasets/Test/Test_denoise.json", "Datasets/Test",)   


        # dataset only contain end sections
        register_coco_instances("MOBOT_Train_end", {}, "Datasets/Train/Train_end.json", "Datasets/Train",)
        register_coco_instances("MOBOT_Val_end", {}, "Datasets/Val/Val_end.json", "Datasets/Val",)   
        register_coco_instances("MOBOT_Test_end", {}, "Datasets/Test/Test_end.json", "Datasets/Test",)   

        for i in ['Train','Train_denoise','Val','Val_denoise','Test','Test_denoise','Train_end','Val_end','Test_end']:
            MetadataCatalog.get("MOBOT_"+i).set(thing_classes=['end','side'])
            MetadataCatalog.get("MOBOT_"+i).set(thing_colors=[(255, 0, 0),(0, 255, 0)])

    except AssertionError:
        DatasetCatalog.remove("MOBOT_Train")
        MetadataCatalog.remove("MOBOT_Train")
        DatasetCatalog.remove("MOBOT_Train_denoise")
        MetadataCatalog.remove("MOBOT_Train_denoise")
        DatasetCatalog.remove("MOBOT_Val")
        MetadataCatalog.remove("MOBOT_Val")
        DatasetCatalog.remove("MOBOT_Val_denoise")
        MetadataCatalog.remove("MOBOT_Val_denoise")
        DatasetCatalog.remove("MOBOT_Test")
        MetadataCatalog.remove("MOBOT_Test")
        DatasetCatalog.remove("MOBOT_Test_denoise")
        MetadataCatalog.remove("MOBOT_Test_denoise")
        
        DatasetCatalog.remove("MOBOT_Train_end")
        MetadataCatalog.remove("MOBOT_Train_end")
        DatasetCatalog.remove("MOBOT_Val_end")
        MetadataCatalog.remove("MOBOT_Val_end")
        DatasetCatalog.remove("MOBOT_Test_end")
        MetadataCatalog.remove("MOBOT_Test_end")


        register_coco_instances("MOBOT_Train", {}, "Datasets/Train/Train.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Train_denoise", {}, "Datasets/Train/Train_denoise.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Val", {}, "Datasets/Val/Val.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Val_denoise", {}, "Datasets/Val/Val_denoise.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Test", {}, "Datasets/Test/Test.json", "Datasets/Test",)   

        register_coco_instances("MOBOT_Test_denoise", {}, "Datasets/Test/Test_denoise.json", "Datasets/Test",)   

        # dataset only contain end sections
        register_coco_instances("MOBOT_Train_end", {}, "Datasets/Train/Train_end.json", "Datasets/Train",)
        register_coco_instances("MOBOT_Val_end", {}, "Datasets/Val/Val_end.json", "Datasets/Val",)   
        register_coco_instances("MOBOT_Test_end", {}, "Datasets/Test/Test_end.json", "Datasets/Test",)  

        for i in ['Train','Train_denoise','Val','Val_denoise','Test','Test_denoise','Train_end','Val_end','Test_end']:
            MetadataCatalog.get("MOBOT_"+i).set(thing_classes=['end','side'])
            MetadataCatalog.get("MOBOT_"+i).set(thing_colors=[(255, 0, 0),(0, 255, 0)])
 
  
def Mobot_Default_Setting():
    # Close the warning
    warnings.filterwarnings("ignore") 