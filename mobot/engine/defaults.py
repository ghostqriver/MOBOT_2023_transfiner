import argparse
import sys
import os
import warnings

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (   
    COCOEvaluator,
    DatasetEvaluators,
)

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog


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
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
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


def Mobot_Dataset_Register():
    
    try:
        register_coco_instances("MOBOT_Train", {}, "Datasets/Train/Train.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Train_denoise", {}, "Datasets/Train/Train_denoise.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Val", {}, "Datasets/Val/Val.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Val_denoise", {}, "Datasets/Val/Val_denoise.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Test", {}, "Datasets/Test/Test.json", "Datasets/Test",)   

        register_coco_instances("MOBOT_Test_denoise", {}, "Datasets/Test/Test_denoise.json", "Datasets/Test",)   


        for i in ['Train','Train_denoise','Val','Val_denoise','Test','Test_denoise']:
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
        
        register_coco_instances("MOBOT_Train", {}, "Datasets/Train/Train.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Train_denoise", {}, "Datasets/Train/Train_denoise.json", "Datasets/Train",)
        
        register_coco_instances("MOBOT_Val", {}, "Datasets/Val/Val.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Val_denoise", {}, "Datasets/Val/Val_denoise.json", "Datasets/Val",)   

        register_coco_instances("MOBOT_Test", {}, "Datasets/Test/Test.json", "Datasets/Test",)   

        register_coco_instances("MOBOT_Test_denoise", {}, "Datasets/Test/Test_denoise.json", "Datasets/Test",)   


        for i in ['Train','Train_denoise','Val','Val_denoise','Test','Test_denoise']:
            MetadataCatalog.get("MOBOT_"+i).set(thing_classes=['end','side'])
            MetadataCatalog.get("MOBOT_"+i).set(thing_colors=[(255, 0, 0),(0, 255, 0)])
 
  
def Mobot_Default_Setting():
    # Close the warning
    warnings.filterwarnings("ignore") 