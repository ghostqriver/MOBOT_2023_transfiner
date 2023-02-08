from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import importlib

class ARGS():
    # A fake args read by initializing the object
    def __init__(config_file='configs/transfiner/mask_rcnn_R_101_FPN_3x.yaml',opts=['MODEL.WEIGHTS', './pretrained_model/output_3x_transfiner_r101.pth'],confidence_threshold=0.5,):
        self.config_file = config_file
        self.opts = opts
        self.confidence_threshold = confidence_threshold



def setup_cfg(args):
    # load default config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    # Read the model weight
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.VIS_PERIOD = 100
    cfg.freeze()
    return cfg



def build_model_():
    '''
    To read the predictor's structure
    '''
    args = ARGS()
    cfg = setup_cfg()
    return build_model(cfg)


def clear_import(module):
    importlib.reload(module)
