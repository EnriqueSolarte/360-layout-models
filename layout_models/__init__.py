from layout_models.utils import load_layout_model
from geometry_perception_utils.config_utils import get_repo_version
import os

LY_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
LY_MODELS_CFG = os.path.join(LY_MODELS_ROOT, 'config')
os.environ['LY_MODELS_CFG'] = LY_MODELS_CFG

VERSION = get_repo_version(DIR_GEOM_UTILS)