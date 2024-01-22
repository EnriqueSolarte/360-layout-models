from layout_models.utils import load_layout_model
# from geometry_perception_utils.config_utils import get_repo_version
import os

LY_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
LY_MODELS_CFG = os.path.join(LY_MODELS_ROOT, 'config')
LY_MODELS_ASSETS = os.path.join(LY_MODELS_ROOT, 'assets')
os.environ['LY_MODELS_CFG'] = LY_MODELS_CFG
os.environ['LY_MODELS_ASSETS'] = LY_MODELS_ASSETS

# VERSION = get_repo_version(__file__)