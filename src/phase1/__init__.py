"""
Phase 1: Data Pipeline & Environment Setup
"""

from .label_parser import CURETSDLabelParser
from .video_loader import VideoLoader
from .feature_extractor import FeatureExtractor

# VideoCompressionEnv requires gymnasium - import only if available
try:
    from .video_compression_env import VideoCompressionEnv
    __all__ = [
        'CURETSDLabelParser',
        'VideoLoader',
        'FeatureExtractor',
        'VideoCompressionEnv'
    ]
except ImportError:
    __all__ = [
        'CURETSDLabelParser',
        'VideoLoader',
        'FeatureExtractor'
    ]
    VideoCompressionEnv = None
