"""
Pico-Body-Fitter: Pure Geometric SMPL Fitting for VR Tracking Data

This package provides tools for fitting SMPL body models to VR tracking data
from Pico devices, with biological constraints and anti-jitter filtering.
"""

from .filters import apply_lowpass_filter, apply_gaussian_smoothing, apply_double_filter
from .utils import (
    load_tracking_data,
    prepare_target_joints,
    export_to_phc_format,
    SMPL_JOINT_NAMES,
    PICO_TO_SMPL_MAPPING
)
from .smpl_solver import fit_smpl_batch

__version__ = "1.0.0"
__all__ = [
    'apply_lowpass_filter',
    'apply_gaussian_smoothing',
    'apply_double_filter',
    'load_tracking_data',
    'prepare_target_joints',
    'export_to_phc_format',
    'fit_smpl_batch',
    'SMPL_JOINT_NAMES',
    'PICO_TO_SMPL_MAPPING'
]
