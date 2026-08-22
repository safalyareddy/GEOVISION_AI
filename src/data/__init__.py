"""
GeoVision AI Satellite Water Body Segmentation Platform
Data Module Initialization
"""

from src.data.dataset import SatelliteWaterDataset, get_dataloader, load_dataset_config

__all__ = ["SatelliteWaterDataset", "get_dataloader", "load_dataset_config"]
