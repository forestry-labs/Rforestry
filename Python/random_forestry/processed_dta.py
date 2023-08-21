from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ProcessedDta:  # pylint: disable=too-many-instance-attributes
    processed_x: pd.DataFrame = field(default_factory=pd.DataFrame)
    y: np.ndarray = field(default_factory=lambda: np.array(0))
    categorical_feature_cols: np.ndarray = field(default_factory=lambda: np.array(0))
    categorical_feature_mapping: List[Dict[Any, Any]] = field(default_factory=list)
    feature_weights: Optional[np.ndarray] = None
    feature_weights_variables: Optional[np.ndarray] = None
    deep_feature_weights: Optional[np.ndarray] = None
    deep_feature_weights_variables: Optional[str] = None
    observation_weights: Optional[str] = None
    monotonic_constraints: Optional[str] = None
    linear_feature_cols: np.ndarray = field(default_factory=lambda: np.array(0))
    group_memberships: np.ndarray = field(default_factory=lambda: np.array(0))
    col_means: np.ndarray = field(default_factory=lambda: np.array(0))
    col_sd: np.ndarray = field(default_factory=lambda: np.array(0))
    has_nas: bool = False
    na_direction: bool = False
    n_observations: int = 0
    num_columns: int = 0
    feat_names: Optional[np.ndarray] = None
