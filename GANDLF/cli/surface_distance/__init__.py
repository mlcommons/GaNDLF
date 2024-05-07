# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Surface distance module: https://github.com/deepmind/surface-distance ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import lookup_tables  # pylint: disable=relative-beyond-top-level
import numpy as np
from scipy import ndimage

from .metrics import compute_surface_distances
from .metrics import compute_average_surface_distance
from .metrics import compute_robust_hausdorff
from .metrics import compute_surface_overlap_at_tolerance
from .metrics import compute_surface_dice_at_tolerance
from .metrics import compute_dice_coefficient

__version__ = "0.1"
