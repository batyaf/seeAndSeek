import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
# Save info for one region of image that is probably a char
@dataclass
class CharInfo:
    bounding_box: Tuple[int, int, int, int]
    image: np.ndarray
    predicted_char: str = ''

# Save info about a area in image that contains char regions and is probably a word
@dataclass
class WordInfo:
    word_id: int
    bounding_box: Tuple[int, int, int, int]
    regions: List[CharInfo]