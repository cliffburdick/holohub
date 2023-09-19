# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from enum import IntEnum
from attrs import frozen


class encoding_type(IntEnum):
    """Encoding type for data portion of message

    Args:
      Enum (int): Type of encoding
    """
    BINARY = 1
    MSGPACK = 2
    JSON = 3    

@frozen
class queue_problem():
    pass

