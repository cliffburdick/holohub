# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from attrs import frozen, field
from enum import Enum, IntEnum

class external_message(IntEnum):
    MSG_TYPE_PSD = 1
    MSG_TYPE_CONSTELLATION = 2


class message_type(Enum):
    """Internal message types for communicating between threads/processes

    Args:
        Enum (int): Type of message
    """
    EXTERNAL = 1,
    SUBSCRIBE = 2
    UNSUBSCRIBE = 3
    SHUTDOWN = 4


@frozen
class unsubscribe:
    subject: str    
    type: message_type = message_type.UNSUBSCRIBE


@frozen
class subscribe:
    subject: str
    type: message_type = message_type.SUBSCRIBE

@frozen
class disconnect:
    type: message_type = message_type.SHUTDOWN


@frozen
class ext_msg:
    subject: str
    payload: field()
    type: message_type = message_type.EXTERNAL
