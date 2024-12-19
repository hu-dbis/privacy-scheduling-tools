from enum import Enum


class Transformation(Enum):
    PROCESSING_TIMES = 0
    AVAILABILITY = 1
    DELETING_JOBS = 2
    ADDING_JOBS = 3
    SWAPPING_JOBS = 4
    MOVE = 5
    MOVE_PROC = 6
    SWAP_PROC = 7
    ALT_MOVE_PROC = 8
    ALT_MOVE = 9
    SWAP_ALL = 10
    SWAP_ALL_PROC = 11
    SWAP_MACHINE = 12
    SWAP_ORDER = 13
    MOVE_REL = 14
    SWAP_ALL_REL = 15
    SRP = 16
    MRP = 17
