from enum import Enum

class RNNType(Enum):
    GRU = 1
    LSTM = 2


class DateFrequency(Enum):
    MONTHLY = 1 / 12
    QUARTERLY = 1 / 4
    YEARLY = 1.0
