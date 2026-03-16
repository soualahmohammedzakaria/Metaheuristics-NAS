import random
from enum import Enum
from typing import List, Tuple, Optional


class LayerType(Enum):
    CONV = "conv"
    FC = "fc"
    POOLING = "pooling"
    DISABLED = "disabled"


class Layer:
    def __init__(self, layer_type: LayerType, **params):
        self.layer_type = layer_type
        self.params = params

    def __repr__(self):
        return f"Layer({self.layer_type.value}, {self.params})"


MAX_LENGTH = 9
MAX_FC = 3
POPULATION_SIZE = 30


def layer_type_from_bytes(byte0: int, byte1: int) -> LayerType:
    """Determine layer type from first byte."""
    if 0 <= byte0 <= 15:
        return LayerType.CONV
    elif 16 <= byte0 <= 23:
        return LayerType.FC
    elif 24 <= byte0 <= 31:
        return LayerType.POOLING
    elif 32 <= byte0 <= 39:
        return LayerType.DISABLED
    else:
        raise ValueError(f"Invalid byte0: {byte0}")


def decode_layer(byte0: int, byte1: int) -> Layer:
    """Decode 2 bytes into a Layer object."""
    layer_type = layer_type_from_bytes(byte0, byte1)
    combined = (byte0 << 8) | byte1  # 16-bit

    if layer_type == LayerType.CONV:
        fs = ((combined >> 9) & 0x7) + 1  # 1-8
        nfm = ((combined >> 2) & 0x7F) + 1  # 1-128
        stride = (combined & 0x3) + 1  # 1-4
        return Layer(LayerType.CONV, filter_size=fs, num_feature_maps=nfm, stride=stride)

    elif layer_type == LayerType.POOLING:
        ks = ((combined >> 9) & 0x3) + 1
        stride = ((combined >> 7) & 0x3) + 1
        pool_type = "max" if ((combined >> 6) & 0x1) == 0 else "avg"
        return Layer(LayerType.POOLING, kernel_size=ks, stride=stride, pool_type=pool_type)

    elif layer_type == LayerType.FC:
        num_neurons = (combined & 0x7FF) + 1  # 1-2048
        return Layer(LayerType.FC, num_neurons=num_neurons)

    elif layer_type == LayerType.DISABLED:
        return Layer(LayerType.DISABLED)

    else:
        raise ValueError("Unknown layer type")


def encode_layer(layer: Layer) -> Tuple[int, int]:
    """Encode a Layer into 2 bytes."""
    layer_type = layer.layer_type
    if layer_type == LayerType.CONV:
        byte0 = random.randint(0, 15)
        byte1 = random.randint(0, 255)
    elif layer_type == LayerType.FC:
        byte0 = random.randint(16, 23)
        byte1 = random.randint(0, 255)
    elif layer_type == LayerType.POOLING:
        byte0 = random.randint(24, 31)
        byte1 = random.randint(0, 255)
    elif layer_type == LayerType.DISABLED:
        byte0 = random.randint(32, 39)
        byte1 = random.randint(0, 255)
    else:
        raise ValueError("Unknown layer type")
    return byte0, byte1


def random_layer(layer_type: LayerType) -> Tuple[int, int]:
    """Generate random bytes for a layer type."""
    return encode_layer(Layer(layer_type))


def is_valid_for_slot(slot: int, byte0: int, position: Optional[List[int]] = None) -> bool:
    """Check if layer type is valid for slot position."""
    try:
        layer_type = layer_type_from_bytes(byte0, 0)
    except ValueError:
        return False
        
    if slot == 0:
        return layer_type == LayerType.CONV
    elif slot == MAX_LENGTH - 1:
        return layer_type == LayerType.FC
    else:
        if slot <= MAX_LENGTH - MAX_FC - 1:
            if layer_type == LayerType.FC:
                return False

        if position is not None:
            fc_seen = False
            for s in range(1, slot):
                b0 = position[s * 2]
                try:
                    if layer_type_from_bytes(b0, 0) == LayerType.FC:
                        fc_seen = True
                        break
                except ValueError:
                    pass
            
            if fc_seen and layer_type not in [LayerType.FC, LayerType.DISABLED]:
                return False

        return layer_type in [LayerType.CONV, LayerType.POOLING, LayerType.FC, LayerType.DISABLED]



def resample_valid_for_slot(slot: int, position: Optional[List[int]] = None) -> Tuple[int, int]:
    """Resample a random valid layer for the slot."""
    if slot == 0:
        return random_layer(LayerType.CONV)
    elif slot == MAX_LENGTH - 1:
        return random_layer(LayerType.FC)
    else:
        fc_seen = False
        if position is not None:
            for s in range(1, slot):
                b0 = position[s * 2]
                try:
                    if layer_type_from_bytes(b0, 0) == LayerType.FC:
                        fc_seen = True
                        break
                except ValueError:
                    pass
        
        if fc_seen:
            types = [LayerType.FC, LayerType.DISABLED]
        elif slot <= MAX_LENGTH - MAX_FC - 1:
            types = [LayerType.CONV, LayerType.POOLING, LayerType.DISABLED]
        else:
            types = [LayerType.CONV, LayerType.POOLING, LayerType.FC, LayerType.DISABLED]
            
        return random_layer(random.choice(types))