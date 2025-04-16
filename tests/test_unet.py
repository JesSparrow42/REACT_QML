import pytest
import torch
from GenAI.models.unet import UNetGenerator

# Fixture defining a default input shape.
@pytest.fixture
def input_shape_default():
    # For the default UNetGenerator, in_channels is 1.
    # We'll use a 128x128 image.
    return (1, 128, 128)

# Fixture defining a custom input shape.
@pytest.fixture
def input_shape_custom():
    # For custom testing, e.g. an image with 3 channels
    # We'll use a 256x256 image.
    return (3, 256, 256)

@pytest.fixture
def batch_size():
    return 4

def test_unet_forward_default(input_shape_default, batch_size):
    """
    Test UNetGenerator's forward pass with default parameters.

    - Default parameters: in_channels=1, out_channels=1, base_filters=64.
    - Input: (batch_size, 1, 128, 128)
    - Output: (batch_size, 1, 128, 128) and values in [0, 1] due to the sigmoid.
    """
    # Create a dummy input tensor.
    x = torch.randn(batch_size, *input_shape_default)
    # Instantiate the UNetGenerator with default parameters.
    model = UNetGenerator()
    # Compute the output.
    output = model(x)
    # Check that the output shape is correct.
    expected_shape = (batch_size, 1, input_shape_default[1], input_shape_default[2])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}."
    # Check that the output values are in the expected range.
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output values must be in the range [0, 1]."

def test_unet_forward_custom(input_shape_custom, batch_size):
    """
    Test UNetGenerator's forward pass with custom parameters.

    - Custom parameters: in_channels=3, out_channels=2, base_filters=32.
    - Input: (batch_size, 3, 256, 256)
    - Expected output: (batch_size, 2, 256, 256) with values in [0, 1].
    """
    in_channels = 3
    out_channels = 2
    H, W = input_shape_custom[1], input_shape_custom[2]
    x = torch.randn(batch_size, in_channels, H, W)
    # Instantiate the UNetGenerator with custom parameters.
    model = UNetGenerator(in_channels=in_channels, out_channels=out_channels, base_filters=32)
    output = model(x)
    expected_shape = (batch_size, out_channels, H, W)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}."
    # Ensure that the output values are between 0 and 1.
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output values must be in the range [0, 1]."
