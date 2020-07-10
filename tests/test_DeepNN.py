import pytest

sys.path.insert(0, '../src')
from src.DeepNN import DeepNN

@pytest.fixture
def deep_nn():
    '''Returns a Sales Rep Class instance'''
    return DeepNN()
