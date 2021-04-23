import pytest


@pytest.fixture(scope="module")
def image():
    return "IMAGE"
