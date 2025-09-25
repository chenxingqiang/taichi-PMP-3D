"""
Pytest configuration and fixtures for Two-Phase MPM Solver Test Suite.
Handles Taichi backend initialization and common test utilities.
"""

import pytest
import os
import sys
import numpy as np
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--backend",
        action="store",
        default="cpu",
        choices=["cpu", "cuda", "vulkan", "opengl"],
        help="Taichi backend to use for testing"
    )
    parser.addoption(
        "--debug-mode",
        action="store_true",
        default=False,
        help="Enable debug mode for detailed output"
    )


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Initialize Taichi with selected backend
    if TAICHI_AVAILABLE:
        backend = config.getoption("--backend")
        debug = config.getoption("--debug-mode")
        
        # Map backend strings to Taichi architectures
        backend_map = {
            "cpu": ti.cpu,
            "cuda": ti.gpu,  # Will use CUDA if available
            "vulkan": ti.vulkan,
            "opengl": ti.opengl
        }
        
        if backend in backend_map:
            try:
                ti.init(arch=backend_map[backend], debug=debug)
                print(f"Taichi initialized with {backend} backend")
            except Exception as e:
                print(f"Failed to initialize {backend} backend: {e}")
                # Fallback to CPU
                ti.init(arch=ti.cpu, debug=debug)
                print("Falling back to CPU backend")
        else:
            ti.init(arch=ti.cpu, debug=debug)


@pytest.fixture(scope="session")
def taichi_backend():
    """Provide current Taichi backend information."""
    if TAICHI_AVAILABLE:
        return {
            "available": True,
            "arch": ti.cfg.arch,
            "debug": ti.cfg.debug
        }
    else:
        return {"available": False}


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def random_seed():
    """Provide consistent random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def tolerance_config():
    """Standard tolerance values for numerical comparisons."""
    return {
        "machine_precision": 1e-14,
        "high_precision": 1e-12,
        "medium_precision": 1e-8,
        "physics_tolerance": 1e-6,
        "experimental_tolerance": 0.10  # 10% for experimental validation
    }


# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Skip tests based on available backends and dependencies."""
    if item.get_closest_marker("gpu") and not TAICHI_AVAILABLE:
        pytest.skip("GPU tests require Taichi")
    
    if item.get_closest_marker("slow") and not item.config.getoption("--runslow", default=False):
        pytest.skip("Slow tests skipped (use --runslow to run)")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add backend markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# Test data constants
EXPERIMENTAL_DATA_DIR = Path(__file__).parent / "tests" / "experimental_data"
TEST_FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"

# Ensure test data directories exist
EXPERIMENTAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)