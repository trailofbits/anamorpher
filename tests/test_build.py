import os
import subprocess
import sys
from pathlib import Path


def test_python_version():
    """Test that Python version is supported"""
    version = sys.version_info
    assert version >= (3, 11), f"Python 3.11+ required, got {version.major}.{version.minor}"
    assert version < (3, 13), f"Python <3.13 required, got {version.major}.{version.minor}"


def test_project_structure():
    """Test that required project files exist"""
    root = Path(__file__).parent.parent
    
    required_files = [
        "pyproject.toml",
        "requirements.txt", 
        "uv.lock",
        "backend/app.py",
        "backend/adversarial_generators/bicubic_gen_payload.py",
        "backend/adversarial_generators/bilinear_gen_payload.py", 
        "backend/adversarial_generators/nearest_gen_payload.py",
        "frontend/index.html",
        "frontend/script.js",
        "frontend/styles.css"
    ]
    
    for file_path in required_files:
        assert (root / file_path).exists(), f"Required file missing: {file_path}"


def test_imports():
    """Test that all Python modules can be imported"""
    import sys
    import os
    
    # Add backend to path
    backend_path = Path(__file__).parent.parent / "backend"
    sys.path.insert(0, str(backend_path))
    
    try:
        # Test main modules can be imported
        import app
        from adversarial_generators import bicubic_gen_payload
        from adversarial_generators import bilinear_gen_payload
        from adversarial_generators import nearest_gen_payload
        
        # Test sanitizer if it exists
        if (backend_path / "sanitizer.py").exists():
            import sanitizer
            
    except ImportError as e:
        assert False, f"Import failed: {e}"
    finally:
        sys.path.remove(str(backend_path))


def test_package_installation():
    """Test that core packages can be imported"""
    try:
        import numpy
        import PIL
        import cv2
        import torch
        import tensorflow
        import flask
        import flask_cors
        import bleach
        import markupsafe
    except ImportError as e:
        assert False, f"Required package import failed: {e}"


def test_flask_app_creation():
    """Test that Flask app can be created without errors"""
    import sys
    from pathlib import Path
    
    backend_path = Path(__file__).parent.parent / "backend"
    sys.path.insert(0, str(backend_path))
    
    try:
        import app
        # Test that app variable exists and is a Flask app
        assert hasattr(app, 'app'), "Flask app not found in app.py"
        assert app.app is not None, "Flask app is None"
    except Exception as e:
        assert False, f"Flask app creation failed: {e}"
    finally:
        sys.path.remove(str(backend_path))