import json
import toml
from pathlib import Path


def test_pyproject_toml_valid():
    """Test that pyproject.toml is valid and has required fields"""
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"
    
    assert pyproject_path.exists(), "pyproject.toml not found"
    
    with open(pyproject_path) as f:
        data = toml.load(f)
    
    # Check required sections
    assert "project" in data, "Missing [project] section"
    assert "build-system" in data, "Missing [build-system] section"
    
    project = data["project"]
    assert "name" in project, "Missing project name"
    assert "version" in project, "Missing project version"
    assert "dependencies" in project, "Missing project dependencies"
    assert "requires-python" in project, "Missing python version requirement"
    
    # Check Python version constraint
    python_req = project["requires-python"]
    assert ">=3.11" in python_req, f"Expected Python >=3.11, got: {python_req}"
    assert "<3.13" in python_req, f"Expected Python <3.13, got: {python_req}"


def test_requirements_txt_valid():
    """Test that requirements.txt exists and is valid"""
    root = Path(__file__).parent.parent
    req_path = root / "requirements.txt"
    
    assert req_path.exists(), "requirements.txt not found"
    
    with open(req_path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    assert len(lines) > 0, "requirements.txt is empty"
    
    # Check for core dependencies
    req_names = [line.split("==")[0] if "==" in line else line.split(">=")[0] for line in lines]
    
    expected_deps = ["Flask", "numpy", "Pillow", "opencv-python", "torch", "tensorflow"]
    for dep in expected_deps:
        assert dep in req_names, f"Missing required dependency: {dep}"


def test_uv_lock_valid():
    """Test that uv.lock exists and is valid"""
    root = Path(__file__).parent.parent
    lock_path = root / "uv.lock"
    
    assert lock_path.exists(), "uv.lock not found"
    
    # Check that it's not empty
    assert lock_path.stat().st_size > 0, "uv.lock is empty"
    
    # Basic validation that it looks like a lock file
    with open(lock_path) as f:
        content = f.read()
    
    # Should contain package information
    assert "[[package]]" in content or "[tool.uv.sources]" in content, "uv.lock doesn't appear to be a valid lock file"


def test_dependencies_consistency():
    """Test that pyproject.toml and requirements.txt dependencies are consistent"""
    root = Path(__file__).parent.parent
    
    # Load pyproject.toml dependencies
    with open(root / "pyproject.toml") as f:
        pyproject_data = toml.load(f)
    
    pyproject_deps = pyproject_data["project"]["dependencies"]
    pyproject_names = set()
    for dep in pyproject_deps:
        name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0]
        pyproject_names.add(name.strip())
    
    # Load requirements.txt dependencies
    with open(root / "requirements.txt") as f:
        req_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    req_names = set()
    for line in req_lines:
        name = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0]
        req_names.add(name.strip())
    
    # Check that requirements.txt contains all pyproject.toml dependencies
    missing_in_req = pyproject_names - req_names
    assert len(missing_in_req) == 0, f"Dependencies in pyproject.toml but not requirements.txt: {missing_in_req}"
    
    # Check that pyproject.toml contains all requirements.txt dependencies  
    missing_in_pyproject = req_names - pyproject_names
    assert len(missing_in_pyproject) == 0, f"Dependencies in requirements.txt but not pyproject.toml: {missing_in_pyproject}"