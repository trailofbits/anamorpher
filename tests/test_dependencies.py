import tomllib
from pathlib import Path


def test_pyproject_toml_valid():
    """Test that pyproject.toml is valid and has required fields"""
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml not found"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

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
    assert "[[package]]" in content, "uv.lock doesn't appear to be a valid lock file"


def test_core_dependencies_present():
    """Test that core dependencies are defined in pyproject.toml"""
    root = Path(__file__).parent.parent

    with open(root / "pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)

    pyproject_deps = pyproject_data["project"]["dependencies"]
    dep_names = []
    for dep in pyproject_deps:
        name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0]
        dep_names.append(name.strip())

    expected_deps = ["Flask", "numpy", "Pillow", "opencv-python", "torch", "tensorflow"]
    for dep in expected_deps:
        assert dep in dep_names, f"Missing required dependency: {dep}"
