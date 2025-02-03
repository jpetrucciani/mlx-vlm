import sys
from pathlib import Path

from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Add the package directory to the Python path
package_dir = root_dir / "mlx_vlm"
sys.path.append(str(package_dir))

# Read the requirements from the requirements.txt file
requirements_path = root_dir / "requirements.txt"
with open(requirements_path) as fid:
    requirements = [l.strip() for l in fid.readlines()]

extras_require = {
    "api": [
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "python-multipart>=0.0.20",
        "pydantic>=2.0.0",
    ]
}

# Import the version from the package
from version import __version__

# Setup configuration
setup(
    name="mlx-vlm",
    version=__version__,
    description="Vision LLMs on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="prince.gdt@gmail.com",
    author="Prince Canuma",
    url="https://github.com/Blaizzy/mlx-vlm",
    license="MIT",
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(where=root_dir),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mlx_vlm.convert = mlx_vlm.convert:main",
            "mlx_vlm.generate = mlx_vlm.generate:main",
            "mlx_vlm.api = mlx_vlm.api:main [api]",
        ]
    },
)
