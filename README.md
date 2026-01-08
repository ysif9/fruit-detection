# fruit-detection
Deep learning–based fruit recognition system that classifies images of different fruits using computer vision techniques.

## Setup & Installation

### Prerequisites

- **Python**: 3.10 or higher
- **uv**: Fast Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ysif9/fruit-detection
   cd fruit-detection
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```
3. **Verify Installation**
   ```bash
   uv run main.py
   ```

### Development Setup (PyCharm)

Follow the official [PyCharm uv integration guide](https://www.jetbrains.com/help/pycharm/uv.html) to configure your
IDE.

### Using uv for Project Management

| Task                 | pip                                   | uv                          |
|----------------------|---------------------------------------|-----------------------------|
| Install dependencies | `pip install -r requirements.txt`     | `uv sync`                   |
| Add a package        | `pip install package_name`            | `uv add package_name`       |
| Add dev dependency   | `pip install --save-dev package_name` | `uv add --dev package_name` |
| Freeze dependencies  | `pip freeze > requirements.txt`       | `uv lock`                   |
| Run a script         | `python script.py`                    | `uv run script.py`          |
