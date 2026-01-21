# Contributing to RL-Based Adaptive Video Compression

Thank you for your interest in contributing to this research project. This document provides guidelines for contributions.

## Code of Conduct

- Be respectful and professional in all interactions
- Provide constructive feedback
- Focus on technical merit and research validity

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Check existing issues** to avoid duplicates
2. **Create a detailed issue** with:
   - Clear description of the problem
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, GPU, CUDA version, Python version)
   - Relevant code snippets or error messages

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes** thoroughly:
   ```bash
   python -m pytest tests/
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Brief description of changes"
   ```

5. **Submit a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Test results

## Coding Standards

### Python Style

Follow PEP 8 guidelines:

- **Indentation**: 4 spaces (no tabs)
- **Line length**: Maximum 100 characters
- **Imports**: Group standard library, third-party, and local imports
- **Naming conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes
  ```python
  def calculate_reward(detections, ground_truth, B_value):
      """Calculate reward for RL agent.
      
      Args:
          detections (int): Number of detected signs.
          ground_truth (int): Number of ground truth signs.
          B_value (int): Compression ratio.
          
      Returns:
          float: Calculated reward value.
      """
  ```

- **Comments**: Explain complex logic, not obvious code
- **README updates**: Update documentation for new features

### Code Quality

- **Type hints**: Use type hints for function signatures
  ```python
  def process_frame(frame: np.ndarray, B: int) -> np.ndarray:
  ```

- **Error handling**: Use try-except blocks appropriately
- **Logging**: Use Python logging module, not print statements
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info("Processing complete")
  ```

## Research Contributions

### Experimental Changes

When modifying experiments or methodology:

1. **Document changes** in `docs/` folder
2. **Preserve reproducibility**:
   - Keep original configurations as references
   - Document hyperparameter changes
   - Save experiment results with version tags

3. **Statistical validation**:
   - Run significance tests for performance claims
   - Report mean, std dev, and sample size
   - Include random seed for reproducibility

### Dataset Modifications

- Maintain compatibility with CURE-TSD format
- Document preprocessing steps
- Preserve label integrity

## Testing

### Required Tests

Before submitting:

1. **Unit tests** for new functions
2. **Integration tests** for pipeline changes
3. **Validation** on sample data

### Test Structure

```python
import pytest

def test_sci_compression():
    """Test SCI compression with B=10."""
    # Arrange
    frames = load_sample_frames(count=10)
    compressor = SCICompressor(B=10)
    
    # Act
    measurement = compressor.compress(frames)
    
    # Assert
    assert measurement.shape == frames[0].shape
    assert measurement.dtype == np.uint8
```

## Commit Message Guidelines

Use conventional commit format:

- `Add:` New features or files
- `Fix:` Bug fixes
- `Update:` Modifications to existing features
- `Refactor:` Code restructuring without functionality change
- `Docs:` Documentation updates
- `Test:` Adding or modifying tests
- `Perf:` Performance improvements

Example:
```
Add: Safety-aware reward function for critical signs

- Implements 2x penalty for Stop, Yield, No Entry signs
- Updates reward calculation in video_compression_env.py
- Adds unit tests for critical sign detection
```

## Review Process

1. **Automated checks**: CI/CD pipeline runs tests and linting
2. **Code review**: Maintainer reviews for:
   - Code quality and style
   - Research validity
   - Documentation completeness
   - Test coverage
3. **Feedback**: Address review comments
4. **Merge**: Approved PRs are merged by maintainers

## Questions?

For questions or discussions:
- Open an issue with the "question" label
- Email the maintainer (see README for contact)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to advancing research in adaptive video compression for autonomous driving!
