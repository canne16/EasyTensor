# EasyTensor

EasyTensor is a lightweight, extensible tensor compiler designed to simplify tensor computations and accelerate machine learning workflows. It provides an easy-to-use interface for defining, optimizing, and compiling tensor operations for various backends.

## Features

- Intuitive API for tensor operations
- Automatic graph optimization and compilation
- Support for multiple hardware backends (CPU, GPU, etc.)
- Extensible architecture for custom operators and passes
- Minimal dependencies and fast setup

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/EasyTensor.git
cd EasyTensor
```

Install dependencies (if any):

```bash
# Example for Python projects
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use EasyTensor:

```python
import easytensor as et

# Define tensors
a = et.Tensor([1, 2, 3])
b = et.Tensor([4, 5, 6])

# Perform operations
c = a + b

# Compile and run
result = et.compile_and_run(c)
print(result)  # Output: [5, 7, 9]
```

For more advanced usage and API documentation, see the [docs](docs/) directory.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or documentation improvements.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your fork and submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or support, open an issue or contact the maintainer at [your.email@example.com](mailto:your.email@example.com).
