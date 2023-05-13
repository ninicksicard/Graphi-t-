# Graphi(t)

![Logo](https://github.com/ninicksicard/Graphi-t-/assets/31396919/7cf71cce-077f-4667-a187-062d9bf675aa)
Graphi(t) is a user-friendly software for creating and visualizing 3D parametric plots with ease. It offers a range of features for defining functions, customizing plot settings, and adding multiple curves. Empower your mathematical exploration with Graphi(t)!

## Features

- Intuitive interface for defining parametric functions
- Adjustable time range for precise control over the plotted interval
- Iteration capability for dynamic function evaluation
- Additional variable support for enhanced flexibility
- Save and load functionality to store and retrieve plots

## Installation

Download the MSI installer from the [Releases](https://github.com/ninicksicard/Graphi-t-/releases) page and follow the installation instructions.

## Usage

![image](https://github.com/ninicksicard/Graphi-t-/assets/31396919/3644f5e3-68c0-46ba-94a5-e5b6449bd42a)

1. Launch the application: `python main.py`
2. Define your parametric functions and set the desired plot settings.
3. Click the "Plot" button to generate the 3D parametric plot.
4. Customize the plot by adjusting the time range, enabling iteration, or adding additional variables.
5. Save your plots for later reference or load existing plots from files.

### Basic Operators:

      Addition: +
      Subtraction: -
      Multiplication: *
      Division: /
      Exponentiation: **

### Mathematical Functions:

      Trigonometric functions: cos, sin, tan
      Inverse trigonometric functions: acos, asin, atan, atan2
      Hyperbolic functions: cosh, sinh, tanh
      Inverse hyperbolic functions: acosh, asinh, atanh
      Square root: sqrt
      Exponential: exp
      Logarithm: log (natural logarithm), log10 (base 10), log2 (base 2)
      Absolute value: abs

### Constants:
Pi: pi
Euler's number: e

### Notes : 
- Please note that explicit multiplication is necessary. For example, if you wish to multiply a with the expression (b + cos(t)), you should write it as a * (b + cos(t)), not as a(b + cos(t)).
- atan2 can be used with a coma in any function :  
      cos(t) + sqrt(atan2(a,b))*c

## Development

To setup a development environment:

1. Clone the repository: `git clone https://github.com/your-username/Graphi-t.git`
2. Install the required dependencies: `pip install -r requirements.txt`

These dependencies include:
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter): CustomTkinter library for advanced GUI features and styling
- [matplotlib](https://matplotlib.org/): Matplotlib library for 3D plotting capabilities
- [numpy](https://numpy.org/): NumPy library for array manipulation and mathematical functions
- [cx_Freeze](https://pypi.org/project/cx-Freeze/): A set of scripts and modules for freezing Python scripts into executables

## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Graphi(t) software is built using the [Python](https://www.python.org/) programming language.
- Special thanks to the contributors and open-source community for their valuable contributions.
