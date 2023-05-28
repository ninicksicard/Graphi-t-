"""
Graphi(t)

Graphi(t) is a user-friendly software for creating and visualizing 3D parametric plots with ease. It provides an
intuitive interface for defining parametric functions and offers various features such as adjustable time range,
iteration capability, additional variable support, curve color control, and the ability to hide/show curves.
The software allows users to save and load plots, toggle between equal axis and auto resize for the 3D plot,
 and export plots as DWG files for use in CAD software like SolidWorks. It also includes an integrated Matplotlib
 toolbar for additional plot control.

Usage:
1. Launch the application: `python Graphi(t).py`.
2. Define your parametric functions and set the desired plot settings.
3. Click the "Plot" button to generate the 3D parametric plot.
4. Customize the plot by adjusting the time range, enabling iteration, adding additional variables, changing curve
    colors, and toggling visibility of curves.
5. Save your plots for later reference or load existing plots from files.
6. Use the checkbox to toggle between equal axis and auto resize for the 3D plot.
7. Export your plots as DWG files for use in CAD software like SolidWorks.
8. Utilize the integrated Matplotlib toolbar for additional plot control.

Basic Operators:
- Addition: +
- Subtraction: -
- Multiplication: *
- Division: /
- Exponentiation: **

Mathematical Functions:
- Trigonometric functions: cos, sin, tan
- Inverse trigonometric functions: acos, asin, atan, atan2
- Hyperbolic functions: cosh, sinh, tanh
- Inverse hyperbolic functions: acosh, asinh, atanh
- Square root: sqrt
- Exponential: exp
- Logarithm: log (natural logarithm), log10 (base 10), log2 (base 2)
- Absolute value: abs

Constants:
- Pi: pi
- Euler's number: e

Please note that explicit multiplication is necessary. For example, if you wish to multiply 'a' with the
expression '(b + cos(t))', you should write it as 'a * (b + cos(t))', not as 'a(b + cos(t))'.
The 'atan2' function can be used with a comma in any function: 'cos(t) + sqrt(atan2(a, b)) * c'.

For installation instructions, refer to the README file.

License: MIT License
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from functools import partial
from tkinter import filedialog
from icecream import ic

import customtkinter as ctk
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import os  # noqa: F401
import tkinter  # noqa: F401
from tkinter import colorchooser
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from typing import List

ic.includeContext = True


@dataclass
class CurveEntry:
    """
    A class representing a curve entry.

    Attributes:
        curve_name: An instance of UndoableCTkEntry representing the curve name.
        x_entry: An instance of UndoableCTkEntry representing the x entry.
        y_entry: An instance of UndoableCTkEntry representing the y entry.
        z_entry: An instance of UndoableCTkEntry representing the z entry.
        points: A list to store the curve points.
        i: An integer representing the instance count.
        color: A string representing the color of the curve.
        hide: A tkinter.BooleanVar indicating whether the curve is hidden or not.

    Methods:
        get_count(): Returns the total number of instances.
        __post_init__(): Initializes the instance count and assigns an ID.
        select_color(): Opens a color chooser dialog to select the color.
        save_as_dict(): Returns a dictionary containing the curve entry data.
        load_from_dict(data): Loads the curve entry data from a dictionary.
    """
    curve_name: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    x_entry: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    y_entry: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    z_entry: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    points: list = field(default_factory=list)
    i: int = None
    color: None = None
    hide: tkinter.BooleanVar = None

    _count = 0  # Class variable to keep track of the total number of instances

    @classmethod
    def get_count(cls):
        """
       Returns the total number of instances of the CurveEntry class.

       Returns:
           int: The total number of instances.
       """
        return cls._count

    def __post_init__(self):
        """
       Post-initialization method that increments the instance count and assigns an ID if not provided.
       """
        if self.i is None:
            self.__class__._count += 1  # Use __class__ to access the class variable
            self.i = self.__class__._count

    def select_color(self):
        """
            Opens a color chooser dialog to select the color of the curve.
        """
        self.color = colorchooser.askcolor()[1]

    def save_as_dict(self):
        """
        Returns a dictionary containing the curve entry data.

        Returns:
            dict: The curve entry data.
        """
        return {
            'curve_name': self.curve_name.get(),
            'x_entry': self.x_entry.get(),
            'y_entry': self.y_entry.get(),
            'z_entry': self.z_entry.get(),
            'points': self.points,
            'i': self.i,
            'color': self.color,
            'hide': self.hide.get()
        }

    def load_from_dict(self, data):
        """
        Loads the curve entry data from a dictionary.

        Args:
            data (dict): The curve entry data.
        """
        ic(data)
        ic(data.get('curve_name', ""))
        ic(self.curve_name)
        ic(type(self.curve_name))
        self.curve_name.set(data.get('curve_name', ""))

        self.x_entry.set(data.get('x_entry', ""))
        self.y_entry.set(data.get('y_entry', ""))
        self.z_entry.set(data.get('z_entry', ""))
        self.points = data.get('points', [])
        self.i = data.get('i')
        self.color = data.get('color')
        self.hide = tkinter.BooleanVar(value=data.get('hide', False))


@dataclass
class VectorField:
    """
    A class representing a vector field.

    Attributes:
        vector_field_name: An instance of UndoableCTkEntry representing the vector field name.
        vectors_lengths: An instance of UndoableCTkEntry representing the vector lengths.
        vectors_positions: An instance of UndoableCTkEntry representing the vector positions.
        vector_scale: A ctk.CTkSlider representing the scale of the vectors.
        vector_density: An instance of UndoableCTkEntry representing the vector density.
        i: An integer representing the instance count.
        color: A string representing the color of the vectors.
        hide: A tkinter.BooleanVar indicating whether the vectors are hidden or not.

    Methods:
        get_count(): Returns the total number of instances.
        __post_init__(): Initializes the instance count and assigns an ID.
        select_color(): Opens a color chooser dialog to select the color.
        save_as_dict(): Returns a dictionary containing the vector field data.
        load_from_dict(data): Loads the vector field data from a dictionary.
    """
    vector_field_name: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    vectors_lengths: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    vectors_positions: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    vector_scale: ctk.CTkSlider = field(default_factory=ctk.CTkSlider)
    vector_density: "UndoableCTkEntry" = field(default_factory=lambda: UndoableCTkEntry())
    i: int = None
    color: None = None
    hide: tkinter.BooleanVar = None

    _count = 0  # Class variable to keep track of the total number of instances

    @classmethod
    def get_count(cls):
        """
        Returns the total number of instances of the VectorField class.

        Returns:
            int: The total number of instances.
        """
        return cls._count

    def __post_init__(self):
        """
        Post-initialization method that increments the instance count and assigns an ID if not provided.
        """
        if self.i is None:
            self.__class__._count += 1  # Use __class__ to access the class variable
            self.i = self.__class__._count

    def select_color(self):
        """
        Opens a color chooser dialog to select the color of the vectors.
        """
        self.color = colorchooser.askcolor()[1]

    def save_as_dict(self):
        """
        Returns a dictionary containing the vector field data.

        Returns:
            dict: The vector field data.
        """
        ic(self.hide)
        ic(type(self.hide))
        return {
            'vector_field_name': self.vector_field_name.get(),
            'vectors_lengths': self.vectors_lengths.get(),
            'vectors_positions': self.vectors_positions.get(),
            'vector_scale': self.vector_scale.get(),
            'vector_density': self.vector_density.get(),
            'i': self.i,
            'color': self.color,
            'hide': self.hide.get()
        }

    def load_from_dict(self, data):
        """
        Loads the vector field data from a dictionary.

        Args:
            data (dict): The vector field data.
        """
        self.vector_field_name.set(data.get('vector_field_name', ""))
        self.vectors_lengths.set(data.get('vectors_lengths', ""))
        self.vectors_positions.set(data.get('vectors_positions', ""))
        self.vector_scale.set(data.get('vector_scale', ""))
        self.vector_density.set(data.get('vector_density', ""))
        self.i = data.get('i')
        self.color = data.get('color')
        self.hide = tkinter.BooleanVar(value=data.get('hide', False))


@dataclass
class AdditionalVarEntry:
    """
    A class representing an additional variable entry.

    Attributes:
        var_id: An identifier for the variable.
        var_entry: An entry for the variable value.
        func_entry: An entry for the function expression.
        init_entry: An entry for the initial value.
        graph_var: An entry for the variable name in the graph.
        i: An integer representing the instance count.

    Methods:
        get_count(): Returns the total number of instances.
        __post_init__(): Initializes the instance count and assigns an ID.
        save_as_dict(): Returns a dictionary containing the additional variable entry data.
        load_from_dict(data): Loads the additional variable entry data from a dictionary.
    """
    var_id: any
    var_entry: any
    func_entry: any
    init_entry: any
    graph_var: any
    i: int = None

    _count = 0  # Class variable to keep track of the total number of instances

    @classmethod
    def get_count(cls):
        """
        Returns the total number of instances of the AdditionalVarEntry class.

        Returns:
            int: The total number of instances.
        """
        return cls._count

    def __post_init__(self):
        """
        Post-initialization method that increments the instance count and assigns an ID if not provided.
        """
        if self.i is None:
            self.__class__._count += 1
            self.i = self.__class__._count

    def save_as_dict(self):
        """
        Returns a dictionary containing the additional variable entry data.

        Returns:
            dict: The additional variable entry data.
        """
        return {
            'var_entry': self.var_entry.get(),
            'func_entry': self.func_entry.get(),
            'init_entry': self.init_entry.get(),
            'graph_var': self.graph_var.get(),
            'i': self.i
        }

    def load_from_dict(self, data):
        """
        Loads the additional variable entry data from a dictionary.

        Args:
            data (dict): The additional variable entry data.
        """
        self.var_entry.set(data.get('var_entry', ""))
        self.func_entry.set(data.get('func_entry', ""))
        self.init_entry.set(data.get('init_entry', ""))
        self.graph_var.set(data.get('graph_var', ""))
        self.i = data.get('i')


@dataclass
class Workspace:
    """
    A class representing the workspace.

    Attributes:
        parent_app: The parent application.
        t_min: A string representing the minimum time range.
        t_max: A string representing the maximum time range.
        allow_iterations: A tkinter.BooleanVar indicating whether iterations are allowed or not.
        iteration_number: A string representing the number of iterations.
        curve_entries: A list of CurveEntry instances.
        additional_vars_entries: A list of AdditionalVarEntry instances.
        vector_field_entries: A list of VectorField instances.

    Methods:
        toggle_iteration(): Toggles the allow_iterations attribute.
        save_as_dict(): Returns a dictionary containing the workspace data.
        load_from_dict(data): Loads the workspace data from a dictionary.
    """
    parent_app: any
    t_min: str = ""
    t_max: str = ""
    allow_iterations: tkinter.BooleanVar = None
    iteration_number: str = ""
    curve_entries: List[CurveEntry] = field(default_factory=list)
    additional_vars_entries: List[AdditionalVarEntry] = field(default_factory=list)
    vector_field_entries: List[VectorField] = field(default_factory=list)

    def toggle_iteration(self):
        """
        Toggles the allow_iterations attribute.
        """
        self.allow_iterations.set(not self.allow_iterations.get())

    def save_as_dict(self):
        """
        Returns a dictionary containing the workspace data.

        Returns:
            dict: The workspace data.
        """
        return {
            't_min': self.parent_app.t_min_entry.get(),
            't_max': self.parent_app.t_max_entry.get(),
            'allow_iterations': self.allow_iterations.get(),
            'iteration_number': self.parent_app.iterations_entry.get(),
            'curve_entries': [entry.save_as_dict() for entry in self.curve_entries],
            'additional_vars_entries': [entry.save_as_dict() for entry in self.additional_vars_entries],
            'vector_field_entries': [entry.save_as_dict() for entry in self.vector_field_entries],
        }

    def load_from_dict(self, data):
        """
        Loads the workspace data from a dictionary.

        Args:
            data (dict): The workspace data.
        """
        self.parent_app.t_min_entry.set(data.get('t_min', ""))
        self.parent_app.t_max_entry.set(data.get('t_max', ""))
        self.allow_iterations = tkinter.BooleanVar(value=data.get('allow_iterations', True))
        self.parent_app.iterations_entry.set(data.get('iteration_number', 100))

        for entry_data in data.get('curve_entries', []):
            self.parent_app.add_curve_entries()
            self.curve_entries[-1].load_from_dict(entry_data)

        for entry_data in data.get('additional_vars_entries', []):
            self.parent_app.add_additional_var_entries()
            self.additional_vars_entries[- 1].load_from_dict(entry_data)

        for entry_data in data.get('vector_field_entries', []):
            self.parent_app.add_vector_field_entries()
            self.vector_field_entries[- 1].load_from_dict(entry_data)


class UndoableCTkEntry(ctk.CTkEntry):
    """
    A custom Tkinter entry widget that supports undo and redo functionality.

    Methods:
        __init__(self, master=None, **kwargs): Initializes the UndoableCTkEntry.
        _on_key(self, _event): Handles the key event and updates the undo stack.
        _on_undo(self, _event): Handles the undo event and performs the undo operation.
        _on_redo(self, _event): Handles the redo event and performs the redo operation.
        get_(self, *args): Returns the current value of the entry.
        __getattr__(self, item): Handles the get attribute operation.
        set(self, value): Sets the value of the entry.
        __str__(self): Returns the string representation of the entry.
        __repr__(self): Returns the string representation of the entry.
    """

    def __init__(self, master=None, **kwargs):
        """
        Initializes the UndoableCTkEntry.

        Args:
            master: The master widget.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(master, **kwargs)
        self._undo_stack = []
        self._redo_stack = []
        self.bind('<Key>', self._on_key)  # Listen for key press
        self.bind('<Control-z>', self._on_undo)  # Listen for undo command
        self.bind('<Control-y>', self._on_redo)  # Listen for redo command

    def _on_key(self, _event):
        """
        Handles the key event and updates the undo stack.
        """
        self._undo_stack.append(self.get())  # Add current state to undo stack
        self._redo_stack = []  # Clear redo stack when a new key is pressed

    def _on_undo(self, _event):
        """
        Handles the undo event and performs the undo operation.
        """
        if self._undo_stack:
            last_state = self._undo_stack.pop()  # Remove last state from undo stack
            self._redo_stack.append(last_state)  # Push it to the redo stack
            self.delete(0, 'end')  # Clear the entry
            if self._undo_stack:
                self.insert(0, self._undo_stack[-1])  # Insert the last state

    def _on_redo(self, _event):
        """
        Handles the redo event and performs the redo operation.
        """
        if self._redo_stack:
            self.delete(0, 'end')  # Clear the entry
            self.insert(0, self._redo_stack[-1])  # Insert the last state from the redo stack
            self._undo_stack.append(self._redo_stack.pop())  # Push it back to the undo stack

    def get_(self, *_args):
        """
        Returns the current value of the entry.

        Args:
            *_args: Additional arguments.

        Returns:
            The current value of the entry.
        """
        return self.get()

    def __getattr__(self, item):
        """
        Handles the get attribute operation.

        Args:
            item: The item to get the attribute for.
        """
        return

    def set(self, value):
        """
        Sets the value of the entry.

        Args:
            value: The value to be set.
        """
        self.delete(0, 'end')
        self.insert(0, value)

    def __str__(self):
        """
        Returns the string representation of the entry.

        Returns:
            str: The string representation of the entry.
        """
        return self.get()

    def __repr__(self):
        """
        Returns the string representation of the entry.

        Returns:
            str: The string representation of the entry.
        """
        return self.get()


class GraphingApplication(ctk.CTk):
    """
    A graphical application for creating and visualizing 3D parametric plots.

    Methods:
        __init__(self): Initializes the GraphingApplication.
        add_curve_entries(self): Adds curve entries to the application.
        add_vector_field_entries(self): Adds vector field entries to the application.
        add_additional_var_entries(self): Adds additional variable entries to the application.
        on_plot_button_click(self, *_args): Handles the plot button click event.
        on_save_button_click(self, *_args): Handles the save button click event.
        on_export_button_click(self, *_args): Handles the export button click event.
        on_load_button_click(self, *_args): Handles the load button click event.
        create_additional_vars_frame(self, frame_): Creates the additional variables frame.
        create_labeled_entry(self, frame, text, default_value): Creates a labeled entry widget.
        create_additional_figure_and_canvas(self, parent_): Creates an additional figure and canvas for plotting.
        create_bordered_frame(self, parent=None): Creates a bordered frame.
        create_frame_grid(self, parent, rows, columns, main_row=1, main_column=1): Creates a grid of frames.
        create_t_range_entry(self, frame): Creates the t range entry.
        toggle_iteration(self): Toggles the iteration functionality.
    """

    def __init__(self):
        """
        Initializes the GraphingApplication.
        """
        super().__init__()
        # Create a new DXF document
        self.doc = ezdxf.new('R2010')

        self.workspace = Workspace(self)

        self.msp = None
        self.curve_entries = self.workspace.curve_entries
        self.additional_vars_entries = self.workspace.additional_vars_entries
        self.vector_fields = self.workspace.vector_field_entries
        self.additional_plots = {}
        self.entries = []

        self.container = self.create_bordered_frame()
        self.container.pack(side='bottom', fill='both', expand=True)

        self.frame_grid = self.create_frame_grid(self.container, 3, 3)

        # T range
        self.t_min_entry, self.t_max_entry = self.create_t_range_entry(self.frame_grid[0][1])

        # Add curve button
        self.add_curve_button = ctk.CTkButton(self.frame_grid[0][0], text="+ Curve", command=self.add_curve_entries)
        self.add_curve_button.pack(side='top', padx=5)

        # Create initial x, y, and z entries
        self.add_curve_entries()

        # Add vector field button
        self.add_vector_field_button = ctk.CTkButton(self.frame_grid[0][0], text="+ vector field",
                                                     command=self.add_vector_field_entries)
        self.add_vector_field_button.pack(side='top', padx=5)

        # Add variable button
        self.add_var_button = ctk.CTkButton(self.frame_grid[0][2], text="+ Variable",
                                            command=self.add_additional_var_entries)
        self.add_var_button.pack(side=ctk.LEFT, padx=5)

        # Settings
        self.workspace.allow_iterations = ctk.BooleanVar(value=True)
        self.allow_iterations_checkbox = ctk.CTkCheckBox(
            self.frame_grid[2][0],
            text="Allow iteration",
            variable=self.workspace.allow_iterations
        )

        self.allow_iterations_checkbox.pack(side='top', pady=2)

        self.iterations_label, self.iterations_entry, _ = self.create_labeled_entry(self.frame_grid[2][0],
                                                                                    "Iterations:",
                                                                                    "100")

        # Add Checkbox for auto size vs equal axis
        self.auto_size_vs_equal_axis = ctk.BooleanVar(value=True)
        self.auto_size_vs_equal_axis_checkbox = ctk.CTkCheckBox(self.frame_grid[2][1], text="Equal Axis",
                                                                variable=self.auto_size_vs_equal_axis)
        self.auto_size_vs_equal_axis_checkbox.pack(side='top', pady=2)

        # Plot button
        self.plot_button = ctk.CTkButton(self.frame_grid[2][1], text="Plot", command=self.on_plot_button_click)
        self.plot_button.pack(side='top', pady=2, fill='none')

        # Save button
        self.save_button = ctk.CTkButton(self.frame_grid[2][2], text="Save", command=self.on_save_button_click)
        self.save_button.pack(side='top', pady=2, fill='none')

        # Export dwg
        self.export_button = ctk.CTkButton(self.frame_grid[2][2], text="Export", command=self.on_export_button_click)
        self.export_button.pack(side='top', pady=2, fill='none')

        # Load button
        self.load_button = ctk.CTkButton(self.frame_grid[2][2], text="Load", command=self.on_load_button_click)
        self.load_button.pack(side='top', pady=2, fill='none')

        # Canvas for the 3D plot
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.autoscale(enable=True)

        self.canvas = FigureCanvasTkAgg(self.figure, self.frame_grid[1][1])

        # add this where you create your canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_grid[1][1])
        self.toolbar.pack(expand=True, side="top")
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(expand=True, fill='both', side='top')

        # Initialize additional_vars_frame
        self.create_additional_vars_frame(self.frame_grid[1][2])

        # Initialization of class properties

        self.doc = ezdxf.new('R2010')
        self.temp_dxf_path = ''

        # Initialize the evaluation environment with numpy mathematical functions and constants
        self.eval_globals = {
            'cos': np.cos,
            'sin': np.sin,
            'tan': np.tan,
            'acos': np.arccos,
            'asin': np.arcsin,
            'atan': np.arctan,
            'atan2': np.arctan2,
            'cosh': np.cosh,
            'sinh': np.sinh,
            'tanh': np.tanh,
            'acosh': np.arccosh,
            'asinh': np.arcsinh,
            'atanh': np.arctanh,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'log2': np.log2,
            'pi': np.pi,
            'e': np.e,
            'abs': np.abs,
            'diff': np.diff
        }
        self.t_range = None

    def evaluate_entry_func(self, entry):
        """
        Evaluates the function specified in the entry and returns the result.

        Args:
           entry: The entry containing the function to evaluate.

        Returns:
           The evaluated function as a NumPy array.
        """
        for i in range(0, 5):
            try:
                evaluation = np.broadcast_to(eval(entry.get(), self.eval_globals), (self.t_range.shape[0] - i,))
            except ValueError:
                continue
            return evaluation

    def process_plot_var_entry(self, additional_var_entry):
        """
        Processes a variable entry for plotting.

        Args:
            additional_var_entry: The additional variable entry to process.

        Returns:
            True if the variable entry was processed successfully, False otherwise.
        """
        # Function to process a variable entry for plotting
        if additional_var_entry.var_entry.get():
            plot_info = self.additional_plots[additional_var_entry.var_id]
            y = self.evaluate_entry_func(additional_var_entry.func_entry)
            plot_info['ax'].clear()
            plot_info['ax'].plot(self.t_range, y)
            if additional_var_entry.graph_var.get():
                plot_info['fig_container'].grid(column=4, row=(plot_info['var_row']))
                plot_info['canvas'].draw()
                return False
            else:
                plot_info['fig_container'].grid_forget()
        return True

    def process_var_entry(self, additional_var_entry):
        """
        Processes a variable entry.

        Args:
            additional_var_entry: The additional variable entry to process.
        """
        # Function to process a variable entry
        var_name = additional_var_entry.var_entry.get()
        if var_name:
            self.eval_globals[var_name] = int(additional_var_entry.init_entry.get())

    def update_variable(self, additional_var_entry):
        """
        Performs an update for one variable.

        Args:
            additional_var_entry: The additional variable entry to update.
        """
        # Function to perform an update for one variable
        var_name = additional_var_entry.var_entry.get()
        if var_name:
            self.eval_globals[var_name] = eval(additional_var_entry.func_entry.get(), self.eval_globals)

    def evaluate_and_plot_curve(self, curve_entry, msp):
        """
        Evaluates the points for a curve and plots it.

        Args:
            curve_entry: The CurveEntry object containing the curve information.
            msp: The model space of the DXF document.
        """

        # Function to evaluate the points for a curve
        x = self.evaluate_entry_func(curve_entry.x_entry)
        y = self.evaluate_entry_func(curve_entry.y_entry)
        z = self.evaluate_entry_func(curve_entry.z_entry)

        if not curve_entry.hide.get():
            length = min(len(x), len(y), len(z))
            self.ax.plot(x[0:length], y[0:length], z[0:length], color=curve_entry.color)
            ic(curve_entry.hide.get())
        points = list(zip(x, y, z))
        msp.add_polyline3d(points)

    def evaluate_and_plot_vector_field(self, vector_field):
        """
        Evaluates and plots the vector field.

        Args:
            vector_field: The VectorField object containing the vector field information.
        """
        for lengths_curve in self.curve_entries:
            if lengths_curve.curve_name.get() == vector_field.vectors_lengths.get():
                for positions_curve in self.curve_entries:
                    if positions_curve.curve_name.get() == vector_field.vectors_positions.get():
                        step_value = 100 - round(eval(vector_field.vector_density.get()))  # Plot every 10th vector

                        vector_scale = np.exp(vector_field.vector_scale.get())
                        # Function to evaluate the points for a curve
                        x_root = self.evaluate_entry_func(positions_curve.x_entry)[::step_value]
                        y_root = self.evaluate_entry_func(positions_curve.y_entry)[::step_value]
                        z_root = self.evaluate_entry_func(positions_curve.z_entry)[::step_value]

                        x_tip = self.evaluate_entry_func(lengths_curve.x_entry)[::step_value] * vector_scale
                        y_tip = self.evaluate_entry_func(lengths_curve.y_entry)[::step_value] * vector_scale
                        z_tip = self.evaluate_entry_func(lengths_curve.z_entry)[::step_value] * vector_scale

                        length = min(len(x_tip), len(y_tip), len(z_tip), len(x_root), len(y_root), len(z_root))

                        if not vector_field.hide.get():
                            self.ax.quiver(x_root[0:length], y_root[0:length], z_root[0:length], x_tip[0:length],
                                           y_tip[0:length], z_tip[0:length], color=vector_field.color)

    # Rest of your methods go here, for example:
    def create_labeled_entry(self, frame, text, default_value):
        """
        Creates a labeled entry widget.

        Args:
            frame: The frame to place the labeled entry widget.
            text: The text to display as the label.
            default_value: The default value for the entry widget.

        Returns:
            A tuple containing the label, entry widget, and container frame.
        """
        container_ = self.create_bordered_frame(frame)
        container_.pack(side='top', padx=5, pady=2)
        label = ctk.CTkLabel(container_, text=text)
        label.pack(side='left', pady=2, padx=5)

        entry = UndoableCTkEntry(container_)
        entry.insert(0, default_value)
        entry.pack(side='left', pady=2, padx=5)
        return label, entry, container_

    def create_additional_figure_and_canvas(self, parent_):
        """
        Creates an additional figure and canvas for plotting.

        Args:
            parent_: The parent widget to place the figure and canvas.

        Returns:
            A tuple containing the figure, axis, and canvas.
        """

        fig___ = plt.figure(figsize=(2, 1))  # Change the 3, 3 to whatever size you want in inches
        ax___ = fig___.add_subplot(111)
        font_size = 6
        ax___.set_xlabel('X', fontsize=font_size)
        ax___.set_ylabel('Y', fontsize=font_size)
        for label in (ax___.get_xticklabels() + ax___.get_yticklabels()):
            label.set_fontsize(font_size)
        canvas___ = FigureCanvasTkAgg(fig___, master=parent_)
        canvas___.get_tk_widget().grid(row=len(self.additional_vars_entries) + 7, column=4)
        return fig___, ax___, canvas___

    def create_bordered_frame(self, parent=None):
        """
        Creates a bordered frame.

        Args:
            parent: The parent widget to place the frame.

        Returns:
            The created frame.
        """

        if parent:
            frame = ctk.CTkFrame(parent, bg_color='transparent', height=0, width=0)
        else:
            frame = ctk.CTkFrame(self, bg_color='transparent', height=0, width=0)
        return frame

    def create_frame_grid(self, parent, rows, columns, main_row=1, main_column=1):
        """
        Creates a grid of frames. This is the main layout of the application.

        Args:
            parent: The parent widget to place the frames.
            rows: The number of rows in the grid.
            columns: The number of columns in the grid.
            main_row: The row index for the main frame.
            main_column: The column index for the main frame.

        Returns:
            A list of lists containing the frames in the grid.
        """
        frames = []
        for row in range(rows):
            if row == main_row:
                parent.grid_rowconfigure(row, weight=1)
            else:
                parent.grid_rowconfigure(row, weight=0, minsize=2)
            frame_row = []
            for column in range(columns):
                if column == main_column:
                    parent.grid_columnconfigure(column, weight=1)
                elif column == 2:
                    parent.grid_columnconfigure(column, weight=0, minsize=350)
                else:
                    parent.grid_columnconfigure(column, weight=0, minsize=2)
                if (column == 2 or column == 0) and row == 1:
                    frame = ctk.CTkScrollableFrame(parent)
                    frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
                else:
                    frame = self.create_bordered_frame(parent)
                    frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")

                frame_row.append(frame)
            frames.append(frame_row)
        return frames

    def create_t_range_entry(self, frame):
        """
        Creates the t range entry.

        Args:
            frame: The frame to place the t range entry.

        Returns:
            A tuple containing the t min entry and t max entry.
        """
        # T range
        t_container = self.create_bordered_frame(frame)
        t_container.pack()
        t_min_entry_ = UndoableCTkEntry(t_container, width=40)
        t_min_entry_.insert(0, '0')
        t_min_entry_.pack(side='left', padx=5, expand=True, )

        t_label = ctk.CTkLabel(t_container, text=" < t < ")
        t_label.pack(side='left', padx=5)

        t_max_entry_ = UndoableCTkEntry(t_container, width=40)
        t_max_entry_.insert(0, '10')
        t_max_entry_.pack(side='left', padx=5, expand=True)
        return t_min_entry_, t_max_entry_

    def create_additional_vars_frame(self, frame_):
        """
        Creates the additional variables frame.

        Args:
            frame_: The frame to place the additional variables frame.
        """
        var_label = ctk.CTkLabel(frame_, text="Variable")
        var_label.grid(column=0, row=len(self.additional_vars_entries) + 6, padx=2, pady=5)
        func_label = ctk.CTkLabel(frame_, text="Function")
        func_label.grid(column=1, row=len(self.additional_vars_entries) + 6, padx=2, pady=5)
        initial_value_label = ctk.CTkLabel(frame_, text="Initial value")
        initial_value_label.grid(column=2, row=len(self.additional_vars_entries) + 6, padx=2, pady=5)

    def right_click_curve_entries(self, event, curve_entry):
        """
        Handles the right-click event on curve entries.

        Args:
            event: The right-click event.
            curve_entry: The CurveEntry object associated with the right-clicked entry.
        """

        popup = tkinter.Menu(self, tearoff=0)
        popup.add_command(label="color", command=curve_entry.select_color)
        popup.add_checkbutton(label="Hide curve", variable=curve_entry.hide)

        # Display the menu
        popup.tk_popup(event.x_root, event.y_root)

    def add_curve_entries(self):
        """
        Adds curve entries to the application.
        """
        container__ = self.create_bordered_frame(self.frame_grid[1][0])
        container__.pack(side=ctk.TOP, padx=5, pady=5)
        curve_name_label, curve_name_entry, _ = self.create_labeled_entry(container__, "Curve name:",
                                                                          str(CurveEntry.get_count()))

        x_label, x_entry, x_container = self.create_labeled_entry(container__, "x(t):", "1")
        y_label, y_entry, y_container = self.create_labeled_entry(container__, "y(t):", "t")
        z_label, z_entry, z_container = self.create_labeled_entry(container__, "z(t):", "0")

        curve_entry = CurveEntry(curve_name_entry, x_entry, y_entry, z_entry)
        curve_entry.hide = tkinter.BooleanVar(value=False)

        clickable = [container__, curve_name_label, x_label, y_label, z_label, x_container, y_container, z_container]
        for item in clickable:
            item.bind("<Button-3>", partial(self.right_click_curve_entries, curve_entry=curve_entry))

        self.workspace.curve_entries.append(curve_entry)

    def add_vector_field_entries(self):
        """
        Adds vector field entries to the application.
        """
        container__ = self.create_bordered_frame(self.frame_grid[1][0])

        container__.pack(side=ctk.TOP, padx=5, pady=5)
        vector_field_name_label, vector_field_name_entry, _ = self.create_labeled_entry(container__,
                                                                                        "Vector field name:",
                                                                                        str(VectorField.get_count()))

        length_label, vectors_lengths, length_container = self.create_labeled_entry(container__, "vectors lengths :",
                                                                                    "write a curve name here")
        position_label, vectors_positions, position_container = self.create_labeled_entry(container__,
                                                                                          "vectors positions : ",
                                                                                          "write a curve name here")

        # Create slider 1
        container_vs = self.create_bordered_frame(container__)
        container_vs.pack(side='top', padx=5, pady=2)
        vector_scale_slider_label = ctk.CTkLabel(container_vs, text="scale:")
        vector_scale_slider_label.pack(side='left', pady=2, padx=5)
        vector_scale_slider = ctk.CTkSlider(container_vs, from_=0, to=10)
        vector_scale_slider.set(1)
        vector_scale_slider.pack(side=ctk.TOP, padx=5, pady=5)

        # Create slider 2
        container_vd = self.create_bordered_frame(container__)
        container_vd.pack(side='top', padx=5, pady=2)
        vector_density_slider_label = ctk.CTkLabel(container_vd, text="density:")
        vector_density_slider_label.pack(side='left', pady=2, padx=5)
        vector_density_slider = UndoableCTkEntry(container_vd)
        vector_density_slider.insert(0, 30)
        vector_density_slider.pack(side=ctk.TOP, padx=5, pady=5)

        vector_field = VectorField(vector_field_name_entry, vectors_lengths, vectors_positions, vector_scale_slider,
                                   vector_density_slider)
        vector_field.hide = tkinter.BooleanVar(value=False)
        clickable = [length_label, length_container, position_label, position_container, container__,
                     container_vd, vector_density_slider_label, container_vs, vector_scale_slider_label,
                     vector_field_name_label]
        for item in clickable:
            item.bind("<Button-3>", partial(self.right_click_curve_entries, curve_entry=vector_field))

        self.workspace.vector_field_entries.append(vector_field)

    def on_plot_button_click(self, *_args):
        """
        Handles the click event on the Plot button.

        Performs the following steps:
        A) Defines the range of 't' from user's input.
        B) Stores the t values to be accessible within the global evaluation environment.
        C) Processes all additional variable entries.
        D) If iterations are allowed, performs variable updates over several iterations.
        E) Initializes a flag to check if any graph is displayed.
        F) Processes all additional variable entries for plotting.
        G) Adjusts the grid size based on whether any graph is displayed.
        H) Clears the 3D plot and creates a new DXF document and 3D polyline in the model space.
        I) Evaluates and plots the curves and vector fields.
        J) Saves the DXF document.
        H) Adjusts the axis of the 3D plot based on the user's preference.
        """
        # A) Define the range of 't' from user's input
        t_start = float(self.t_min_entry.get())
        t_end = float(self.t_max_entry.get())
        self.t_range = np.linspace(t_start, t_end, 1000)

        # B) Store the t values to be accessible within the global evaluation environment
        self.eval_globals['t'] = self.t_range

        # C) Process all additional variable entries
        for additional_var_entry in self.additional_vars_entries:
            self.process_var_entry(additional_var_entry)

        # D) If iterations are allowed, perform variable updates over several iterations
        if self.workspace.allow_iterations.get():
            iteration_count = int(self.iterations_entry.get())
            for _ in range(iteration_count):
                for additional_var_entry in self.additional_vars_entries:
                    self.update_variable(additional_var_entry)

        # E) Initialize flag to check if any graph is displayed
        no_graph = True

        # F) Process all additional variable entries for plotting
        for additional_var_entry in self.additional_vars_entries:
            no_graph = min(self.process_plot_var_entry(additional_var_entry), no_graph)

        # G) Adjust the grid size based on whether any graph is displayed
        if no_graph:
            self.container.grid_columnconfigure(2, weight=0, minsize=350)
        else:
            self.container.grid_columnconfigure(2, weight=0, minsize=550)

        self.ax.clear()  # Clear the 3D plot

        # H) Create a new DXF document and 3D polyline in the model space
        self.doc = ezdxf.new('R2010')
        self.msp = self.doc.modelspace()

        # I)  For each curve entry, evaluate the functions and plot the curves
        for curve_entry in self.curve_entries:
            self.evaluate_and_plot_curve(curve_entry, self.msp)

        for vector_field in self.vector_fields:
            self.evaluate_and_plot_vector_field(vector_field)

        # J) Save the DXF document
        self.doc.saveas(temp_dxf_path)

        # H) Adjust the axis of the 3D plot based on user's preference
        if self.auto_size_vs_equal_axis.get():
            self.ax.axis('equal')
        else:
            self.ax.autoscale(enable=True)

        # Set the labels of the 3D plot
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Redraw the canvas to reflect all updates
        self.canvas.draw()

    def on_export_button_click(self, *_args):
        """
        Handles the click event on the Export button.

        Opens the file dialog to get the path where the user wants to save the DWG file.
        If a valid file path is selected, loads the DXF document from the temporary file and saves it as a DWG file.
        """
        # Open the file dialog to get the path where the user wants to save the DWG file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".dwg",
            filetypes=[("DWG files", "*.dwg")])
        if file_path:
            # Load the DXF document from the temporary file
            self.doc = ezdxf.readfile(temp_dxf_path)

            # Save the DXF document as a DWG file at the selected path
            self.doc.saveas(file_path)

    def on_save_button_click(self, *_args):
        """
       Handles the click event on the Save button.

       Saves the current state of the application to a JSON file at the selected file path.
       """
        data: dict = self.workspace.save_as_dict()
        ic(data)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as file:
                ic(data)
                json.dump(data, file)

    def on_load_button_click(self, *_args):
        """
        Handles the click event on the Load button.

        Opens the file dialog to select a JSON file to load the saved state of the application.
        Loads the data from the selected file and updates the application accordingly.
        Calls the `on_plot_button_click` method to plot the loaded data.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")])

        with open(file_path, 'r') as file:
            data = json.load(file)

        self.workspace.load_from_dict(data)
        self.on_plot_button_click()

    def add_additional_var_entries(self):
        """
        Adds additional variable entries to the application.
        Creates the necessary widgets for a new additional variable entry.
        """
        var_row = len(self.additional_vars_entries) + 7
        var_entry = UndoableCTkEntry(self.frame_grid[1][2], width=40)

        var_entry.grid(column=0, row=var_row, padx=1)

        func_entry = UndoableCTkEntry(self.frame_grid[1][2])
        func_entry.grid(column=1, row=var_row, padx=1)

        initial_value_entry = UndoableCTkEntry(self.frame_grid[1][2], width=40)
        initial_value_entry.grid(column=2, row=var_row, padx=1)
        initial_value_entry.insert(0, "1")

        plot_var = ctk.BooleanVar()
        plot_var_checkbox = ctk.CTkCheckBox(self.frame_grid[1][2], text="plot", variable=plot_var, width=60)
        plot_var_checkbox.grid(column=3, row=var_row)

        var_id = id(var_entry)
        additional_var = AdditionalVarEntry(var_id, var_entry, func_entry, initial_value_entry, plot_var)
        self.workspace.additional_vars_entries.append(additional_var)

        # create a new entry in additional_plots for the new variable
        fig_container = self.create_bordered_frame(self.frame_grid[1][2])

        fig_container.grid(column=4, row=var_row, padx=1)

        figure__, ax__, canvas__ = self.create_additional_figure_and_canvas(fig_container)
        self.additional_plots[var_id] = {'entry': var_entry, 'figure': figure__, 'ax': ax__, 'canvas': canvas__,
                                         'fig_container': fig_container, 'var_row': var_row}

    def set_entries_data_(self, data):
        """
        Sets the data of the entries based on the provided data.

        Args:
            data: The data to set the entries.
        """
        self.t_min_entry.delete(0, 'end')
        self.t_min_entry.insert(0, data.t_min)
        self.t_max_entry.delete(0, 'end')
        self.t_max_entry.insert(0, data.t_max)
        self.workspace.allow_iterations.set(data.allow_iterations)
        self.iterations_entry.delete(0, 'end')
        self.iterations_entry.insert(0, data.iteration_number)

        for i, (curve_name, x, y, z, *_) in enumerate(data.curve_entries.__dir__()):
            while i >= len(self.workspace.curve_entries):
                self.add_curve_entries()
            self.workspace.curve_entries[i].curve_name.delete(0, 'end')
            self.workspace.curve_entries[i].curve_name.insert(0, curve_name)
            self.workspace.curve_entries[i].x_entry.delete(0, 'end')
            self.workspace.curve_entries[i].x_entry.insert(0, x)
            self.workspace.curve_entries[i].y_entry.delete(0, 'end')
            self.workspace.curve_entries[i].y_entry.insert(0, y)
            self.workspace.curve_entries[i].z_entry.delete(0, 'end')
            self.workspace.curve_entries[i].z_entry.insert(0, z)

        for i, (var_id, var_name, func, init, plot_var) in enumerate(data.additional_vars_entries):
            while i >= len(self.workspace.additional_vars_entries):
                self.add_additional_var_entries()
            # additional_vars_entries[i][0] = var_id
            self.workspace.additional_vars_entries[i].var_entry.delete(0, 'end')
            self.workspace.additional_vars_entries[i].var_entry.insert(0, var_name)
            self.workspace.additional_vars_entries[i].func_entry.delete(0, 'end')
            self.workspace.additional_vars_entries[i].func_entry.insert(0, func)
            self.workspace.additional_vars_entries[i].init_entry.delete(0, 'end')
            self.workspace.additional_vars_entries[i].init_entry.insert(0, init)
            self.workspace.additional_vars_entries[i].graph_var.set(plot_var)
        for i, (vector_field_name, vectors_lengths, vectors_positions, vector_scale, vector_density) in enumerate(
                data.vector_field_entries):
            while i >= len(self.workspace.vector_field_entries):
                self.add_vector_field_entries()
            self.workspace.vector_field_entries[i].vector_field_name.delete(0, 'end')
            self.workspace.vector_field_entries[i].vector_field_name.insert(0, vector_field_name)
            self.workspace.vector_field_entries[i].vectors_lengths.delete(0, 'end')
            self.workspace.vector_field_entries[i].vectors_lengths.insert(0, vectors_lengths)
            self.workspace.vector_field_entries[i].vectors_positions.delete(0, 'end')
            self.workspace.vector_field_entries[i].vectors_positions.insert(0, vectors_positions)
            self.workspace.vector_field_entries[i].vector_scale.set(vector_scale)
            try:
                self.workspace.vector_field_entries[i].vector_density.delete(0, 'end')
                self.workspace.vector_field_entries[i].vector_density.insert(0, vector_density)
            except AttributeError:
                self.workspace.vector_field_entries[i].vector_density.set(vector_density)


def on_closing():
    """
    Handles the closing event of the application window.

    Performs any necessary cleanup operations before closing the application.
    By default, the function calls `app.quit()` to terminate the application.
    If you want to completely terminate the program, you can use `root.destroy()`.
    """
    app.quit()


# main function
if __name__ == "__main__":
    # Create a temporary file to store the DXF document
    temp_dxf_file = tempfile.NamedTemporaryFile(delete=False)
    temp_dxf_path = temp_dxf_file.name
    temp_dxf_file.close()

    # Initialize the list of curve entries, additional variables entries, additional plots, and entries
    curve_entries = []
    additional_vars_entries = []
    additional_plots = {}
    entries = []

    # Create an instance of the GraphingApplication class
    app = GraphingApplication()

    # Set appearance and window properties
    ctk.set_default_color_theme("Assets/Graphit.json")
    ctk.set_appearance_mode("light")
    app.title("Graphi(t)")
    app.iconbitmap('Assets/Icon.ico')
    app.geometry("800x600")
    app.minsize(1200, 600)

    # Set the closing protocol
    app.protocol("WM_DELETE_WINDOW", on_closing)

    app.bind('<Return>', app.on_plot_button_click)
    app.bind('<Control-s>', app.on_save_button_click)
    app.bind('<Control-o>', app.on_load_button_click)
    app.bind('<Control-e>', app.on_export_button_click)

    # Run the main loop of the application
    app.mainloop()
