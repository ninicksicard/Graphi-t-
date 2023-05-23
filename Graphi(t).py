import json
import tempfile
from dataclasses import dataclass
from tkinter import filedialog

import customtkinter as ctk
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import os  # noqa: F401
import tkinter  # noqa: F401
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class CurveEntry:
    curve_name: any
    x_entry: any
    y_entry: any
    z_entry: any
    points: list = None
    i: int = None

    _count = 0  # Class variable to keep track of the total number of instances

    @classmethod
    def get_count(cls):
        return cls._count

    def __post_init__(self):
        if self.i is None:
            self.__class__._count += 1  # Use __class__ to access the class variable
            self.i = self.__class__._count


@dataclass
class VectorField:
    vector_field_name: any
    vectors_lengths: any
    vectors_positions: any
    vector_scale: any
    vector_density: any
    i: int = None

    _count = 0  # Class variable to keep track of the total number of instances

    @classmethod
    def get_count(cls):
        return cls._count

    def __post_init__(self):
        if self.i is None:
            self.__class__._count += 1  # Use __class__ to access the class variable
            self.i = self.__class__._count


@dataclass
class AdditionalVarEntry:
    var_id: any
    var_entry: any
    func_entry: any
    init_entry: any
    graph_var: any
    i: int = None

    _count = 0  # Class variable to keep track of the total number of instances

    @classmethod
    def get_count(cls):
        return cls._count

    def __post_init__(self):
        if self.i is None:
            self.__class__._count += 1  # Use __class__ to access the class variable
            self.i = self.__class__._count


class UndoableCTkEntry(ctk.CTkEntry):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._undo_stack = []
        self._redo_stack = []
        self.bind('<Key>', self._on_key)  # Listen for key press
        self.bind('<Control-z>', self._on_undo)  # Listen for undo command
        self.bind('<Control-y>', self._on_redo)  # Listen for redo command

    def _on_key(self, event):
        self._undo_stack.append(self.get())  # Add current state to undo stack
        self._redo_stack = []  # Clear redo stack when new key is pressed

    def _on_undo(self, event):
        if self._undo_stack:
            last_state = self._undo_stack.pop()  # Remove last state from undo stack
            self._redo_stack.append(last_state)  # Push it to redo stack
            self.delete(0, 'end')  # Clear the entry
            if self._undo_stack:
                self.insert(0, self._undo_stack[-1])  # Insert the last state

    def _on_redo(self, event):
        if self._redo_stack:
            self.delete(0, 'end')  # Clear the entry
            self.insert(0, self._redo_stack[-1])  # Insert the last state from redo stack
            self._undo_stack.append(self._redo_stack.pop())  # Push it back to undo stack



class GraphingApplication(ctk.CTk):

    def __init__(self):
        super().__init__()
        # Create a new DXF document
        self.doc = ezdxf.new('R2010')

        self.msp = None
        self.curve_entries = []
        self.additional_vars_entries = []
        self.vector_fields = []
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
        self.allow_iterations = ctk.BooleanVar(value=True)
        self.allow_iterations_checkbox = ctk.CTkCheckBox(self.frame_grid[2][0], text="Allow iteration",
                                                         variable=self.allow_iterations)
        self.allow_iterations_checkbox.pack(side='top', pady=2)

        self.iterations_label, self.iterations_entry = self.create_labeled_entry(self.frame_grid[2][0], "Iterations:",
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

    def get_entries_data(self):
        data_ = {'t_min': self.t_min_entry.get(),
                 't_max': self.t_max_entry.get(),
                 'allow_iterations': self.allow_iterations.get(),
                 'iteration_number': self.iterations_entry.get(),
                 'curve_entries': [(
                     curve_entry.curve_name.get(),
                     curve_entry.x_entry.get(),
                     curve_entry.y_entry.get(),
                     curve_entry.z_entry.get()
                 ) for curve_entry in self.curve_entries],
                 'additional_vars_entries': [(
                     additional_var_entry.var_id,
                     additional_var_entry.var_entry.get(),
                     additional_var_entry.func_entry.get(),
                     additional_var_entry.init_entry.get(),
                     additional_var_entry.graph_var.get()
                 ) for additional_var_entry in self.additional_vars_entries],
                 'vector_field_entries': [(
                     vector_field.vector_field_name.get(),
                     vector_field.vectors_lengths.get(),
                     vector_field.vectors_positions.get(),
                     vector_field.vector_scale.get(),
                     vector_field.vector_density.get()
                 ) for vector_field in self.vector_fields]}
        return data_

    def evaluate_entry_func(self, entry):
        for i in range(0, 5):
            try:
                evaluation = np.broadcast_to(eval(entry.get(), self.eval_globals), (self.t_range.shape[0] - i,))
            except ValueError:
                continue
            return evaluation

    def process_plot_var_entry(self, additional_var_entry):
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
        # Function to process a variable entry
        var_name = additional_var_entry.var_entry.get()
        if var_name:
            self.eval_globals[var_name] = int(additional_var_entry.init_entry.get())

    def update_variable(self, additional_var_entry):
        # Function to perform an update for one variable
        var_name = additional_var_entry.var_entry.get()
        if var_name:
            self.eval_globals[var_name] = eval(additional_var_entry.func_entry.get(), self.eval_globals)

    def evaluate_and_plot_curve(self, curve_entry, msp):
        # Function to evaluate the points for a curve
        x = self.evaluate_entry_func(curve_entry.x_entry)
        y = self.evaluate_entry_func(curve_entry.y_entry)
        z = self.evaluate_entry_func(curve_entry.z_entry)
        self.ax.plot(x, y, z, color=plt.colormaps['plasma'](curve_entry.i / len(self.curve_entries)))
        points = list(zip(x, y, z))
        msp.add_polyline3d(points)

    def evaluate_and_plot_vector_field(self, vector_field):
        for lengths_curve in self.curve_entries:
            if lengths_curve.curve_name.get() == vector_field.vectors_lengths.get():
                for positions_curve in self.curve_entries:
                    if positions_curve.curve_name.get() == vector_field.vectors_positions.get():
                        step_value = 100-round(eval(vector_field.vector_density.get()))  # Plot every 10th vector

                        vector_scale = np.exp(vector_field.vector_scale.get())
                        # Function to evaluate the points for a curve
                        x_root = self.evaluate_entry_func(positions_curve.x_entry)[::step_value]
                        y_root = self.evaluate_entry_func(positions_curve.y_entry)[::step_value]
                        z_root = self.evaluate_entry_func(positions_curve.z_entry)[::step_value]

                        x_tip = self.evaluate_entry_func(lengths_curve.x_entry)[::step_value] * vector_scale
                        y_tip = self.evaluate_entry_func(lengths_curve.y_entry)[::step_value] * vector_scale
                        z_tip = self.evaluate_entry_func(lengths_curve.z_entry)[::step_value] * vector_scale

                        length = min(len(x_tip), len(y_tip), len(z_tip), len(x_root), len(y_root), len(z_root))

                        self.ax.quiver(x_root[0:length], y_root[0:length], z_root[0:length], x_tip[0:length],
                                       y_tip[0:length], z_tip[0:length])

    # Rest of your methods go here, for example:
    def create_labeled_entry(self, frame, text, default_value):
        container_ = self.create_bordered_frame(frame)
        container_.pack(side='top', padx=5, pady=2)
        label = ctk.CTkLabel(container_, text=text)
        label.pack(side='left', pady=2, padx=5)

        entry = UndoableCTkEntry(container_)
        entry.insert(0, default_value)
        entry.pack(side='left', pady=2, padx=5)
        print(entry.__class__.__name__)
        return label, entry

    def create_additional_figure_and_canvas(self, parent_):
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
        if parent:
            frame = ctk.CTkFrame(parent, bg_color='transparent', height=0, width=0)
        else:
            frame = ctk.CTkFrame(self, bg_color='transparent', height=0, width=0)
        return frame

    def create_frame_grid(self, parent, rows, columns, main_row=1, main_column=1):
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
        # T range
        t_container = self.create_bordered_frame(frame)
        t_container.pack()
        t_min_entry_ = ctk.CTkEntry(t_container, width=40)
        t_min_entry_.insert(0, '0')
        t_min_entry_.pack(side='left', padx=5, expand=True, )

        t_label = ctk.CTkLabel(t_container, text=" < t < ")
        t_label.pack(side='left', padx=5)

        t_max_entry_ = ctk.CTkEntry(t_container, width=40)
        t_max_entry_.insert(0, '10')
        t_max_entry_.pack(side='left', padx=5, expand=True)
        return t_min_entry_, t_max_entry_

    def create_additional_vars_frame(self, frame_):
        var_label = ctk.CTkLabel(frame_, text="Variable")
        var_label.grid(column=0, row=len(self.additional_vars_entries) + 6, padx=2, pady=5)
        func_label = ctk.CTkLabel(frame_, text="Function")
        func_label.grid(column=1, row=len(self.additional_vars_entries) + 6, padx=2, pady=5)
        initial_value_label = ctk.CTkLabel(frame_, text="Initial value")
        initial_value_label.grid(column=2, row=len(self.additional_vars_entries) + 6, padx=2, pady=5)

    def add_curve_entries(self):
        container__ = self.create_bordered_frame(self.frame_grid[1][0])

        container__.pack(side=ctk.TOP, padx=5, pady=5)
        curve_name_label, curve_name_entry = self.create_labeled_entry(container__, "Curve name:",
                                                                       str(CurveEntry.get_count()))

        _, x_entry = self.create_labeled_entry(container__, "x(t):", "1")
        _, y_entry = self.create_labeled_entry(container__, "y(t):", "t")
        _, z_entry = self.create_labeled_entry(container__, "z(t):", "0")
        curve_entry = CurveEntry(curve_name_entry, x_entry, y_entry, z_entry)
        self.curve_entries.append(curve_entry)

    def add_vector_field_entries(self):
        container__ = self.create_bordered_frame(self.frame_grid[1][0])

        container__.pack(side=ctk.TOP, padx=5, pady=5)
        vector_field_name_label, vector_field_name_entry = self.create_labeled_entry(container__, "Vector field name:",
                                                                                     str(VectorField.get_count()))

        _, vectors_lengths = self.create_labeled_entry(container__, "vectors lengths :", "write a curve name here")
        _, vectors_positions = self.create_labeled_entry(container__, "vectors positions : ", "write a curve name here")

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
        vector_density_slider = ctk.CTkEntry(container_vd)
        vector_density_slider.insert(0, 30)
        vector_density_slider.pack(side=ctk.TOP, padx=5, pady=5)

        vector_field = VectorField(vector_field_name_entry, vectors_lengths, vectors_positions, vector_scale_slider,
                                   vector_density_slider)
        self.vector_fields.append(vector_field)

    def on_plot_button_click(self, *args):

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
        if self.allow_iterations.get():
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

    def on_export_button_click(self, *args):
        # Open the file dialog to get the path where the user wants to save the DWG file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".dwg",
            filetypes=[("DWG files", "*.dwg")])
        if file_path:
            # Load the DXF document from the temporary file
            self.doc = ezdxf.readfile(temp_dxf_path)

            # Save the DXF document as a DWG file at the selected path
            self.doc.saveas(file_path)

    def on_save_button_click(self, *args):
        data = self.get_entries_data()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")])
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def on_load_button_click(self, *args):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")])

        with open(file_path, 'r') as file:
            data = json.load(file)

        self.set_entries_data_(data)
        self.on_plot_button_click()

    def add_additional_var_entries(self):
        var_row = len(self.additional_vars_entries) + 7
        var_entry = ctk.CTkEntry(self.frame_grid[1][2], width=40)

        var_entry.grid(column=0, row=var_row, padx=1)

        func_entry = ctk.CTkEntry(self.frame_grid[1][2])
        func_entry.grid(column=1, row=var_row, padx=1)

        initial_value_entry = ctk.CTkEntry(self.frame_grid[1][2], width=40)
        initial_value_entry.grid(column=2, row=var_row, padx=1)
        initial_value_entry.insert(0, "1")

        plot_var = ctk.BooleanVar()
        plot_var_checkbox = ctk.CTkCheckBox(self.frame_grid[1][2], text="plot", variable=plot_var, width=60)
        plot_var_checkbox.grid(column=3, row=var_row)

        var_id = id(var_entry)
        additional_var = AdditionalVarEntry(var_id, var_entry, func_entry, initial_value_entry, plot_var)
        self.additional_vars_entries.append(additional_var)

        # create a new entry in additional_plots for the new variable
        fig_container = self.create_bordered_frame(self.frame_grid[1][2])

        fig_container.grid(column=4, row=var_row, padx=1)

        figure__, ax__, canvas__ = self.create_additional_figure_and_canvas(fig_container)
        self.additional_plots[var_id] = {'entry': var_entry, 'figure': figure__, 'ax': ax__, 'canvas': canvas__,
                                         'fig_container': fig_container, 'var_row': var_row}

    def set_entries_data_(self, data):
        self.t_min_entry.delete(0, 'end')
        self.t_min_entry.insert(0, data['t_min'])
        self.t_max_entry.delete(0, 'end')
        self.t_max_entry.insert(0, data['t_max'])
        self.allow_iterations.set(data['allow_iterations'])
        self.iterations_entry.delete(0, 'end')
        self.iterations_entry.insert(0, data['iteration_number'])

        for i, (curve_name, x, y, z) in enumerate(data['curve_entries']):
            while i >= len(self.curve_entries):
                self.add_curve_entries()
            self.curve_entries[i].curve_name.delete(0, 'end')
            self.curve_entries[i].curve_name.insert(0, curve_name)
            self.curve_entries[i].x_entry.delete(0, 'end')
            self.curve_entries[i].x_entry.insert(0, x)
            self.curve_entries[i].y_entry.delete(0, 'end')
            self.curve_entries[i].y_entry.insert(0, y)
            self.curve_entries[i].z_entry.delete(0, 'end')
            self.curve_entries[i].z_entry.insert(0, z)

        for i, (var_id, var_name, func, init, plot_var) in enumerate(data['additional_vars_entries']):
            while i >= len(self.additional_vars_entries):
                self.add_additional_var_entries()
            # additional_vars_entries[i][0] = var_id
            self.additional_vars_entries[i].var_entry.delete(0, 'end')
            self.additional_vars_entries[i].var_entry.insert(0, var_name)
            self.additional_vars_entries[i].func_entry.delete(0, 'end')
            self.additional_vars_entries[i].func_entry.insert(0, func)
            self.additional_vars_entries[i].init_entry.delete(0, 'end')
            self.additional_vars_entries[i].init_entry.insert(0, init)
            self.additional_vars_entries[i].graph_var.set(plot_var)
        for i, (vector_field_name, vectors_lengths, vectors_positions, vector_scale, vector_density) in enumerate(
                data['vector_field_entries']):
            while i >= len(self.vector_fields):
                self.add_vector_field_entries()
            self.vector_fields[i].vector_field_name.delete(0, 'end')
            self.vector_fields[i].vector_field_name.insert(0, vector_field_name)
            self.vector_fields[i].vectors_lengths.delete(0, 'end')
            self.vector_fields[i].vectors_lengths.insert(0, vectors_lengths)
            self.vector_fields[i].vectors_positions.delete(0, 'end')
            self.vector_fields[i].vectors_positions.insert(0, vectors_positions)
            self.vector_fields[i].vector_scale.set(vector_scale)
            try :
                self.vector_fields[i].vector_density.delete(0, 'end')
                self.vector_fields[i].vector_density.insert(0, vector_density)
            except AttributeError:
                self.vector_fields[i].vector_density.set(vector_density)


def on_closing():
    # Here you can do any cleanup if needed
    app.quit()  # Or root.destroy(), if you want to completely terminate the program


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
