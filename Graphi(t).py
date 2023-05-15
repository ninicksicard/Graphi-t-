import json
import os
import tempfile
from tkinter import filedialog

import customtkinter as ctk
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D


def on_save_button_click():
    def get_entries_data():
        data_ = {'t_min': t_min_entry.get(),
                 't_max': t_max_entry.get(),
                 'allow_iterations': allow_iterations.get(),
                 'iteration_number': iterations_entry.get(),
                 'curve_entries': [(
                     entry[0].get(),
                     entry[1].get(),
                     entry[2].get(),
                     entry[3].get()
                 ) for entry in curve_entries],
                 'additional_vars_entries': [(
                     entry[0],
                     entry[1].get(),
                     entry[2].get(),
                     entry[3].get(),
                     entry[4].get()
                 ) for entry in additional_vars_entries]}
        return data_

    data = get_entries_data()
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")])
    with open(file_path, 'w') as file:
        json.dump(data, file)


def on_plot_button_click():
    global doc
    t_start = float(t_min_entry.get())
    t_end = float(t_max_entry.get())
    t_range = np.linspace(t_start, t_end, 1000)

    eval_globals['t'] = t_range

    for var_id, var_entry, func_entry, init_entry, graph_var in additional_vars_entries:
        var_name = var_entry.get()
        if var_name:
            eval_globals[var_name] = int(init_entry.get())

    if allow_iterations.get():
        iteration_count = int(iterations_entry.get())
        for _ in range(iteration_count):
            for var_id, var_entry, func_entry, init_entry, graph_var in additional_vars_entries:
                var_name = var_entry.get()
                if var_name:
                    eval_globals[var_name] = eval(func_entry.get(), eval_globals)
    no_graph = True

    for var_id, var_entry, func_entry, init_entry, graph_var in additional_vars_entries:
        if var_entry.get():
            plot_info = additional_plots[var_id]

            # assuming func_entry.get() returns a string representing a function of t
            y_function = lambda t: np.broadcast_to(eval(func_entry.get(), eval_globals), t.shape)
            plot_info['ax'].clear()
            plot_info['ax'].plot(t_range, y_function(t_range))
            if graph_var.get():
                plot_info['fig_container'].grid(column=4, row=(plot_info['var_row']))

                plot_info['canvas'].draw()
                no_graph = False
            else:
                plot_info['fig_container'].grid_forget()

    if no_graph:
        container.grid_columnconfigure(2, weight=0, minsize=350)
    else:
        container.grid_columnconfigure(2, weight=0, minsize=550)

    ax.clear()

    # Create a new DXF document
    doc = ezdxf.new('R2010')

    # Create a new 3D polyline in the modelspace
    msp = doc.modelspace()

    for i, (curve_name, x_entry, y_entry, z_entry) in enumerate(curve_entries):
        x_function = lambda t: np.broadcast_to(eval(x_entry.get(), eval_globals), t.shape)
        y_function = lambda t: np.broadcast_to(eval(y_entry.get(), eval_globals), t.shape)
        z_function = lambda t: np.broadcast_to(eval(z_entry.get(), eval_globals), t.shape)

        plot_parametric_3d(x_function, y_function, z_function, t_range, ax, color=plt.cm.jet(i / len(curve_entries)))

        x = x_function(t_range)
        y = y_function(t_range)
        z = z_function(t_range)
        points = list(zip(x, y, z))
        polyline = msp.add_polyline3d(points)

    doc.saveas(temp_dxf_path)
    if auto_size_vs_equal_axis.get():
        ax.axis('equal')
    else:
        ax.autoscale(enable=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    canvas.draw()


def on_export_button_click():
    global doc
    # Open the file dialog to get the path where the user wants to save the DWG file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".dwg",
        filetypes=[("DWG files", "*.dwg")])
    if file_path:
        # Load the DXF document from the temporary file
        doc = ezdxf.readfile(temp_dxf_path)

        # Save the DXF document as a DWG file at the selected path
        doc.saveas(file_path)


def plot_parametric_3d(x_func, y_func, z_func, t, ax_, color):
    x = x_func(t)
    y = y_func(t)
    z = z_func(t)
    ax_.plot(x, y, z, color=color)


def create_widgets():
    container_ = create_bordered_frame(root)
    container_.pack(side='bottom', fill='both', expand=True)

    frame_grid_ = create_frame_grid(container_, 3, 3)

    # T range
    t_min_entry_, t_max_entry_ = create_t_range_entry(frame_grid_[0][1])

    def add_curve_entries_():
        global curve_count

        container__ = create_bordered_frame(frame_grid_[1][0])

        container__.pack(side=ctk.TOP, padx=5, pady=5)
        curve_name_label, curve_name_entry = create_labeled_entry(container__, "Curve name:", str(curve_count))

        _, x_entry = create_labeled_entry(container__, "x(t):", "1")
        _, y_entry = create_labeled_entry(container__, "y(t):", "t")
        _, z_entry = create_labeled_entry(container__, "z(t):", "0")
        curve_entries.append((curve_name_entry, x_entry, y_entry, z_entry))
        curve_count += 1

    # Add curve button
    add_curve_button = ctk.CTkButton(frame_grid_[0][0], text="+ Curve", command=add_curve_entries_)
    add_curve_button.pack(side='top', padx=5)

    # Create initial x, y, and z entries
    add_curve_entries_()

    def add_additional_var_entries_():
        var_row = len(additional_vars_entries) + 7
        var_entry = ctk.CTkEntry(frame_grid_[1][2], width=40)

        var_entry.grid(column=0, row=var_row, padx=1)

        func_entry = ctk.CTkEntry(frame_grid_[1][2])
        func_entry.grid(column=1, row=var_row, padx=1)

        initial_value_entry = ctk.CTkEntry(frame_grid_[1][2], width=40)
        initial_value_entry.grid(column=2, row=var_row, padx=1)
        initial_value_entry.insert(0, "1")

        plot_var = ctk.BooleanVar()
        plot_var_checkbox = ctk.CTkCheckBox(frame_grid_[1][2], text="plot", variable=plot_var, width=60)
        plot_var_checkbox.grid(column=3, row=var_row)

        var_id = id(var_entry)
        additional_vars_entries.append([var_id, var_entry, func_entry, initial_value_entry, plot_var])

        # create a new entry in additional_plots for the new variable
        fig_container = create_bordered_frame(frame_grid_[1][2])

        fig_container.grid(column=4, row=var_row, padx=1)

        def create_additional_figure_and_canvas(parent_):
            fig___ = plt.figure(figsize=(2, 1), )  # Change the 3, 3 to whatever size you want in inches
            ax___ = fig___.add_subplot(111)
            font_size = 6
            ax___.set_xlabel('X', fontsize=font_size)
            ax___.set_ylabel('Y', fontsize=font_size)
            for label in (ax___.get_xticklabels() + ax___.get_yticklabels()):
                label.set_fontsize(font_size)
            canvas___ = FigureCanvasTkAgg(fig___, master=parent_)
            canvas___.get_tk_widget().grid(row=len(additional_vars_entries) + 7, column=4)
            return fig___, ax___, canvas___

        figure__, ax__, canvas__ = create_additional_figure_and_canvas(fig_container)
        additional_plots[var_id] = {'entry': var_entry, 'figure': figure__, 'ax': ax__, 'canvas': canvas__,
                                    'fig_container': fig_container, 'var_row': var_row}

    # Add variable button
    add_var_button = ctk.CTkButton(frame_grid_[0][2], text="+ Variable", command=add_additional_var_entries_)
    add_var_button.pack(side=ctk.LEFT, padx=5)

    # Settings
    allow_iterations_ = ctk.BooleanVar(value=True)
    allow_iterations_checkbox = ctk.CTkCheckBox(frame_grid_[2][0], text="Allow iteration", variable=allow_iterations_)
    allow_iterations_checkbox.pack(side='top', pady=2)

    iterations_label, iterations_entry_ = create_labeled_entry(frame_grid_[2][0], "Iterations:", "100")

    # Add Checkbox for auto size vs equal axis
    auto_size_vs_equal_axis_ = ctk.BooleanVar(value=True)
    auto_size_vs_equal_axis_checkbox = ctk.CTkCheckBox(frame_grid_[2][1], text="Equal Axis",
                                                       variable=auto_size_vs_equal_axis_)
    auto_size_vs_equal_axis_checkbox.pack(side='top', pady=2)

    # Plot button
    plot_button = ctk.CTkButton(frame_grid_[2][1], text="Plot", command=lambda: on_plot_button_click())
    plot_button.pack(side='top', pady=2, fill='none')

    # Save button
    save_button = ctk.CTkButton(frame_grid_[2][2], text="Save", command=on_save_button_click)
    save_button.pack(side='top', pady=2, fill='none')

    # export dwg
    export_button = ctk.CTkButton(frame_grid_[2][2], text="Export", command=on_export_button_click)
    export_button.pack(side='top', pady=2, fill='none')

    def on_load_button_click():
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")])

        with open(file_path, 'r') as file:
            data = json.load(file)

        def set_entries_data_(data___):
            t_min_entry.delete(0, 'end')
            t_min_entry.insert(0, data___['t_min'])
            t_max_entry.delete(0, 'end')
            t_max_entry.insert(0, data___['t_max'])
            allow_iterations.set(data___['allow_iterations'])
            iterations_entry.delete(0, 'end')
            iterations_entry.insert(0, data___['iteration_number'])

            for i, (curve_name, x, y, z) in enumerate(data___['curve_entries']):
                while i >= len(curve_entries):
                    add_curve_entries_()
                curve_entries[i][0].delete(0, 'end')
                curve_entries[i][0].insert(0, curve_name)
                curve_entries[i][1].delete(0, 'end')
                curve_entries[i][1].insert(0, x)
                curve_entries[i][2].delete(0, 'end')
                curve_entries[i][2].insert(0, y)
                curve_entries[i][3].delete(0, 'end')
                curve_entries[i][3].insert(0, z)

            for i, (var_id, var_name, func, init, plot_var) in enumerate(data___['additional_vars_entries']):
                while i >= len(additional_vars_entries):
                    add_additional_var_entries_()
                # additional_vars_entries[i][0] = var_id
                additional_vars_entries[i][1].delete(0, 'end')
                additional_vars_entries[i][1].insert(0, var_name)
                additional_vars_entries[i][2].delete(0, 'end')
                additional_vars_entries[i][2].insert(0, func)
                additional_vars_entries[i][3].delete(0, 'end')
                additional_vars_entries[i][3].insert(0, init)
                additional_vars_entries[i][4].set(plot_var)

        set_entries_data_(data)
        on_plot_button_click()

    # Load button
    load_button = ctk.CTkButton(frame_grid_[2][2], text="Load", command=on_load_button_click)
    load_button.pack(side='top', pady=2, fill='none')

    # Canvas for the 3D plot
    figure = plt.figure()
    ax_ = figure.add_subplot(111, projection='3d')
    # Add your plot commands here

    ax_.set_xlabel('X')
    ax_.set_ylabel('Y')
    ax_.set_zlabel('Z')

    ax_.autoscale(enable=True)

    canvas_ = FigureCanvasTkAgg(figure, frame_grid_[1][1])

    # add this where you create your canvas
    toolbar_ = NavigationToolbar2Tk(canvas_, frame_grid_[1][1])

    toolbar_.pack(expand=True, side="top")
    toolbar_.update()

    canvas_.get_tk_widget().pack(expand=True, fill='both', side='top')

    def onclick(event):
        if event.inaxes is not None:
            data_coords = event.inaxes.transData.inverted().transform((event.x, event.y))
            print(f'Data coordinates: {data_coords}')

    # canvas_.mpl_connect('button_press_event', onclick)

    # Initialize additional_vars_frame
    create_additional_vars_frame(frame_grid_[1][2])
    return canvas_, ax_, allow_iterations_, iterations_entry_, frame_grid_, container_, t_min_entry_, t_max_entry_, auto_size_vs_equal_axis_


def create_additional_vars_frame(frame_):
    var_label = ctk.CTkLabel(frame_, text="Variable")
    var_label.grid(column=0, row=len(additional_vars_entries) + 6, padx=2, pady=5)
    func_label = ctk.CTkLabel(frame_, text="Function")
    func_label.grid(column=1, row=len(additional_vars_entries) + 6, padx=2, pady=5)
    initial_value_label = ctk.CTkLabel(frame_, text="Initial value")
    initial_value_label.grid(column=2, row=len(additional_vars_entries) + 6, padx=2, pady=5)


def create_frame_grid(parent, rows, columns, main_row=1, main_column=1):
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
            if column == 2 and row == 1:
                frame = ctk.CTkScrollableFrame(parent)
                frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
            else:
                frame = create_bordered_frame(parent)
                frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")

            frame_row.append(frame)
        frames.append(frame_row)
    return frames


def create_bordered_frame(parent):
    frame = ctk.CTkFrame(parent, bg_color='transparent', height=0, width=0)
    return frame


def create_t_range_entry(frame):
    # T range
    t_container = create_bordered_frame(frame)
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


def create_labeled_entry(frame, text, default_value):
    container_ = create_bordered_frame(frame)
    container_.pack(side='top', padx=5, pady=2)
    label = ctk.CTkLabel(container_, text=text)
    label.pack(side='left', pady=2, padx=5)

    entry = ctk.CTkEntry(container_)
    entry.insert(0, default_value)
    entry.pack(side='left', pady=2, padx=5)

    return label, entry


def on_closing():
    # Here you can do any cleanup if needed
    root.quit()  # Or root.destroy(), if you want to completely terminate the program


if __name__ == "__main__":

    # Create a new DXF document outside of the function, so it can be accessed by both functions
    doc = ezdxf.new('R2010')

    # Create a temporary file to store the DXF document
    temp_dxf_file = tempfile.NamedTemporaryFile(delete=False)
    temp_dxf_path = temp_dxf_file.name
    temp_dxf_file.close()

    curve_entries = []
    additional_vars_entries = []
    additional_plots = {}
    entries = []
    curve_count = 1

    eval_globals = {
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
        'abs': np.abs
    }

    root = ctk.CTk()
    ctk.set_default_color_theme("Assets/Graphit.json")
    ctk.set_appearance_mode("light")

    root.title("Graphi(t)")
    root.iconbitmap('Assets/Icon.ico')  # Add this line
    root.geometry("800x600")  # Set initial window size
    root.minsize(1200, 600)  # Set minimum window size
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    canvas, ax, allow_iterations,\
        iterations_entry, frame_grid, container, t_min_entry, t_max_entry, auto_size_vs_equal_axis = create_widgets()

    root.mainloop()
