import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["os"],
    "include_files": [
        ("Assets/Icon.ico", "Assets/Icon.ico"),
        ("Assets/Graphit.json", "Assets/Graphit.json")
    ]
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

shortcut_table = [
    ("DesktopShortcut",  # Shortcut
     "DesktopFolder",  # Directory_
     "Graphi(t)",  # Name that will be shown on the link
     "TARGETDIR",  # Component_
     "[TARGETDIR]Graphi(t).exe",  # Target exe to execute
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     "Icon.ico",  # Icon
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     )
    ,
    ("StartMenuShortcut",  # Shortcut
     "StartMenuFolder",  # Directory_
     "Graphi(t)",  # Name that will be shown on the link
     "TARGETDIR",  # Component_
     "[TARGETDIR]Graphi(t).exe",  # Target exe to execute
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     "Icon.ico",  # Icon
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     )
]

# Now create the table dictionary
msi_data = {"Shortcut": shortcut_table, "Icon": [("Icon.ico", "Assets/Icon.ico")]}

# Change some default MSI options and specify the use of the above-defined tables
bdist_msi_options = {
    'data': msi_data}

setup(
    name="Graphi(t)",
    version="0.1",
    description="Graphi(t) is a user-friendly software for creating and visualizing 3D parametric plots with ease.",
    author="Nicolas Sicard",
    author_email="nicolassicardroy+Github@gmail.com",
    url="https://github.com/ninicksicard/Graphi-t-",
    license="MIT",
    options={
        "bdist_msi": bdist_msi_options,
        "build_exe": build_exe_options
    },
    executables=[
        Executable(
            "Graphi(t).py",
            base=base,
            icon="Assets/Icon.ico",
        )
    ]
)
