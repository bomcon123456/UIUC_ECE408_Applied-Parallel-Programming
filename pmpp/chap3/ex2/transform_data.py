from pathlib import Path

import numpy as np


def read_raw_file_to_numpy(filepath):
    """
    Reads a raw file with dimensions on the first line and data afterwards
    into a NumPy array.

    Args:
        filepath (str): The path to the raw file.

    Returns:
        numpy.ndarray: The data from the file as a NumPy array.
    """
    with open(filepath, "r") as f:
        # Read the first line to get dimensions
        dims_str = f.readline().strip().split()
        rows = int(dims_str[0])
        cols = int(dims_str[1])

        # Read the rest of the file into a 1D array
        data_1d = np.loadtxt(f)

        # Reshape the 1D array into the specified dimensions
        # Use 'order='C'' for row-major (C-style) order, which is common.
        # If your data is column-major, you might need 'order='F''.
        data_2d = data_1d.reshape((rows, cols), order="C")
    return data_2d


def save_numpy_to_raw_file(numpy_array, filepath):
    """
    Saves a NumPy array to a raw file format, with dimensions on the first line
    and the array data afterwards.

    Args:
        numpy_array (numpy.ndarray): The NumPy array to save.
        filepath (str): The path to the file where the array will be saved.
    """
    # Get the dimensions of the NumPy array
    rows, cols = numpy_array.shape

    with open(filepath, "w") as f:
        # Write the dimensions to the first line
        f.write(f"{rows} {cols}\n")

        # Save the array data, ensuring it's flattened for savetxt and then
        # reshaped to its original dimensions for writing, or just write
        # the 2D array directly. np.savetxt handles 2D arrays well.
        # Use 'fmt' to specify the format of the numbers (e.g., '%.1f' for one decimal place)
        # and 'delimiter' to specify the separator between numbers.
        np.savetxt(f, numpy_array, fmt="%.2f", delimiter=" ")


dir = Path("./data")
for subdir in dir.iterdir():
    print(subdir)
    inp1path = subdir / "input0.raw"
    inp2path = subdir / "input1.raw"
    outputpath = subdir / "output.raw"
    inp1 = read_raw_file_to_numpy(inp1path)
    inp2 = read_raw_file_to_numpy(inp2path)
    inp2 = inp2[:, 0]
    out = inp1 * inp2
    save_numpy_to_raw_file(inp2[None, ...], inp2path)
    save_numpy_to_raw_file(out, outputpath)
