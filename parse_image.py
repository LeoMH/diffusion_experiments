import tifffile
import numpy as np
import matplotlib.pyplot as plt

def parse_image(image_path):
    """
    Parse a TIFF image and return the data as a numpy array.
    :param
    image_path: Path to the TIFF image.
    :return: Numpy array of the image data.
    """
    # Read the TIFF image
    image = tifffile.imread(image_path)

    # Convert the image to a numpy array
    data = np.array(image)

    # Normalize the data to be between 0 and 1
    data = data / 255.0
    
    # Make sure that data is a 2d numpy array
    if data.ndim == 3:
        data = data.mean(axis=2)  # Convert to grayscale by averaging the color channels
    elif data.ndim > 2:
        raise ValueError("Image has more than 2 dimensions and cannot be converted to grayscale.")

    # Threshold the data to be between 0 and 1
    data = data < 0.5
    data = data.astype(np.float64)

    # Make the shape divisible by 8 by removing the last few rows and columns
    rows, cols = data.shape
    rows_to_remove = rows % 8
    cols_to_remove = cols % 8
    if rows_to_remove > 0:
        data = data[:-rows_to_remove, :]
    if cols_to_remove > 0:
        data = data[:, :-cols_to_remove]

    return data


if __name__ == "__main__":
    # Example usage
    image_path = "Cu-Sinterstruktur(200x200µm).tif"
    data = parse_image(image_path)
    
    # Print the shape of the data
    print(f"Data shape: {data.shape}")
    
    # Plot the data
    plt.imshow(data, cmap='gray')
    plt.show()