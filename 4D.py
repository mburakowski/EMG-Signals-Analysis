import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def read_mat_file(file_path):
    """
    This function reads a .mat file and returns its content.
    
    Parameters:
    file_path (str): The path to the .mat file
    
    Returns:
    dict: The content of the .mat file
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        return mat_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_full_width(data, index4D, window_title):
    """
    This function plots a 4D dataset in full width.
    
    Parameters:
    data (dict): The content of the .mat file
    index4D (int): The index for the 4th dimension
    window_title (str): The title of the window
    """
    keys = list(data.keys())
    keys = [key for key in keys if not key.startswith('__')]
    
    plt.figure(figsize=(15, 5))  # Set figure size to full width
    
    for key in keys:
        dataset = data[key]
        if dataset.ndim == 4:
            # Assuming the shape of the dataset is (dim1, dim2, dim3, dim4)
            plt.plot(dataset[:, :, :, index4D].flatten(), label=key)
        else:
            plt.plot(dataset.flatten(), label=key)
    
    plt.title(window_title)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path to your .mat file here
    file_path = 'C:/Users/MaciejBurakowski(258/Desktop/praca_inz/osoba_2.mat'
    index4D = 0  # Adjust this index based on your data
    
    data = read_mat_file(file_path)
    if data:
        # Plot the data in full width
        plot_full_width(data, index4D, "Full Width Plot")
