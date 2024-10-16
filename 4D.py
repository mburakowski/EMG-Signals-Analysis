import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def read_mat_file(file_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        print("Zawartość pliku .mat:")
        for key, value in mat_data.items():
            if not key.startswith('__'):
                print(f"\nKlucz: {key}")
                print(f"Kształt danych: {value.shape}")  # Dodanie wyświetlania kształtu danych
                print(value)
        return mat_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_full_width(data, index4D, window_title):
    keys = list(data.keys())
    keys = [key for key in keys if not key.startswith('__')]
    
    plt.figure(figsize=(15, 5))  
    
    for key in keys:
        dataset = data[key]
        if dataset.ndim == 4:
            print(f"Rysowanie danych 4D: {key}, wybrany indeks: {index4D}")
            # Zakładam, że chcesz narysować płaski sygnał dla konkretnego wymiaru, np. czasowy
            plt.plot(dataset[:, :, :, index4D].flatten(), label=key)
        else:
            print(f"Rysowanie danych {dataset.ndim}D: {key}")
            plt.plot(dataset.flatten(), label=key)
    
    plt.title(window_title)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #file_path = 'C:/Users/MaciejBurakowski(258/Desktop/praca_inz/osoba_2.mat'
    file_path = 'C:/Users/MaciejBurakowski(258/Desktop/praca_inz/myo_segmented_data.mat'
    #file_path = 'C:/Users/MaciejBurakowski(258/Desktop/praca_inz/myo_data.mat'
    
    index4D = 0  # Możesz zmieniać indeks, aby zobaczyć inne ruchy lub sygnały
    
    data = read_mat_file(file_path)
    if data:
        plot_full_width(data, index4D, "Wykres")
