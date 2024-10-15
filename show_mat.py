import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.io

def visualize_data(mat_file):
    # Wczytaj dane z pliku .mat
    mat_data = scipy.io.loadmat(mat_file)
    
    # Zakładam, że dane są w zmiennej 'segments'
    segmented_data = mat_data.get('segments')
    
    if segmented_data is None:
        print("Nie znaleziono klucza 'segments' w pliku .mat.")
        return
    
    # Drukowanie liczby segmentów
    print(f"Liczba segmentów: {len(segmented_data)}")
    
    # Przeglądanie segmentów
    for i, segment in enumerate(segmented_data):
        print(f"\nSegment {i + 1}:")
        
        # Struktura segmentu (pose, orientations, emg)
        pose, orientations, emg = segment[0]  # Dostęp do zagnieżdżonych danych
        
        # Wyświetlanie informacji o gestach i orientacjach
        print(f"Pose: {pose}")
        print(f"Orientations shape: {orientations.shape}")
        
        # Sprawdzanie danych EMG
        if emg.size > 0:
            print(f"EMG data shape: {emg.shape}")
            print(f"First few EMG data points: {emg[:5]}")  # Wyświetlanie kilku pierwszych wartości EMG
        else:
            print("Brak danych EMG w tym segmencie.")

# Wywołaj funkcję
visualize_data('myo_segmented_data.mat')



if __name__ == "__main__":
    visualize_data('myo_segmented_data.mat')
