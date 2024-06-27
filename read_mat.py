import scipy.io

def read_mat_file(file_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        return mat_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Specify the path to your .mat file here
    file_path = 'C:/Users/MaciejBurakowski(258/Desktop/praca_inz/osoba_2.mat'
    
    data = read_mat_file(file_path)
    if data:
        print("Content of the .mat file:")
        for key, value in data.items():
            # Skip the default meta entries
            if key.startswith('__'):
                continue
            print(f"{key}: {value}")

