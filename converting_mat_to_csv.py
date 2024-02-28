import scipy.io
import pandas as pd
import os

# Directory containing the datasets
dataset_directory = './dataset' 

# Iterate through directories numbered 1 to 10
for i in range(1, 11):
    directory = os.path.join(dataset_directory, str(i))
    
    # Check if the directory exists
    if os.path.exists(directory):
        mat_file_path = os.path.join(directory, 'Vel.mat')

        # Check if the Vel.mat file exists in the directory
        if os.path.exists(mat_file_path):
            # Load the .mat file
            data = scipy.io.loadmat(mat_file_path)
            df = pd.DataFrame(data['V'], columns=['vx', 'vy', 'vz', 'omega_z'])

            # Save to CSV in the same directory
            csv_file_path = os.path.join(directory, 'data_out.csv')
            df.to_csv(csv_file_path, index=False)

        else:
            print(f"'Vel.mat' file not found in {directory}")
    else:
        print(f"Directory {directory} does not exist")
