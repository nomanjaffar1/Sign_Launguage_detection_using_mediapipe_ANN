import splitfolders

# Specify your input directory containing the dataset
input_folder = r'C:\Sign_Language_Detection\gesture_data_set'

# Specify the output directory where split data will be saved
output_folder = r'C:\Sign_Language_Detection\split_dataset'

# Perform the split with a ratio of 80% training and 20% validation
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.2))
