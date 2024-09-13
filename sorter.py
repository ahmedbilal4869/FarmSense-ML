import pandas as pd

# Define the file paths
input_file = 'data/cvb_in_ava_format/ava_train_set.csv'
output_file = 'data/cvb_in_ava_format/sorted_train_annotations_across_seconds.csv'

# Define column names
column_names = [
    'Video Name', 'Position (Seconds)', 'Bounding Box (x1)', 'Bounding Box (y1)',
    'Bounding Box (x2)', 'Bounding Box (y2)', 'Behavior Category', 'Cow ID'
]

# Load the CSV file
df = pd.read_csv(input_file, header=None, names=column_names, dtype=str, low_memory=False)

# Convert numeric columns to appropriate data types
df['Position (Seconds)'] = pd.to_numeric(df['Position (Seconds)'], errors='coerce')
df['Bounding Box (x1)'] = pd.to_numeric(df['Bounding Box (x1)'], errors='coerce')
df['Bounding Box (y1)'] = pd.to_numeric(df['Bounding Box (y1)'], errors='coerce')
df['Bounding Box (x2)'] = pd.to_numeric(df['Bounding Box (x2)'], errors='coerce')
df['Bounding Box (y2)'] = pd.to_numeric(df['Bounding Box (y2)'], errors='coerce')
df['Behavior Category'] = pd.to_numeric(df['Behavior Category'], errors='coerce')
df['Cow ID'] = pd.to_numeric(df['Cow ID'], errors='coerce')

# Ensure that columns are properly sorted and data is in the right format
df.sort_values(by=['Video Name', 'Position (Seconds)', 'Cow ID'], inplace=True)

# Group by Video Name and Cow ID
grouped = df.groupby(['Video Name', 'Cow ID'])

# List to hold the processed data
processed_data = []

# Process each group
for (video_name, cow_id), group in grouped:
    # Sort by Position (Seconds)
    group_sorted = group.sort_values(by=['Position (Seconds)'])
    
    # Append the sorted data for the current cow ID
    processed_data.append(group_sorted)

# Combine all processed data into a single DataFrame
result_df = pd.concat(processed_data)

# Save the result to a new CSV file
result_df.to_csv(output_file, index=False)
