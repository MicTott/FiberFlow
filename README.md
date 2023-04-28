# Fiber photometry Visualization and analysis
This Streamlit application allows users to visualize, preprocess, and analyze calcium imaging data by uploading CSV files. The application is designed to handle both raw data and event data, providing an interactive interface for visualizing and exporting the results.

[streamlit-fiberflow-2023-04-28-11-04-63.webm](https://user-images.githubusercontent.com/32395683/235192757-a65ad1d9-6ab4-4a2b-be9d-25f1099925e1.webm)

## Features
- Upload raw data CSV file containing 'time', 'signal', and 'control' columns.
- Preprocess the data by applying denoising, debleaching, and motion correction.
- Calculate the delta F/F for the preprocessed data.
- Visualize the preprocessed data using line plots.
- Upload event data CSV file containing 'events', 'start', and 'stop' columns.
- Overlay events on the preprocessed data plot with shaded regions.
- Average output data across trials based on events and specified time ranges.
- Automatically exports all generated figures.
- Export preprocessed data, averaged trials, and results to CSV files.

## Installation
To set up and run the application on your local machine, follow these steps:

1. Clone the repository to your local machine.
```bash
git clone https://github.com/mictott/FiberFlow.git
```

2. Change into the repository directory.

```bash
cd /your/path/to/FiberFlow
```

3. Create a virtual environment and activate it (recommended).
```bash
conda create -n fiberflow python=3
conda activate fiberflow
```

4. Install the required packages.
```
pip install -r requirements.txt
```

5. Run the Streamlit application.
```bash
streamlit run fiberflow.py
```

Now, the application should be running on your local machine. Open your web browser and go to the provided URL (usually http://localhost:8501).


## Usage
1. Upload a CSV file containing raw data with 'time', 'signal', and 'control' columns using the file uploader.
2. The application will preprocess the data and display a line plot of the preprocessed data.
3. (Optional) If you have event data, upload a CSV file containing 'events', 'start', and 'stop' columns using the second file uploader.
4. (Optional) If event data is uploaded, click the "Plot Events" button to overlay events on the preprocessed data plot.
5. (Optional) If event data is uploaded, input the desired time ranges before and after the event start, then click the "Average Trials and Plot Output" button to display the averaged trials.
6. Use the checkboxes and buttons to export preprocessed data, averaged trials, and results to CSV files.
7. 
Enjoy exploring your calcium imaging data!

## Future plans

- [ ] Time zero for averaged trial data should occur at time of the event. 
- [ ] Add a box to add subject name, include in output file names.
- [ ] Add a checkbox for exporting high quality images. 
- [ ] Have outfiles save to new "output" folder.
- [ ] Maybe one day: add batch processing
