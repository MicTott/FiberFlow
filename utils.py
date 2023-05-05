import os
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import pandas as pd
import plotly.express as px

def load_data(file, is_event=False):
    key = f"{'events_' if is_event else ''}data_{file.name}"
    if key not in st.session_state:
        data = pd.read_csv(file)
        st.session_state[key] = data
    return st.session_state[key]

def save_plot(fig, subject_ID, plot_name,  scale_factor=4.17):
    output_dir = os.path.join("output", subject_ID)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{subject_ID}_{plot_name}.png")
    pio.write_image(fig, file_path, scale=scale_factor)

def export_averaged_trials(results, subject_ID):
    output_dir = os.path.join('output', subject_ID)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for event, averaged_data in results.items():
        output_path = os.path.join(output_dir, f'{subject_ID}_{event}_averaged_data.csv')
        averaged_data.to_csv(output_path, index=False)
        st.success(f"Averaged data for {event} has been exported to '{output_path}'.")

def export_new_data(df, subject_ID):
    output_dir = os.path.join('output', subject_ID)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f'{subject_ID}_proprocessed_data.csv')
    df.to_csv(output_path, index=False)
    st.success(f"New data has been exported to '{output_path}'.")