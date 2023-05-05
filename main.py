# import dependencies
import streamlit as st
import pandas as pd
import plotly.express as px

# import FiberFlow functions
from data_processing import *
from utils import *


def reset_session_state():
    st.session_state.output_df = None
    st.session_state.events_df = None
    st.session_state.averaged_trials = None
    st.session_state.preprocessed = None
    st.session_state.preprocessed_plot = None
    st.session_state.events_plot = None
    st.session_state.average_trial_plot = None

def main():
    st.title("FiberFlow: Fiber Photometry Procesing, Visualization, and Analysis")

    # Add input for the subject name
    subject_ID = st.text_input("Enter the subject ID (e.g., Rat1):")

    # Upload the CSV file
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Reset session state if CSV file is erased
    if csv_file is None:
        reset_session_state()

    if csv_file is not None:
        # Load the CSV file into a pandas DataFrame
        df = load_data(csv_file)

        if set(['time', 'signal', 'control']).issubset(df.columns):
            # Reset the time array to start from 0
            df['time'] = df['time'] - df['time'].iloc[0]

            # Visualize the data using line plots
            visualize_data(df, subject_ID)

            # Check if the plots are present in session state
            if st.session_state.preprocessed_plot is not None:
                st.plotly_chart(st.session_state['preprocessed_plot'], use_container_width=True)

            # add a button to preprocessed control data from signal data and plot the output
            if st.button("Preprocess Data and Plot", key='preprocess'):
                preprocess_df = preprocess_and_plot(df, subject_ID)
                st.session_state.preprocess_df = preprocess_df

            # Check if the output dataframe is in session state
            if st.session_state.output_df is not None:

                #  Add a checkbox to export the new data along with the old data
                preprocess_df = st.session_state.preprocess_df
                if st.checkbox("Export Processed Data"):
                    export_new_data(preprocess_df, subject_ID)

            # Upload the events CSV file
            events_csv_file = st.file_uploader("Upload an events CSV file", type=["csv"])

            if events_csv_file is not None:
                # Load the events CSV file into a pandas DataFrame
                events_df = pd.read_csv(events_csv_file)

                # Check if the required columns are present in the events DataFrame
                if set(['events', 'start', 'stop']).issubset(events_df.columns):

                    with st.container():
                        
                        # Check if the output plot are present in session state
                        if st.session_state.events_plot is not None:
                            st.plotly_chart(st.session_state['events_plot'], use_container_width=True)

                        # Add a button to plot events with shaded regions only if output data is present
                        if st.button("Plot Events", key='plot_events'):
                            plot_events(df, events_df, subject_ID)
                            st.session_state.events_df = events_df

                        # Check if the events dataframe is in session state
                        if st.session_state.events_df is not None:
                            events_df = st.session_state.events_df

                            # Add a button to average trials and plot output
                            st.subheader("Average Output Data Across Trials")
                            pre_time = st.number_input("Seconds before event start:", value=0.0, step=0.1, key='pre_time')
                            post_time = st.number_input("Seconds after event start:", value=1.0, step=0.1, key='post_time')



                            # Check if the average trial plot are present in session state
                            if st.session_state.average_trial_plot is not None:
                                st.plotly_chart(st.session_state['average_trial_plot'], use_container_width=True)

                            if st.button("Average Trials and Plot Output", key='average_trials'):
                                results_df = average_trials(df, events_df, pre_time, post_time, subject_ID)  

                            if st.session_state.averaged_trials is not None:
                                results_df = st.session_state.averaged_trials
                                if st.button(f"Export Results to CSV", key='export_results'):
                                    export_averaged_trials(results_df, subject_ID)

                else:
                    st.error("The uploaded events CSV file does not contain the required columns: 'events', 'start', and 'stop'.")
        else:
            st.error("The uploaded CSV file does not contain the required columns: 'time', 'signal', and 'control'.")

if __name__ == "__main__":
    main()