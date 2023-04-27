import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

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
            visualize_data(df)

            # Check if the plots are present in session state
            if st.session_state.preprocessed_plot is not None:
                st.plotly_chart(st.session_state['preprocessed_plot'], use_container_width=True)

            # add a button to preprocessed control data from signal data and plot the output
            if st.button("Preprocess Data and Plot", key='preprocess'):
                preprocess_df = preprocess_and_plot(df)
                st.session_state.preprocess_df = preprocess_df

            # Check if the output dataframe is in session state
            if st.session_state.output_df is not None:

                #  Add a checkbox to export the new data along with the old data
                preprocess_df = st.session_state.preprocess_df
                if st.checkbox("Export New Data"):
                    export_new_data(preprocess_df)

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
                            plot_events(df, events_df)
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
                                results_df = average_trials(df, events_df, pre_time, post_time)  

                            if st.session_state.averaged_trials is not None:
                                results_df = st.session_state.averaged_trials
                                if st.button(f"Export Results to CSV", key='export_results'):
                                    export_averaged_trials(results_df)

                else:
                    st.error("The uploaded events CSV file does not contain the required columns: 'events', 'start', and 'stop'.")
        else:
            st.error("The uploaded CSV file does not contain the required columns: 'time', 'signal', and 'control'.")





def load_data(file, is_event=False):
    key = f"{'events_' if is_event else ''}data_{file.name}"
    if key not in st.session_state:
        data = pd.read_csv(file)
        st.session_state[key] = data
    return st.session_state[key]

def export_averaged_trials(results):
    for event, result_df in results.items():
        file_name = f"{event}_Averaged_trials.csv"
        result_df.to_csv(file_name, index=False)
        st.success(f"{file_name} has been exported.")

def visualize_data(df):
    # Create line plots for signal and control data
    fig = px.line(df, x='time', y=['signal', 'control'], title="Signal and Control Data", labels={"value": "Data", "variable": "Type"})

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


def preprocess_and_plot(df, fs=30):

    time_seconds = df['time'].values
    GCaMP_raw = df['signal'].values
    ISOS_raw = df['control'].values

    GCaMP_prepro, GCaMP_denoised = preprocess(GCaMP_raw, time_seconds, fs)
    ISOS_prepro, ISOS_denoised = preprocess(ISOS_raw, time_seconds, fs)

    GCaMP_corrected = correct_motion(GCaMP_prepro, ISOS_prepro)
    GCaMP_dF_F = deltaF_F(GCaMP_corrected, GCaMP_denoised, fs)

    df['GCaMP_corrected'] = GCaMP_corrected
    df['Delta_F'] = GCaMP_dF_F

        # Store the 'output' column in the session state
    st.session_state['preprocessed'] = df['Delta_F']

    fig = px.line(df, x='time', y='Delta_F', title="Preprocessed Data", labels={"value": "Delta F/F", "variable": "Type"})

    # Store the plot in the session state
    st.session_state['preprocessed_plot'] = fig

    st.plotly_chart(fig, use_container_width=True)

    return df

def preprocess(raw, time_seconds, fs):
    '''This function denoises GCaMP or ISOS signals
    with a median ad lowpass filter. Then it fits a 4th order 
    polyonmial to the data subtracts the polyomial fit from the
    raw data.'''

    # Median and lowpass filter with filtfilt
    denoised_med = medfilt(raw, kernel_size=5)

    b,a = butter(2, 10, btype='low', fs=fs)
    denoised = filtfilt(b,a, denoised_med)

    # Fit 4th order polynomial to GCaMP signal and sutract
    coefs = np.polyfit(time_seconds, denoised, deg=4)
    polyfit_data = np.polyval(coefs, time_seconds)

    debleached = denoised - polyfit_data
    
    return debleached, denoised

def correct_motion(GCaMP_prepro, ISOS_prepro):
    '''This function takes preprocessed GCaMP and Isosbestic
    sigals and finds the linear fit, then estimates the 
    motion correction and substracts it from GCaMP.'''
    
    # find linear fit
    slope, intercept, r_value, p_value, std_err = linregress(x=ISOS_prepro, y=GCaMP_prepro)
    
    # estimate motion correction and subtract
    GCaMP_est_motion = intercept + slope * ISOS_prepro
    GCaMP_corrected = GCaMP_prepro - GCaMP_est_motion
    
    return GCaMP_corrected

def deltaF_F(GCaMP_corrected, denoised, fs):
    '''This function calculates the dF/F using the 
    denoised data and the motion corrected.'''
    
    b,a = butter(2, 0.001, btype='low', fs=fs)
    baseline_fluorescence = filtfilt(b,a, denoised, padtype='even')
    
    GCaMP_dF_F = GCaMP_corrected/baseline_fluorescence
    
    return GCaMP_dF_F


def export_new_data(df):
    # Export the new data along with the old data to a CSV file
    df.to_csv('new_data.csv', index=False)
    st.success("New data has been exported to 'new_data.csv'.")


def plot_events(df, events_df):
    # Create a line plot for the output data
    fig = px.line(df, x='time', y='Delta_F', title="Output Data with Events", labels={"value": "Data", "variable": "Type"})

    # Store the plot in the session state
    st.session_state['events_plot'] = fig

    # Add shaded regions for events
    colors = px.colors.qualitative.Plotly
    unique_events = events_df['events'].unique()
    event_color_map = {event: colors[i % len(colors)] for i, event in enumerate(unique_events)}

    # Add custom legend entries for unique events
    for event, color in event_color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                showlegend=True,
                name=event,
                legendgroup=event,
            )
        )

    for _, event_row in events_df.iterrows():
        event_color = event_color_map[event_row['events']]
        fig.add_shape(
            type="rect",
            x0=event_row['start'],
            x1=event_row['stop'],
            y0=0,
            y1=1,
            yref="paper",
            fillcolor=event_color,
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)



def average_trials(df, events_df, pre_time, post_time):
    unique_events = events_df['events'].unique()
    results = {}

    for event in unique_events:
        event_data = []

        for _, event_row in events_df[events_df['events'] == event].iterrows():
            start_index = np.where(df['time'] >= event_row['start'] - pre_time)[0][0]
            stop_index = np.where(df['time'] <= event_row['start'] + post_time)[0][-1]

            event_data.append(df['Delta_F'].iloc[start_index:stop_index + 1].values)

        # Calculate the average and SEM for the output data of each event
        avg_data = np.mean(event_data, axis=0)
        sem_data = np.std(event_data, axis=0, ddof=1) / np.sqrt(len(event_data))

        # Create a new DataFrame for the averaged output data, SEM, and each original trial
        avg_df = pd.DataFrame(avg_data, columns=['average'])
        sem_df = pd.DataFrame(sem_data, columns=['SEM'])
        trial_df = pd.DataFrame(np.column_stack(event_data), columns=[f'trial_{i+1}' for i in range(len(event_data))])

        result_df = pd.concat([df['time'][:len(avg_df)], avg_df, sem_df, trial_df], axis=1)
        results[event] = result_df

        # Plot the averaged output data with SEM as shaded error bars
        fig = go.Figure()

        # Store the plot in the session state
        st.session_state['average_trial_plot'] = fig

        fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['average'],
                                 mode='lines',
                                 name=f'{event} Mean',
                                 line=dict(width=2)))

        fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['average'] + result_df['SEM'],
                                 showlegend=False,
                                 mode='lines',
                                 line=dict(width=0)))

        fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['average'] - result_df['SEM'],
                                 showlegend=False,
                                 mode='lines',
                                 line=dict(width=0),
                                 fillcolor='rgba(68, 68, 68, 0.3)',
                                 fill='tonexty'))

        fig.update_layout(title=f'{event} Averaged Output Data with SEM',
                          xaxis_title='Time',
                          yaxis_title='Delta F/F',
                          legend_title='Events')

        st.plotly_chart(fig, use_container_width=True)

    # Store the results in the session state
    st.session_state['averaged_trials'] = results

    return results

if __name__ == "__main__":
    main()

