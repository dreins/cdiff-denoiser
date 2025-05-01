import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import src.utils.settings as s
import random


args = s.configure_args()


def create_inference_dataloader(
    df,
    feature_columns=['E_channel', 'N_channel', 'Z_channel'],
    trace_name_column='trace_name',
    normalize=True
):
    max_value_dict = {}

    def stack_channels(row):
        event = np.stack([row['Z_channel'], row['E_channel'],
                         row['N_channel']], axis=0)  # (3, N)
        if normalize:
            src_max = np.abs(event).max()
            event = event / (src_max + 1e-8)
            max_value_dict[row[trace_name_column]] = src_max
        return torch.tensor(event, dtype=torch.float32)

    feature_tensors = torch.stack([stack_channels(row)
                                  for _, row in df.iterrows()])
    indices = torch.arange(len(df))
    targets = torch.zeros((len(df), 2), dtype=torch.float32)

    dataset = TensorDataset(indices, feature_tensors, targets)
    loader = DataLoader(dataset, shuffle=False)

    index_to_trace_name = {idx: name for idx,
                           name in enumerate(df[trace_name_column])}

    return loader, index_to_trace_name, max_value_dict


def generate_mock_trace(length=6000):
    return np.random.randn(length)


def generate_noise(length=6000, noise_std=(args.Range_RNF)):
    # Generate random trace data
    noisy_trace = np.random.randn(*args.Range_RNF) * 0.01

    return noisy_trace


def generate_noise_dataframe(
    num_traces,
    length,
    noise_std
):
    data = []
    for i in range(num_traces):
        trace = {
            'Z_channel': np.array([random.randint(args.Range_RNF[0], args.Range_RNF[1]) * 0.01 for _ in range(length)]),
            'E_channel': np.array([random.randint(args.Range_RNF[0], args.Range_RNF[1]) * 0.01 for _ in range(length)]),
            'N_channel': np.array([random.randint(args.Range_RNF[0], args.Range_RNF[1]) * 0.01 for _ in range(length)]),
            'trace_name': f"noise_trace_{i}"
        }
        data.append(trace)

    df = pd.DataFrame(data)
    return df


def validate_trace_lengths(df, feature_columns=['E_channel', 'N_channel', 'Z_channel'], trace_name_column='trace_name'):
    expected_length = None

    for i, row in df.iterrows():
        # Check within-row consistency (E, N, Z same length)
        channel_lengths = [len(row[col]) for col in feature_columns]
        if len(set(channel_lengths)) != 1:
            raise ValueError(
                f"Trace '{row[trace_name_column]}' has inconsistent channel lengths: {channel_lengths}"
            )

        # Check global consistency across all rows
        current_length = channel_lengths[0]
        if expected_length is None:
            expected_length = current_length
        elif current_length != expected_length:
            raise ValueError(
                f"Trace '{row[trace_name_column]}' has length {current_length}, expected {expected_length}"
            )


# Create mock dataset
mock_data = [
    {
        "Z_channel": generate_mock_trace(),
        "E_channel": generate_mock_trace(),
        "N_channel": generate_mock_trace(),
        "trace_name": "trace_001"
    },
    {
        "Z_channel": generate_mock_trace(),
        "E_channel": generate_mock_trace(),
        "N_channel": generate_mock_trace(),
        "trace_name": "trace_002"
    }
]

df = pd.DataFrame(mock_data)

# Run the function
loader, mapping, max_values = create_inference_dataloader(df)

# Check output
print("Index to Trace Mapping:", mapping)
print("Max Values for Normalization:", max_values)
