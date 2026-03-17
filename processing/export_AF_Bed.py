import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
from dateutil import tz
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import urllib3
import sys
import time  # Imported for request throttling sleep
from bed_info import *
# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def to_utc_epoch(time_str, tz_str="Asia/Shanghai"):
    local = tz.gettz(tz_str)
    utc = tz.gettz("UTC")
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
    dt = dt.replace(tzinfo=local)
    dt_utc = dt.astimezone(utc)
    return dt_utc.timestamp()


def _build_query(axis_name, unit, start_epoch, end_epoch):
    return f"""
SELECT "value" FROM "Z"
WHERE "location" = '{unit}'
AND time >= {int(start_epoch * 1e9)} AND time <= {int(end_epoch * 1e9)}
ORDER BY time ASC
"""


def _sleep_before_request(start_epoch, end_epoch):
    # Avoid overloading the server: sleep 1 second for every 60 seconds requested.
    duration_seconds = end_epoch - start_epoch
    if duration_seconds <= 0:
        return
    sleep_duration = int(duration_seconds // 60)
    if sleep_duration > 0:
        print(
            f"Requested time span is {duration_seconds:.2f}s; sleeping {sleep_duration}s to reduce server load..."
        )
        time.sleep(sleep_duration)


def _clean_and_interpolate(points, start_epoch, end_epoch, sampling_rate):
    if len(points) == 0:
        raise ValueError(
            "No data points were retrieved; processing cannot continue.")

    df = pd.DataFrame(points)
    df["time"] = pd.to_datetime(
        df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    if df.empty:
        raise ValueError(
            "Timestamp parsing failed; all time fields are invalid.")

    df.set_index("time", inplace=True)
    df["value"] = df["value"].astype(float)

    ideal_index = pd.date_range(
        start=datetime.fromtimestamp(start_epoch, tz=tz.gettz("UTC")),
        end=datetime.fromtimestamp(end_epoch, tz=tz.gettz("UTC")),
        freq=pd.Timedelta(seconds=1 / sampling_rate),
    )
    df_resampled = df.reindex(ideal_index)
    df_resampled["value"] = df_resampled["value"].interpolate(method="linear")
    df_resampled["value"] = df_resampled["value"].bfill()
    df_resampled["value"] = df_resampled["value"].ffill()
    return df_resampled["value"].to_numpy()


def _build_zero_signal(num_points):
    return np.zeros(num_points, dtype=float)


def _query_points_with_timeout(client, query, timeout_seconds=15):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.query, query)
        result = future.result(timeout=timeout_seconds)
    return list(result.get_points())


def _build_timestamps_array(start_epoch, end_epoch, sampling_rate):
    ideal_index = pd.date_range(
        start=datetime.fromtimestamp(start_epoch, tz=tz.gettz("UTC")),
        end=datetime.fromtimestamp(end_epoch, tz=tz.gettz("UTC")),
        freq=pd.Timedelta(seconds=1 / sampling_rate),
    )

    # Save Unix epoch nanoseconds (UTC, int64) so pd.to_datetime can parse directly.
    return ideal_index.asi8.copy()


def get_bsg_3axis_readings(
    bed_address_info,
    start_time_str,
    end_time_str,
    sampling_rate=100,
    chunk_duration=10,
    compression_factor=1,
    tz_str="Asia/Shanghai",
):
    """Input bed address info and return 3-axis BSG data.

    bed_address_info must include keys: bsg_x, bsg_y, bsg_z
    Returns: {"x": np.ndarray, "y": np.ndarray, "z": np.ndarray, "timestamps": np.ndarray}
    Array shape per axis: (num_points,)
    timestamps shape: (num_points,), UTC Unix epoch in nanoseconds (int64)
    """
    required_keys = {"bsg_x", "bsg_y", "bsg_z"}
    missing_keys = required_keys - set(bed_address_info.keys())
    if missing_keys:
        raise ValueError(
            f"bed_address_info is missing required keys: {sorted(missing_keys)}")

    start_epoch = to_utc_epoch(start_time_str, tz_str=tz_str)
    end_epoch = to_utc_epoch(end_time_str, tz_str=tz_str)
    if end_epoch <= start_epoch:
        raise ValueError("end_time_str must be later than start_time_str")

    client = InfluxDBClient(
        host="sensorweb.us",
        port=8086,
        username="algtest",
        password="sensorweb711",
        database="shake",
        ssl=True,
        verify_ssl=False,
    )

    _sleep_before_request(start_epoch, end_epoch)

    timestamps = _build_timestamps_array(
        start_epoch,
        end_epoch,
        sampling_rate=sampling_rate,
    )
    expected_points = len(timestamps)

    axis_map = {"x": "X", "y": "Y", "z": "Z"}
    axis_chunks = {}

    for axis_key in ("x", "y", "z"):
        axis_name = axis_map[axis_key]
        unit = bed_address_info[f"bsg_{axis_key}"]
        query = _build_query(axis_name, unit, start_epoch, end_epoch)

        print(f"Querying InfluxDB for axis {axis_name}, device {unit}...")
        try:
            points = _query_points_with_timeout(
                client, query, timeout_seconds=30)
        except FutureTimeoutError:
            print(
                f"\033[91mAxis {axis_name}: query waited more than 30s, filling this axis with zeros and continuing.\033[0m"
            )
            axis_chunks[axis_key] = _build_zero_signal(expected_points)
            continue
        print(f"Axis {axis_name}: retrieved {len(points)} raw points.")

        values = _clean_and_interpolate(
            points, start_epoch, end_epoch, sampling_rate)
        if len(values) != expected_points:
            raise ValueError(
                f"Axis {axis_name}: signal length ({len(values)}) does not match timestamps length ({expected_points})."
            )
        print(f"Axis {axis_name}: interpolated series length = {len(values)}")

        axis_chunks[axis_key] = values
        print(f"Axis {axis_name}: final signal length = {len(values)}")

    client.close()
    axis_chunks["timestamps"] = timestamps
    return axis_chunks


if __name__ == "__main__":
    start_time_str = "2026-3-16T11:05:40"
    end_time_str = "2026-3-16T11:09:08"
    status = "chirp_mat_2weights_"
    status = "sub_3_d15_"
    selected_bed = bed_AF_ID_30_up  # Change this to select different bed/device
    selected_bed = bed_AF_ID_30_compare
    selected_bed = bed_AF_ID_18_mid
    # selected_bed = bed_AF_ID_on
    output_filename = f"./data/{selected_bed['SID']}_{status}.npy"
    # try:
    bsg_3axis = get_bsg_3axis_readings(
        bed_address_info=selected_bed,
        start_time_str=start_time_str,
        end_time_str=end_time_str,
        sampling_rate=100,
        chunk_duration=10,
        compression_factor=1,
    )

    # Save as [X, Y, Z] stacked array with shape: (3, num_points)
    stacked_3axis = np.stack(
        [bsg_3axis["x"], bsg_3axis["y"], bsg_3axis["z"]], axis=0)
    payload = {
        "data": stacked_3axis,
        "timestamps": bsg_3axis["timestamps"],
    }
    np.save(output_filename, payload)
    print(
        f"BSG payload saved to {output_filename}, data shape: {stacked_3axis.shape}, timestamps shape: {bsg_3axis['timestamps'].shape}")
    print("Done.")
    # except Exception as e:
    #     print(f"Processing failed: {e}")
    #     sys.exit(1)
