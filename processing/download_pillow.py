# -*- coding: utf-8 -*-
"""
=================================================================================================
InfluxDB Data Fetcher with SSH Tunnel (Auto-Fill Support)
版本: 1.5

描述:
1. 通过 SSH 隧道连接远程 InfluxDB 2.x 数据库。
2. 查询三轴加速度计数据。
3. 若指定时间内无数据，则以 62.5Hz (16ms 步长) 自动生成全零占位数据。

运行要求:
- influxdb-client-python
- pandas
- numpy
- sshtunnel
=================================================================================================
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from sshtunnel import SSHTunnelForwarder

# --- 1. SSH 连接配置 ---
SSH_HOST = '61.160.108.42'
SSH_PORT = 22
SSH_USERNAME = 'user'
SSH_PASSWORD = 'njulab307'

# --- 2. InfluxDB 配置 ---
INFLUXDB_TOKEN = "2BhmEDq_zJ6S3fI769boM5h0afRuqZ-Q2ZOwlX2xYnBaehwdFy_cshb4ydkJXwNdR9TkDJxW1_BLjMFR8QOpcA=="
INFLUXDB_ORG = "icu"
INFLUXDB_BUCKET = "vital_signs"
INFLUXDB_TIMEOUT_MS = 300_000

REMOTE_INFLUX_HOST = 'localhost'
REMOTE_INFLUX_PORT = 8086

# --- 3. 查询参数配置 ---
DEVICE_ID_TO_FETCH = "device1"
    # start_time_str = "2026-3-16T11:00:09"
    # end_time_str = "2026-3-16T11:05:25"
START_TIME_LOCAL = "2026-3-16 11:11:30"
END_TIME_LOCAL = "2026-3-16 11:14:28"
# status = 'chirp_womat_'
status = 'sub_3_d30_'
LOCAL_TIMEZONE = "Asia/Shanghai"
OUTPUT_DIR = r"./data/"

# 采样频率配置
TARGET_FREQ_HZ = 62
TIME_STEP_MS = 1000 / TARGET_FREQ_HZ  # 16.0 ms


def generate_placeholder_data(start_dt, end_dt):
    """
    当数据库无数据时，生成 62.5Hz 的空白占位数据
    """
    print(f"生成占位数据中: {start_dt} -> {end_dt} (频率: {TARGET_FREQ_HZ}Hz)")

    # 生成时间序列 (freq='16ms' 对应 62.5Hz)
    # 使用 '16ms' 作为频率别名
    time_index = pd.date_range(
        start=start_dt, end=end_dt, freq=f'{TIME_STEP_MS}ms')

    # 构造 DataFrame
    placeholder_df = pd.DataFrame({
        '_time': time_index,
        'x_axis': 0.0,
        'y_axis': 0.0,
        'z_axis': 0.0
    })

    return placeholder_df


def fetch_accelerometer_data(device_id, start_time_utc, end_time_utc, current_url):
    """
    通过 SSH 隧道查询数据
    """
    print(f"正在初始化 InfluxDB 客户端 (隧道地址: {current_url})...")

    with InfluxDBClient(url=current_url,
                        token=INFLUXDB_TOKEN,
                        org=INFLUXDB_ORG,
                        timeout=INFLUXDB_TIMEOUT_MS) as client:
        try:
            if not client.ping():
                print(f"无法连接到 InfluxDB，请检查隧道状态。")
                return None

            query_api = client.query_api()

            flux_query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                  |> range(start: {start_time_utc}, stop: {end_time_utc})
                  |> filter(fn: (r) => r["_measurement"] == "accelerometer_raw")
                  |> filter(fn: (r) => r["device_id"] == "{device_id}")
                  |> filter(fn: (r) => r["_field"] == "x_axis" or r["_field"] == "y_axis" or r["_field"] == "z_axis")
                  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                  |> keep(columns: ["_time", "x_axis", "y_axis", "z_axis"])
                  |> sort(columns: ["_time"], desc: false)
            '''
            print(f"正在从数据库查询设备 '{device_id}' 的原始数据...")

            result_df = query_api.query_data_frame(query=flux_query)

            if result_df is None or (isinstance(result_df, pd.DataFrame) and result_df.empty):
                print("数据库中未查询到相关数据。")
                return None

            # 确保列顺序和名称正确
            final_df = result_df[['_time', 'x_axis', 'y_axis', 'z_axis']]
            print(f"从数据库获取到 {len(final_df)} 条数据。")
            return final_df

        except Exception as e:
            print(f"查询过程中出错: {e}")
            return None


def save_data(df, device_id, start_time_str):
    """
    将 DataFrame 转换为 NumPy 格式并保存
    """
    if df is None or df.empty:
        print("跳过保存：数据为空。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 构造文件名
    time_part = start_time_str.replace(" ", "_").replace(":", "-")
    # base_filename = f"device_{device_id}_{time_part}"
    base_filename = f"device_{device_id}_{status}"
    npy_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}.npy")

    try:
        df_npy = df.copy()

        # 将时间戳转换为纳秒级 Unix 时间戳 (int64)
        # 兼容数据库返回的 Timestamp 和 pd.date_range 生成的 Timestamp
        df_npy['timestamp_ns'] = df_npy['_time'].apply(
            lambda t: int(t.timestamp() * 1e9))

        # 提取数组：[timestamp_ns, x, y, z]
        np_arr = df_npy[['timestamp_ns', 'x_axis',
                         'y_axis', 'z_axis']].to_numpy(dtype=np.float64)

        np.save(npy_filepath, np_arr)
        print(f"NPY 文件保存成功: {npy_filepath}")
        print(f"数组形状: {np_arr.shape} (包含时间戳列)")

    except Exception as e:
        print(f"保存 NPY 文件失败: {e}")


if __name__ == '__main__':
    # 1. 准备时间参数
    try:
        start_dt = pd.to_datetime(START_TIME_LOCAL).tz_localize(LOCAL_TIMEZONE)
        end_dt = pd.to_datetime(END_TIME_LOCAL).tz_localize(LOCAL_TIMEZONE)

        # InfluxDB 需要 UTC 时间字符串
        start_utc = start_dt.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        end_utc = end_dt.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"时间配置解析错误: {e}")
        exit(1)

    print(f"=== 任务启动: {DEVICE_ID_TO_FETCH} ===")

    # 2. 建立 SSH 隧道并执行任务
    try:
        with SSHTunnelForwarder(
            (SSH_HOST, SSH_PORT),
            ssh_username=SSH_USERNAME,
            ssh_password=SSH_PASSWORD,
            remote_bind_address=(REMOTE_INFLUX_HOST, REMOTE_INFLUX_PORT)
        ) as server:

            local_port = server.local_bind_port
            tunnel_url = f"http://localhost:{local_port}"

            print(f"SSH 隧道连接成功 -> {tunnel_url}")

            # 执行数据库查询
            raw_data_df = fetch_accelerometer_data(
                device_id=DEVICE_ID_TO_FETCH,
                start_time_utc=start_utc,
                end_time_utc=end_utc,
                current_url=tunnel_url
            )

            # 3. 结果处理：如果为空，则生成 62.5Hz 占位数据
            if raw_data_df is None or raw_data_df.empty:
                print(">>> 检测到空数据，正在触发自动补全机制...")
                raw_data_df = generate_placeholder_data(start_dt, end_dt)

            # 保存数据
            save_data(raw_data_df, DEVICE_ID_TO_FETCH, START_TIME_LOCAL)

    except Exception as e:
        print(f"执行过程中发生严重错误: {e}")

    print("=== 流程结束 ===")
