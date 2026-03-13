import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
from dateutil import tz
from datetime import datetime, timedelta
import urllib3
import sys
import time  # 新增：引入 time 库以使用 sleep 功能

# 关闭 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置参数
unit = "cc:8d:a2:e8:d0:18"  # 替换为您的设备 MAC 地址
start_time_str = "2026-03-10T14:15:50"
end_time_str = "2026-03-10T14:16:21"
sampling_rate = 100  # 采样率 100Hz
chunk_duration = 10  # 每个数据块持续时间为10秒
compression_factor = 1  # 压缩因子


# 将时间字符串转换为 UTC 时间戳
def to_utc_epoch(time_str, tz_str="Asia/Shanghai"):
    local = tz.gettz(tz_str)
    utc = tz.gettz("UTC")
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
    dt = dt.replace(tzinfo=local)
    dt_utc = dt.astimezone(utc)
    return dt_utc.timestamp()


start_epoch = to_utc_epoch(start_time_str)
end_epoch = to_utc_epoch(end_time_str)

# 连接 InfluxDB
client = InfluxDBClient(
    host="sensorweb.us",
    port=8086,
    username="algtest",
    password="sensorweb711",
    database="shake",
    ssl=True,
    verify_ssl=False
)

# 查询数据
query = f"""
SELECT "value" FROM "Z"
WHERE "location" = '{unit}'
AND time >= {int(start_epoch * 1e9)} AND time <= {int(end_epoch * 1e9)}
ORDER BY time ASC
"""

# --- 新增：请求前延时模块 ---
# 为了防止因请求时间跨度过长而导致服务器崩溃，
# 我们在此加入一个延时逻辑：每请求一分钟的数据，就暂停一秒。
duration_seconds = end_epoch - start_epoch
if duration_seconds > 0:
    # 计算需要暂停的总秒数 (每60秒的数据暂停1秒)
    sleep_duration = int(duration_seconds // 60)
    if sleep_duration > 0:
        print(
            f"数据请求时长为 {duration_seconds:.2f} 秒，为防止服务器过载，将暂停 {sleep_duration} 秒...")
        time.sleep(sleep_duration)
# -----------------------------

print("正在查询 InfluxDB...")
try:
    result = client.query(query)
    points = list(result.get_points())
    print(f"成功从数据库获取到 {len(points)} 个原始数据点。")
except Exception as e:
    print(f"查询 InfluxDB 时出错: {e}")
    sys.exit(1)

# --- 数据清洗与插值模块 ---
if len(points) == 0:
    print("错误：未获取到任何数据点，无法继续处理。")
    sys.exit(1)
else:
    print("开始进行数据清洗与线性插值，以保证时间轴连续...")

    # 1. 将原始数据转换为 pandas DataFrame，便于时序操作
    df = pd.DataFrame(points)
    df['time'] = pd.to_datetime(
        df['time'], format='ISO8601', utc=True, errors='coerce')
    df = df.dropna(subset=['time'])
    if df.empty:
        print("错误：时间戳解析失败，所有 time 字段都无效。")
        sys.exit(1)
    df.set_index('time', inplace=True)
    df['value'] = df['value'].astype(float)

    # 2. 创建一个完整、均匀的“理想时间索引”
    ideal_index = pd.date_range(
        start=datetime.fromtimestamp(start_epoch, tz=tz.gettz('UTC')),
        end=datetime.fromtimestamp(end_epoch, tz=tz.gettz('UTC')),
        freq=pd.Timedelta(seconds=1 / sampling_rate)
    )

    # 3. 将真实数据对齐到理想时间索引上 (缺失点填充为 NaN)
    df_resampled = df.reindex(ideal_index)

    # 4. 使用线性插值填充所有缺失值 (NaN)
    df_resampled['value'] = df_resampled['value'].interpolate(method='linear')

    # 5. 处理边界情况 (开头或末尾的 NaN)
    df_resampled['value'] = df_resampled['value'].bfill()
    df_resampled['value'] = df_resampled['value'].ffill()

    # 6. 提取出干净、完整的值数组
    values = df_resampled['value'].to_numpy()
    print(f"数据插值完成，现在我们有了一个长度为 {len(values)} 的、时间连续的数据序列。")

# 将数据分块
chunk_size = sampling_rate * chunk_duration
num_chunks = len(values) // chunk_size

if num_chunks == 0:
    print(
        f"错误：获取到的数据点 ({len(values)}) 不足以构成一个完整的 {chunk_duration} 秒数据块 (需要 {chunk_size} 个点)。")
    sys.exit(1)

# 仅使用完整的块进行处理
data_chunks_raw = np.array(
    values[:num_chunks * chunk_size]).reshape((num_chunks, chunk_size))
print(f"数据已整理为 {num_chunks} 个数据块，每个块包含 {chunk_size} 个数据点。")

# --- 应用围绕平均值的纵向压缩 ---
print(f"正在对数据进行围绕平均值的纵向压缩，因子为 {compression_factor}...")

# 1. 计算每个 chunk 的平均值 (基线)
mean_per_chunk = np.mean(data_chunks_raw, axis=1, keepdims=True)
print(f"计算得到各数据块的平均值(基线)。")

# 2. 数据中心化 (减去平均值)
centered_data = data_chunks_raw - mean_per_chunk
print("数据已中心化处理 (原始数据 - 平均值)。")

# 3. 压缩波动部分
scaled_variation = centered_data * compression_factor
print(f"中心化后的数据已乘以压缩因子 {compression_factor}。")

# 4. 恢复基线 (加上原始平均值)
compressed_data_chunks = scaled_variation + mean_per_chunk
print("已将压缩后的波动加回到原始基线。")

# 保存为 .npy 文件
# output_filename = f"raw_signal_before_{start_time_str.replace(':', '')}_{end_time_str.replace(':', '')}.npy"
output_filename = "test.npy"
try:
    np.save(output_filename, compressed_data_chunks)
    print(f"围绕平均值压缩后的数据已保存为 {output_filename}")
except Exception as e:
    print(f"保存 .npy 文件时出错: {e}")
    sys.exit(1)

print("处理完成。")
