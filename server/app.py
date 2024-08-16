from flask import Flask, send_from_directory, jsonify, g, request
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import logging
import argparse
import time
from sklearn.preprocessing import MinMaxScaler
import joblib

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run the Flask server.')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
parser.add_argument('-p', '--port', type=int, required=True, help='Port number to run the Flask server on')
args = parser.parse_args()

# 配置日志记录
if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################
app = Flask(__name__)

# 运行脚本的路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 记录请求到回复的时间
@app.before_request
def start_timer():
    g.start_time = time.time()

@app.after_request
def log_duration(response):
    if 'start_time' in g:
        elapsed_time = time.time() - g.start_time
        logger.info(f"Request to {request.path} took {elapsed_time:.2f} seconds")
    return response

# 初始化请求计数器
request_counter = defaultdict(int)

@app.route('/api/predictLocation/', methods=['POST'])
def predictLocation():
    """
    处理POST请求，根据日志文件中RSSI等数据预测衣服在商店内的具体货架位置，并将结果保存为CSV文件。

    参数:
        无直接参数。函数从请求的JSON正文中读取：
        - abs_read_path (str): 日志文件的绝对路径。
        - abs_save_path (str): 结果保存的绝对路径。
        - abs_style_path (str): epcHeader和款式对应的csv表格的绝对路径。

    返回:
        JSON响应，包含操作状态信息：
        - 200 OK: 如果文件已存在并被更新。
        - 201 Created: 如果结果文件在服务器上被首次创建。
        - 400 Bad Request: 如果请求中未提供必需的路径。
        - 404 Not Found: 如果指定的读取路径不存在。
        - 500 Internal Server Error: 如果遇到未预期的错误。

    示例:
        curl -X POST http://localhost:8000/api/predictLocation/ \
            -H "Content-Type: application/json" \
            -d '{"abs_read_path": "/Users/liuhankang/Desktop/flaskProject/data/chenlie/20240611/123",
                "abs_save_path": "/Users/liuhankang/Desktop/flaskProject/data/result",
                "abs_style_path": "/Users/liuhankang/Desktop/flaskProject/data/style.csv"}'
        -> 返回 {
        "files_read": [
            "chenliedata3.log", 
            "chenliedata2.log", 
            "chenliedata.log"
        ], 
        "message": "location output file created", 
        "path": "/Users/liuhankang/Desktop/flaskProject/data/result/chenlieresult.2024-06-11_task123.csv"
        }
    """
    ##################### LOG请求次数
    request_counter['predictLocation'] += 1
    logger.info(f"Predict location request {request_counter['predictLocation']}:")

    ##################### 读取json和检查路径
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "Invalid JSON data."}), 400
        abs_read_path = json_data.get('abs_read_path')
        abs_save_path = json_data.get('abs_save_path')
        abs_style_path = json_data.get('abs_style_path')
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return jsonify({"error": "Invalid JSON format."}), 400

    if not abs_read_path:
        return jsonify({"error": "Please provide absolute read path."}), 400
    if not abs_save_path:
        return jsonify({"error": "Please provide absolute save path."}), 400   
    if not abs_style_path:
        return jsonify({"error": "Please provide absolute style path."}), 400

    # 检查读取路径是否存在
    read_path_exists = os.path.exists(abs_read_path)
    logger.debug(f"Does the folder {abs_read_path} exist? {read_path_exists}")

    if not read_path_exists:
        return jsonify({"error": "The specified absolute read path does not exist"}), 404
    if not os.path.isdir(abs_read_path):
        return jsonify({"error": "The specified absolute read path is not a valid directory"}), 400

    style_path_exists = os.path.exists(abs_style_path)
    logger.debug(f"Does the csv file {abs_style_path} exist? {style_path_exists}")
    if not style_path_exists:
        return jsonify({"error": "The specified absolute style path does not exist"}), 404
    if not abs_style_path.endswith('.csv'):
        return jsonify({"error": "The specified absolute style path is not a CSV file"}), 400

    # 检查保存路径是否存在
    save_path_exists = os.path.exists(abs_save_path)
    logger.debug(f"Does the folder {abs_save_path} exist? {save_path_exists}")

    if not save_path_exists:
        try:
            os.makedirs(abs_save_path)
            logger.info(f"Directory {abs_save_path} created.")
        except Exception as e:
            logger.error(f"Unable to create the directory {abs_save_path}: {e}")
            return jsonify({"error": f"Unable to create the directory {abs_save_path}"}), 500


    # 定义设备和天线的数量
    num_devices = 20 # 设备用1-20数字代表
    num_antennas = 4 # 天线用0-3数字代表

    # 为每个天线的RSSI总和和检测计数生成列名
    result_columns = {
        'epc': 'str' 
    }
    for device in range(1, num_devices + 1):
        for antenna in range(num_antennas):
            rssi_sum_column = f'设备{device}天线{antenna}_rssi_sum'
            count_column = f'设备{device}天线{antenna}_count'
            result_columns[rssi_sum_column] = 'int'
            result_columns[count_column] = 'int'

    # 用于模型预测的DataFrame
    result_df = pd.DataFrame(columns=result_columns.keys()).astype(result_columns)

    ##################### 循环读取log文件
    files_read = []
    date = None
    for file_name in os.listdir(abs_read_path):
        if not("chenliedata" in file_name and file_name.endswith(".log")):
            continue

        full_file_path = os.path.join(abs_read_path, file_name)
        with open(full_file_path, "r") as file:
            log_content = file.readlines()

        # 准备处理的行
        rows = []
        for line in log_content:
            fields = line.strip().split()
            # 本行缺少内容
            if len(fields) < 8:
                continue
            if date is None:  # 存储数据的日期，用于输出文件命名
                date = fields[-2]
            time = fields[-2] + ' ' + fields[-1]
            row = fields[:-2] + [time]
            rows.append(row)
        files_read.append(file_name)
        try:
            # 使用新读取的行更新DataFrame
            update_dataframe(rows, result_df)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        logger.debug(f"{file_name} RSSI统计完成")
        
    if len(files_read) == 0:
        return jsonify({"error": f"No chenlie log files were in {abs_read_path}."}), 400
    
    # 通过RSSI总和和总数算出每个检测点RSSI均值
    for column in result_df.columns:
        if 'rssi_sum' in column:
            count_column = column.replace('rssi_sum', 'count')
            mean_column = column.replace('rssi_sum', 'rssi_mean')
            result_df[mean_column] = result_df.apply(
                lambda row: row[column] / row[count_column] if row[count_column] != 0 else -100,
                axis=1
            )

    # 删除RSSI总和列
    columns_to_drop = [col for col in result_df.columns if 'rssi_sum' in col]
    result_df.drop(columns=columns_to_drop, inplace=True)

    # 获取DataFrame中的 'epc' 列，用于输出表格和位置信息对应
    epcs = result_df['epc']
    result_df = result_df.drop(columns=['epc'])

    # 获取所有以 'count' 结尾的列并将其归一化
    scaler = MinMaxScaler()
    count_columns = [col for col in result_df.columns if col.endswith('_count')]
    result_df[count_columns] = scaler.fit_transform(result_df[count_columns])

    ##################### 导入模型 （TODO：替换傻瓜模型）
    model_path = os.path.join(script_dir, 'dummy_location_classifier.joblib')
    model = joblib.load(model_path)
    predictions = model.predict(result_df)
    output_df = pd.DataFrame({
        'epcs': epcs,
        'location': predictions
    })

    taskID = os.path.basename(os.path.normpath(abs_read_path))
    file_path = os.path.join(abs_save_path, f"chenlieresult.{date}_task{taskID}.csv")

    # 检查输出文件是否存在
    if os.path.exists(file_path):
        response_status = 200
        message = "location output file updated"
    else:
        response_status = 201
        message = "location output file created"

    # 输出文件保存格式为.csv
    output_df.to_csv(file_path)
    logger.info(f'{date}: TaskID{taskID}位置预测完成')
    return jsonify({
        "message": message,
        "path": file_path,
        "files_read": files_read
    }), response_status

                   
def update_dataframe(rows, df):
    """
    更新DataFrame。

    参数:
    rows (list): 包含要处理的行数据的列表。
    df (DataFrame): 要更新的DataFrame。

    返回:
    DataFrame: 更新后的DataFrame。

    描述:
    此函数遍历提供的行数据，根据EPC、天线和设备ID对DataFrame进行更新。
    每一行数据会更新相应EPC的RSSI总和和计数，并将其批量应用到DataFrame中。
    如果EPC首次出现，则在DataFrame中创建新记录。
    """
    # 批量更新的临时存储
    updates = {}

    for row in rows:
        epc, antenna, deviceID = row[0], row[1], row[2]
        rssi_sum_col = f'设备{deviceID}天线{antenna}_rssi_sum'
        count_col = f'设备{deviceID}天线{antenna}_count'

        # 检查必要的列是否存在
        if rssi_sum_col not in df.columns or count_col not in df.columns:
            raise ValueError(f"Missing columns in DataFrame: {rssi_sum_col} or {count_col}")

        if epc not in updates:
            updates[epc] = {col: 0 for col in df.columns}
            updates[epc]['epc'] = epc

        # 累加RSSI总和并增加计数
        updates[epc][rssi_sum_col] += int(row[3])
        updates[epc][count_col] += 1

    for epc, update_data in updates.items():
        idx = df[df['epc'] == epc].index
        if idx.empty:
            # 如果EPC不存在于DataFrame中，则添加新行
            df.loc[len(df)] = update_data 
        else:
            # 如果EPC已存在，则更新相应的RSSI总和和计数
            idx = idx[0]
            for col, value in update_data.items():
                if col != 'epc':
                    df.at[idx, col] += value


@app.route('/api/detectTryOns/', methods=['POST'])
def detectTryOns():
    """
    处理POST请求，根据日志文件中RSSI的波动检测试穿事件，并将结果保存为CSV文件。

    参数:
        无直接参数。函数从请求的JSON正文中读取：
        - abs_read_path (str): 日志文件的绝对路径。
        - abs_save_path (str): 结果保存的绝对路径。

    返回:
        JSON响应，包含操作状态信息：
        - 200 OK: 如果文件已存在并被更新。
        - 201 Created: 如果结果文件在服务器上被首次创建。
        - 400 Bad Request: 如果请求中未提供必需的路径。
        - 404 Not Found: 如果指定的读取路径不存在。
        - 500 Internal Server Error: 如果遇到未预期的错误。

    示例:
        curl -X POST http://localhost:8000/api/detectTryOns/ \
            -H "Content-Type: application/json" \
            -d '{"abs_read_path": "/Users/liuhankang/Desktop/flaskProject/data/shiyi/20240611",
                "abs_save_path": "/Users/liuhankang/Desktop/flaskProject/data/result"}'
        -> 返回 {
        "files_read": [
            "shiyidata.2024-06-11.1.log", 
            "shiyidata.2024-06-11.0.log"
        ], 
        "message": "tryon output file created", 
        "path": "/Users/liuhankang/Desktop/flaskProject/data/result/shiyiresult.2024-06-11.csv"
        }
    """

    ##################### LOG请求次数
    request_counter['detectTryons'] += 1
    logger.info(f"Detect tryons request {request_counter['detectTryons']}:")

    ##################### 算法参数 （TODO：优化调整）
    window_size = 5
    threshold = 18
    time_window = 120
    min_tryons = 20

    ##################### 读取json和检查路径
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "Invalid JSON data."}), 400
        abs_read_path = json_data.get('abs_read_path')
        abs_save_path = json_data.get('abs_save_path')
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return jsonify({"error": "Invalid JSON format."}), 400

    if not abs_read_path:
        return jsonify({"error": "Please provide absolute read path."}), 400

    if not abs_save_path:
        return jsonify({"error": "Please provide absolute save path."}), 400

    # 检查读取路径是否存在
    read_path_exists = os.path.exists(abs_read_path)
    logger.debug(f"Does the folder {abs_read_path} exist? {read_path_exists}")

    if not read_path_exists:
        return jsonify({"error": "The specified absolute read path does not exist"}), 404
    if not os.path.isdir(abs_read_path):
        return jsonify({"error": "The specified absolute read path is not a valid directory"}), 400

    # 检查保存路径是否存在
    save_path_exists = os.path.exists(abs_save_path)
    logger.debug(f"Does the folder {abs_save_path} exist? {save_path_exists}")

    if not save_path_exists:
        try:
            os.makedirs(abs_save_path)
            logger.info(f"Directory {abs_save_path} created.")
        except Exception as e:
            logger.error(f"Unable to create the directory {abs_save_path}: {e}")
            return jsonify({"error": f"Unable to create the directory {abs_save_path}"}), 500


    ##################### 定义读取的log文件和输出的csv文件的列名
    log_columns = {
        'epc': 'str',               # epc字符串列
        '天线号': 'str',             # 天线号字符串列
        '设备ID': 'str',            # 设备ID字符串列
        'RSSI': 'int',              # RSSI整数列
        '频率': 'int',              # 频率整数列
        '开始相位': 'float',         # 开始相位浮点列
        '结束相位': 'float',         # 结束相位浮点列
        '时间': 'datetime64[ns]'    # 时间日期时间列
    }

    output_columns = {
        'epc': 'str',                # 字符串列
        '试穿时间': 'datetime64[ns]',    # 时间列
        '天线号+设备ID': 'str'        # 天线号和设备ID组合字符串列
    }
    output_df = pd.DataFrame(columns=output_columns.keys()).astype(output_columns)

    ##################### 循环读取log文件
    tryon_record = []
    files_read = []
    date = None
    for file_name in os.listdir(abs_read_path):
        if not("shiyidata" in file_name and file_name.endswith(".log")):
            continue

        full_file_path = os.path.join(abs_read_path, file_name)

        # 读取文件内容
        with open(full_file_path, "r") as file:
            log_content = file.readlines()

        rows = []
        for line in log_content:
            fields = line.strip().split()
            # 本行缺少内容
            if len(fields) < 8:
                continue
            if date is None:  # 存储数据的日期，用于输出文件命名
                date = fields[-2]
            time = fields[-2] + ' ' + fields[-1]
            row = fields[:-2] + [time]
            rows.append(row)
        files_read.append(file_name)
        df = pd.DataFrame(rows, columns=log_columns.keys())
        df = df.astype(log_columns)
        df["天线号+设备ID"] = df["天线号"] + '+' + df["设备ID"]

        # 转换"时间"列 为 datetime类型
        df["时间"] = pd.to_datetime(df["时间"])

        # 记录第一行的时间作为起始时间
        start_time = df["时间"].iloc[0]
        df = df.drop(columns=['天线号', '设备ID', '频率', '开始相位', '结束相位'])
        # 添加新列 "秒"，表示从起始时间开始的秒数差
        df["秒"] = (df["时间"] - start_time).dt.total_seconds()

        epcs = df['epc']
        # 获取每个EPC的出现次数
        epc_counts = epcs.value_counts()

        # 筛选出出现次数超过 min_tryons 次的EPC
        frequent_epcs = epc_counts[epc_counts > min_tryons].index.values

        # 将目标EPC id展开为一维数组
        target_epcs = frequent_epcs.flatten()

        # 遍历所有目标EPC id
        for target_epc in target_epcs:
            
            epc_data = df[df['epc'] == target_epc]
                
            unique_devices = epc_data['天线号+设备ID'].unique()

            # 遍历每个天线号+设备ID的唯一值
            for device in unique_devices:
                filtered_data = epc_data[epc_data['天线号+设备ID'] == device]

                # 提取RSSI和时间数据
                rssi_data = filtered_data['RSSI'].values
                time_data = filtered_data['秒'].values
                real_time_data = filtered_data['时间'].values
                
                
                # 将时间从ms转换为s
                time_data = time_data / 1000
                # 预分配一个数组来存储每个窗口的方差
                num_windows = len(rssi_data) - window_size + 1
                variances = np.zeros(num_windows)
                # 计算每个窗口的方差
                for j in range(num_windows):
                    window_data = rssi_data[j:j + window_size]
                    variances[j] = np.var(window_data, ddof=1)
                # 找到超过阈值的窗口索引
                exceed_indices = np.where(variances > threshold)[0]
                # 初始化试穿动作计数
                num_actions = 0
                in_action = False

                # 遍历每个时间点，判断窗口内是否有超出阈值的点
                for t in range(len(time_data)):
                    # 定义当前窗口的结束时间
                    window_end_time = time_data[t] + time_window
                    
                    # 找到窗口内的所有点
                    window_points = np.where((time_data >= time_data[t]) & (time_data <= window_end_time))[0]
                    # 找到窗口内超出阈值的点
                    points_in_window = np.intersect1d(window_points, exceed_indices)
                    if len(points_in_window) > 0:
                        if not in_action:
                            num_actions += 1
                            tryon_record.append({'epc': target_epc, 
                                                 '试穿时间': real_time_data[t] + np.timedelta64(int(np.floor(time_window / 2)), 's'), 
                                                 '天线号+设备ID': device})
                            in_action = True
                    else:
                        in_action = False
        logger.debug(f"{file_name} 试穿检测完成")
    if len(files_read) == 0:
        return jsonify({"error": f"No shiyi log files were in {abs_read_path}."}), 400
    
    output_df = pd.DataFrame(tryon_record)

    # 移除每个epc同一时间被多个天线检测到而导致的重复计数
    output_df = output_df.groupby('epc').apply(remove_excessive_counts)

    # 按时间排序
    output_df.sort_values(by='试穿时间', inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    file_path = os.path.join(abs_save_path, f"shiyiresult.{date}.csv")

    # 检查输出文件是否存在
    if os.path.exists(file_path):
        response_status = 200
        message = "tryon output file updated"
    else:
        response_status = 201
        message = "tryon output file created"

    # 输出文件保存格式为.csv
    output_df.to_csv(file_path)
    logger.info(f'{date}: 试穿检测完成')

        
    return jsonify({
        "message": message,
        "path": file_path,
        "files_read": files_read
    }), response_status


def remove_excessive_counts(group):
    """
    移除冗余计数。

    参数:
    group (DataFrame): 要处理的数据组。

    返回:
    DataFrame: 清理后的数据组。

    描述:
    对数据组按时间排序，仅保留第一行以及与上一保留行时间间隔达到指定秒数的行。
    例如，如果一条标签如果在1:00，1:30，和2:00被不同设备判定为试穿，间隔阈值为40s，
    那么只保留第一条读取并只视为一次试穿。
    """
    # 间隔阈值
    seconds_threshold = 120

    # 保留第一行以及与上一保留行时间间隔达到或超过指定秒数的行
    group = group.sort_values(by='试穿时间')
    keep = [True]
    last_kept_time = group['试穿时间'].iloc[0]
    for i in range(1, len(group)):
        curr_time = group['试穿时间'].iloc[i]
        if (curr_time - last_kept_time) >= pd.Timedelta(seconds=seconds_threshold):
            keep.append(True)
        else:
            keep.append(False)
        last_kept_time = curr_time  # 更新最后保留的时间
    return group[keep]


# 全局异常处理器，捕获并记录所有未处理的异常
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"未处理的异常: {e}")
    return jsonify({"error": "Server error"}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(port=args.port, threaded=True, debug=args.debug)
