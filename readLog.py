import re
import sys

def parse_log(input_file, output_file):
    results = []
    current_run = None
    max_process_time = 0

    # 匹配运行对象和 `total_process_time` 的正则表达式
    run_pattern = re.compile(r"sudo\s+mpirun.*?--map-by socket\s+(.*)")
    process_time_pattern = re.compile(r"total_process_time\s+=([0-9.]+)\(s\)")

    try:
        with open(input_file, "r") as log_file:
            for line in log_file:
                # 检测是否是新的运行对象
                run_match = run_pattern.search(line)
                if run_match:
                    # 如果有当前运行对象，记录结果
                    if current_run:
                        results.append((current_run, max_process_time))
                    # 更新新的运行对象
                    current_run = run_match.group(1).strip()
                    max_process_time = 0  # 重置最大时间

                # 检测 total_process_time
                process_time_match = process_time_pattern.search(line)
                if process_time_match:
                    process_time = float(process_time_match.group(1))
                    max_process_time = max(max_process_time, process_time)

        # 记录最后一个运行对象的结果
        if current_run:
            results.append((current_run, max_process_time))

        # 将结果写入输出文件
        with open(output_file, "w") as out_file:
            for run, max_time in results:
                out_file.write(f"Run: {run}\nMax Total Process Time: {max_time}(s)\n\n")

        print(f"Results successfully written to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 主程序入口
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 readLog.py <input_log_file> <output_file>")
    else:
        input_log_file = sys.argv[1]
        output_file = sys.argv[2]
        parse_log(input_log_file, output_file)
