#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STDK 发布限流测试工具
====================

向设备串口批量发送命令，同时把串口输出写入日志文件，结束后自动统计限流特征。

依赖：pyserial      pip install pyserial

常用示例
--------
  # 默认：70 次 button 1，间隔 0.3 秒，COM8（限流阈值 60 时用来触发）
  python tools/ratelimit_test.py

  # 自定义次数/周期/串口
  python tools/ratelimit_test.py --port COM8 --count 80 --interval 0.4

  # 只抓日志不发送
  python tools/ratelimit_test.py --count 0 --watch 30

  # 全量回显（默认只打印命中的关键行）
  python tools/ratelimit_test.py -v

  # 等设备进入云端在线状态再开始发
  python tools/ratelimit_test.py --wait-online
"""

import argparse
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None

# ---------------------------------------------------------------- 关键特征串
# key -> (模式, 说明)
PATTERNS = {
    "limit_local":   ("Publish rate limit: should wait",  "本地窗口超限（WARN）"),
    "limit_block":   ("Exceed rate limit. Can't send",    "发布被拦（WARN）"),
    "limit_err":     ("MQTT pub error(-206)",             "上层收到限流错误码"),
    "limit_app":     ("[rate limit] Remaining time",      "App 收到限流通知"),
    "limit_release": ("[rate limit] Released",            "惩罚期结束"),
    "pub_event":     ("publish event, topic",             "属性事件已发出"),
    "send_ok":       ("send success request id",          "云端 ACK 成功"),
    "send_fail":     ("send failed request id",           "发送失败"),
    "chunk_timeout": ("_iot_mqtt_chunk_expire_timeout",   "MQTT 分片超时"),
    "net_exit":      ("Exit Socket listen thread",        "MQTT 连接断开"),
    "net_reconnect": ("MQTT connect success",             "MQTT 重连成功"),
    "status":        ("Device status",                    "设备状态变化"),
    "crash":         ("stack overflow",                   "栈溢出崩溃"),
    "onboard_err":   ("wrong onboarding config",          "配网配置错误"),
}
KEYWORD_RE = re.compile("|".join(re.escape(p[0]) for p in PATTERNS.values()))
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")   # 剥离 ANSI 颜色/控制符


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class SerialLogger(threading.Thread):
    """后台读串口：全部写入文件，命中特征的行回显并记录首次出现时间。"""

    def __init__(self, ser, log_path, verbose, q):
        super().__init__(daemon=True)
        self.ser = ser
        self.log_path = log_path
        self.verbose = verbose
        self.q = q
        self.stop_event = threading.Event()
        self.hits = {}            # key -> list of (offset_sec, line)
        self.t0 = time.monotonic()
        self.line_count = 0

    def elapsed(self):
        return time.monotonic() - self.t0

    def run(self):
        buf = b""
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as f:
            while not self.stop_event.is_set():
                try:
                    data = self.ser.read(256)
                except Exception as e:
                    self.q.put(("error", f"串口读取异常: {e}"))
                    break
                if not data:
                    continue
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = strip_ansi(raw.decode("utf-8", "ignore").replace("\r", "").rstrip())
                    if not line:
                        continue
                    self.line_count += 1
                    stamp = f"[{now_str()}] {self.elapsed():7.3f}s | {line}"
                    f.write(stamp + "\n")
                    f.flush()

                    matched = KEYWORD_RE.search(line)
                    if matched:
                        for key, (pat, _) in PATTERNS.items():
                            if pat in line:
                                self.hits.setdefault(key, []).append((self.elapsed(), line))
                                self.q.put(("hit", key, line))
                                break
                    elif self.verbose:
                        self.q.put(("raw", line))

    def stop(self):
        self.stop_event.set()


def parse_args():
    p = argparse.ArgumentParser(
        description="STDK 发布限流测试：批量发指令 + 记录串口日志 + 统计限流特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--port", default="COM8", help="串口号（默认 COM8）")
    p.add_argument("--baud", type=int, default=115200, help="波特率（默认 115200）")
    p.add_argument("--cmd", default="button 1", help="每条发送的命令（默认 'button 1'）")
    p.add_argument("--count", type=int, default=70, help="发送次数（默认 70；0 = 只抓日志）")
    p.add_argument("--interval", type=float, default=0.3,
                   help="发送周期，单位秒（默认 0.3）")
    p.add_argument("--watch", type=float, default=80,
                   help="发送结束后继续监听的秒数（默认 80，够等一轮 62 秒窗口）")
    p.add_argument("--wait-online", action="store_true",
                   help="先等设备出现 'Device status 6'（云端在线）再开始发送")
    p.add_argument("--wait-timeout", type=float, default=60,
                   help="--wait-online 的超时秒数（默认 60）")
    p.add_argument("-v", "--verbose", action="store_true", help="回显所有串口输出")
    p.add_argument("--log-dir", default=None,
                   help="日志目录（默认 <仓库根>/logs）")
    return p.parse_args()


def wait_for_online(q, logger, timeout):
    """阻塞直到日志里出现 Device status 6，返回是否成功。"""
    print(f"[*] 等待设备进入云端在线状态（Device status 6），超时 {timeout:.0f}s ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for off, line in logger.hits.get("status", []):
            if line.rstrip().endswith("6"):
                print(f"    设备已在线（{line.strip()}）")
                return True
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item[0] == "hit" and item[1] == "status" and item[2].rstrip().endswith("6"):
            print(f"    设备已在线（{item[2].strip()}）")
            return True
    print("    等待超时，仍然继续发送")
    return False


def print_summary(logger, sent, args, log_path):
    print()
    print("=" * 68)
    print("汇总")
    print("=" * 68)
    print(f"  串口          : {args.port} @ {args.baud}")
    print(f"  发送命令      : '{args.cmd}'")
    print(f"  计划/实际发送 : {args.count} / {sent}")
    print(f"  发送周期      : {args.interval} 秒   (理论速率 {1/args.interval:.2f} 条/秒)" if args.interval else "")
    print(f"  总日志行数    : {logger.line_count}")
    print(f"  日志文件      : {log_path}")
    print("-" * 68)

    order = ["pub_event", "send_ok", "send_fail", "limit_local", "limit_block",
             "limit_err", "limit_app", "limit_release", "chunk_timeout",
             "net_exit", "net_reconnect", "status", "crash", "onboard_err"]
    for key in order:
        entries = logger.hits.get(key, [])
        if not entries:
            continue
        first_t = entries[0][0]
        print(f"  {PATTERNS[key][1]:<18} {len(entries):>4} 次   首次 {first_t:7.3f}s")
    print("-" * 68)

    lim = logger.hits.get("limit_local") or logger.hits.get("limit_block")
    if lim:
        t = lim[0][0]
        print(f"  ★ 已触发限流，首次出现在 {t:.3f}s")
        print(f"    原始行：{lim[0][1]}")
        rel = logger.hits.get("limit_release")
        if rel:
            print(f"  ★ 惩罚期结束于 {rel[0][0]:.3f}s，持续 {rel[0][0] - t:.1f} 秒")
    else:
        print("  ✗ 未观察到限流特征")
        print("    提示：限流是 62 秒滑动窗口，需要 COUNT 条发布全部落在 62 秒内")
    if logger.hits.get("crash"):
        print("  ! 检测到栈溢出崩溃")
    if logger.hits.get("net_exit") and not logger.hits.get("net_reconnect"):
        print("  ! MQTT 断开且未见重连")
    print("=" * 68)


def main():
    args = parse_args()

    if serial is None:
        sys.exit("缺少 pyserial，请先安装：  python -m pip install pyserial")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = args.log_dir or os.path.join(repo_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ratelimit_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except Exception as e:
        sys.exit(f"无法打开 {args.port}：{e}\n"
                 "         请先关闭 idf_monitor（Ctrl+]）或其它占用串口的工具。")

    q = queue.Queue()
    logger = SerialLogger(ser, log_path, args.verbose, q)
    logger.start()

    print(f"[*] 串口 {args.port} 已打开，日志写入 {log_path}")
    print(f"[*] 命令 '{args.cmd}'  次数 {args.count}  周期 {args.interval}s")
    print("[*] Ctrl+C 可随时中断\n")

    sent = 0
    interrupted = False
    try:
        if args.wait_online:
            wait_for_online(q, logger, args.wait_timeout)

        if args.count > 0:
            print(f"[*] 开始发送（{now_str()}）")
            t_start = time.monotonic()
            for i in range(args.count):
                ser.write((args.cmd + "\r\n").encode("utf-8"))
                ser.flush()
                sent += 1
                if i == 0 or (i + 1) % 10 == 0 or i + 1 == args.count:
                    print(f"    已发送 {sent}/{args.count}   "
                          f"耗时 {time.monotonic() - t_start:6.1f}s")
                time.sleep(args.interval)
            print(f"[*] 发送完毕，继续监听 {args.watch:.0f}s 观察限流结果 ...")
        else:
            print(f"[*] 纯抓日志模式，监听 {args.watch:.0f}s ...")

        # 监听阶段：持续回显/落盘
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.watch:
            try:
                item = q.get(timeout=0.3)
            except queue.Empty:
                continue
            kind = item[0]
            if kind == "hit":
                print(f"  [{now_str()}] {item[2]}")
            elif kind == "raw":
                print(f"    {item[1]}")
            elif kind == "error":
                print(f"  !! {item[1]}")
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[!] 已中断")
    finally:
        logger.stop()
        ser.close()
        # 把队列里剩余的行清空到文件（读线程已写入，这里只补打最后几条）
        time.sleep(0.2)
        print_summary(logger, sent, args, log_path)
        if interrupted:
            print("（中断运行，数据已保存）")


if __name__ == "__main__":
    main()
