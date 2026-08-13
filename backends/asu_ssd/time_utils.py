#!/usr/bin/env python3
"""ASU后端内部时间换算工具。

一个微秒被划分成15000个整数时间单位。
这样0.1 us、0.1024 us和1/6 us都能表示成整数，事件比较不需要浮点容差。
"""

TIME_UNITS_PER_US = 15000  # 一个微秒包含15000个内部整数时间单位。
TIME_UNITS_PER_SECOND = TIME_UNITS_PER_US * 1_000_000  # 一秒对应的内部时间单位数量。


def us_to_time(us_value):
    """把微秒转换成后端内部整数时间。"""
    return int(round(us_value * TIME_UNITS_PER_US))  # 配置中的整数微秒可以被精确转换。


def time_to_us(time_value):
    """把后端内部整数时间转换成微秒。"""
    return time_value / TIME_UNITS_PER_US  # 结果用于打印，不参与事件先后判断。


def rate_to_interval(rate_per_second):
    """把每秒处理数量转换成相邻启动事件的整数间隔。"""
    return TIME_UNITS_PER_SECOND // rate_per_second  # 当前配置中的6M和10M都可以整除。


def bandwidth_to_interval(command_size_bytes, bandwidth_bytes_per_second):
    """根据命令Byte数和带宽计算相邻命令的整数启动间隔。"""
    numerator = command_size_bytes * TIME_UNITS_PER_SECOND  # 先计算一个命令需要占用的时间单位分子。
    return numerator // bandwidth_bytes_per_second  # 4 KiB和40 GB/s得到精确的0.1024 us。
