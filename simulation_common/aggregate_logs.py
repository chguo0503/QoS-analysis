"""只保留仿真摘要需要的请求计数和字节数。"""


class CountOnlyAppendLog:
    """提供 list.append/len 接口，但不保存逐请求记录。"""

    def __init__(self):
        self.count = 0

    def append(self, record):
        self.count += 1

    def __len__(self):
        return self.count


class DispatchAggregateLog:
    """统计 QoS 下发的请求、字节和 CIR/EXCESS 数量。"""

    def __init__(self):
        self.count = 0
        self.byte_count = 0
        self.cir_count = 0
        self.excess_count = 0

    def append(self, request):
        self.count += 1
        self.byte_count += request["size_bytes"]
        if request["qos_rate_class"] == "CIR":
            self.cir_count += 1
        else:
            self.excess_count += 1

    def __len__(self):
        return self.count
