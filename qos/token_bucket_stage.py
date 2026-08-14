"""维护每个Queue的CIR/PIR令牌桶和输入FIFO。

Group只负责两件事：为Queue CIR提供组级分配预算，以及供分层WRR选择。
Group不再创建运行时令牌桶，也不参与IO的准入判断或令牌扣除。SSD总出口
同样不在本模块重复创建Root令牌桶，物理带宽上限继续由SSD后端负责。
"""

from collections import deque
from decimal import Decimal

from .token_bucket import TokenBucket


def _rate_to_bytes_per_tick(rate_gb_s, update_period_us):
    """功能：把十进制GB/s速率精确换算成每个补充周期的整数Byte数。

    目的：让YAML初始速率和DPU运行时设置使用完全相同的换算规则。
    YAML中的速率先经由字符串构造 ``Decimal``，避免二进制浮点数让本应是
    整数的周期令牌出现 ``102399.999...`` 一类误差。正式配置中的速率与
    80 us周期都能得到整数Byte，因此最后直接转换成 ``int`` 交给令牌桶。

    输入：
        rate_gb_s: 非负的十进制GB/s数值。
        update_period_us: 正数令牌补充周期，单位微秒。

    输出：
        int: 每个补充周期应增加的整数Byte数。
    """
    # GB/s × 10^9 Byte/GB × (us / 10^6 us/s) = GB/s × 1000 × us。
    return int(
        Decimal(str(rate_gb_s))
        * Decimal(1_000)
        * Decimal(update_period_us)
    )


def _parse_pir_rate(configured_rate):
    """功能：把Queue的PIR配置转换成有限速率或 ``None``。

    目的：保留显式 ``uncapped`` 能力，同时允许项目默认值和DPU策略使用
    数值PIR关闭无限峰值模式。

    输入：
        configured_rate: 非负数值或字符串 ``uncapped``。

    输出：
        number | None: 有限PIR原值；``uncapped`` 返回None。

    ``uncapped`` 表示该Queue不施加峰值限制，返回 ``None`` 后既不检查也不
    扣除PIR令牌。数值PIR保持原值，并在构造Queue控制器时创建真实PIR桶。
    """
    if configured_rate == "uncapped":
        return None
    return configured_rate


def _capacity_for_rate(minimum_capacity, fill_per_tick, max_io_size_bytes):
    """功能：为一个动态速率计算不会截断补充量的令牌桶容量。

    目的：DPU可能把CIR/PIR提高到整盘带宽。若容量仍小于一个周期补充量，
    ``refill`` 会在每个周期丢弃令牌，使实际速率低于设置值；额外保留一个
    最大IO空间还可以让上周期不足一个完整IO的余量跨周期累积。

    输入：
        minimum_capacity: YAML配置允许的最小突发容量，单位Byte。
        fill_per_tick: 当前速率对应的每周期补充Byte数。
        max_io_size_bytes: QoS入口允许的最大完整IO大小，单位Byte。

    输出：
        int: 同时满足最小突发量和动态速率所需的桶容量。
    """
    return max(
        minimum_capacity,
        fill_per_tick + max_io_size_bytes,
    )


def build_group_cir_parameters(queue_layout, token_config):
    """功能：把各Group的CIR预算换算成每周期Byte数。

    目的：继续按权重计算组内各Queue的CIR。这里不创建Group
    令牌桶；Group CIR是配置层分配预算，不是运行时IO准入条件。

    输入：
        queue_layout: 已展开的Queue和Group布局。
        token_config: 包含补充周期与Group CIR的令牌配置。

    输出：
        dict: ``group_id -> cir_fill_bytes_per_tick`` 参数映射。
    """
    update_period_us = token_config["update_period_us"]
    group_rates = {
        item["group_id"]: item
        for item in token_config["group_rates"]
    }
    parameters = {}

    for group_id in queue_layout.group_order:
        configured = group_rates[group_id]
        cir_gb_s = configured["cir_gb_s"]
        cir_fill = _rate_to_bytes_per_tick(cir_gb_s, update_period_us)
        parameters[group_id] = {
            "cir_fill_bytes_per_tick": cir_fill,
        }

    return parameters


def build_queue_parameters(queue_layout, token_config, group_cir_parameters=None):
    """功能：为布局中的每个Queue生成CIR/PIR令牌参数。

    目的：默认Queue CIR不把Group速率当作单队列基础值相乘，而是把Group每周期
    的整数CIR令牌按32项权重之和归一化分配。正式的 ``[4,3,2,1] × 8``
    权重总和为80，所以每组32个Queue的补充量严格等于本组补充量。单Queue
    override最后覆盖默认CIR、CBS或PIR；未单独指定的桶容量使用
    Queue默认值，并会按动态速率自动扩大。

    输入：
        queue_layout: 已展开的Queue/Group布局。
        token_config: Queue权重、默认速率、桶容量和override配置。
        group_cir_parameters: 可选的Group周期CIR预算；None时现场生成。

    输出：
        dict: ``queue_id -> CIR/PIR补充量与桶容量`` 参数映射。
    """
    if group_cir_parameters is None:
        # Group参数只是Queue CIR的计算输入，不会创建运行时Group令牌桶。
        group_cir_parameters = build_group_cir_parameters(
            queue_layout,
            token_config,
        )

    update_period_us = token_config["update_period_us"]
    cir_weights = token_config["queue_cir_weight_bitmap"]
    total_cir_weight = sum(cir_weights)
    default_pir = token_config["queue_default_pir_gb_s"]
    default_pbs_bytes = token_config["queue_pbs_bytes"]
    max_io_size_bytes = token_config["queue_max_io_size_bytes"]
    overrides = token_config["queue_overrides"]
    parameters = {}

    for group_id in queue_layout.group_order:
        group_cir_fill = group_cir_parameters[group_id][
            "cir_fill_bytes_per_tick"
        ]
        for queue_position, queue_id in enumerate(queue_layout.group_queues[group_id]):
            queue_override = overrides.get(queue_id, {})
            cir_weight = cir_weights[queue_position]

            if "cir_gb_s" in queue_override:
                cir_fill = _rate_to_bytes_per_tick(
                    queue_override["cir_gb_s"],
                    update_period_us,
                )
            else:
                # 全程在整数Byte域内分配，避免32次浮点计算产生组内累计误差。
                cir_fill = group_cir_fill * cir_weight // total_cir_weight

            cbs_bytes = queue_override.get(
                "cbs_bytes",
                token_config["queue_cbs_bytes"],
            )
            cbs_bytes = _capacity_for_rate(
                cbs_bytes,
                cir_fill,
                max_io_size_bytes,
            )

            configured_pir = queue_override.get("pir_gb_s", default_pir)
            pir_gb_s = _parse_pir_rate(configured_pir)
            if pir_gb_s is None:
                pir_fill = None
                pbs_bytes = None
            else:
                pir_fill = _rate_to_bytes_per_tick(pir_gb_s, update_period_us)
                pbs_bytes = _capacity_for_rate(
                    queue_override.get("pbs_bytes", default_pbs_bytes),
                    pir_fill,
                    max_io_size_bytes,
                )

            parameters[queue_id] = {
                "cir_fill_bytes_per_tick": cir_fill,
                "pir_fill_bytes_per_tick": pir_fill,
                "cbs_bytes": cbs_bytes,
                "pbs_bytes": pbs_bytes,
            }

    return parameters


class QueueTokenController:
    """维护一个Queue的CIR/PIR令牌桶以及输入FIFO。"""

    def __init__(self, parameters, initial_state):
        """功能：依据Queue参数创建CIR桶、可选PIR桶和FIFO。

        目的：``full`` 启动把初始令牌设置为桶容量，用于允许配置范围内的起始突发；
        ``empty`` 启动则从零开始积累。uncapped Queue的 ``pir_bucket`` 保存
        为 ``None``，资格检查和扣令牌都会据此跳过PIR。

        输入：
            parameters: 一个Queue的CIR/PIR补充量、容量和显示速率。
            initial_state: ``full`` 或 ``empty`` 初始令牌模式。

        输出：
            None: 初始化Queue令牌控制器和空FIFO。
        """
        cir_tokens = parameters["cbs_bytes"] if initial_state == "full" else 0
        self.cir_bucket = TokenBucket(
            fill_per_tick=parameters["cir_fill_bytes_per_tick"],
            capacity=parameters["cbs_bytes"],
            tokens=cir_tokens,
        )

        if parameters["pir_fill_bytes_per_tick"] is None:
            self.pir_bucket = None
        else:
            pir_tokens = parameters["pbs_bytes"] if initial_state == "full" else 0
            self.pir_bucket = TokenBucket(
                fill_per_tick=parameters["pir_fill_bytes_per_tick"],
                capacity=parameters["pbs_bytes"],
                tokens=pir_tokens,
            )
        self.waiting_requests = deque()

    def reconfigure_rates(self, cir_fill, pir_fill, cbs_bytes, pbs_bytes):
        """功能：在保留Queue FIFO的同时替换本Queue的CIR/PIR令牌状态。

        目的：实现DPU到QoS的运行时速率设置，不改变FIFO内容。

        输入：
            cir_fill: CIR每周期补充Byte数。
            pir_fill: PIR每周期补充Byte数；None表示uncapped。
            cbs_bytes: CIR桶容量。
            pbs_bytes: PIR桶容量；uncapped时为None。

        输出：
            None: 原地更新本Queue速率控制状态，FIFO内容保持不变。
        """
        self.cir_bucket.reconfigure(
            fill_per_tick=cir_fill,
            capacity=cbs_bytes,
        )

        # DPU可以在运行时在“有限PIR”和“uncapped”之间切换。
        # uncapped不需要PIR状态，直接移除原有令牌桶。
        if pir_fill is None:
            self.pir_bucket = None
        elif self.pir_bucket is None:
            self.pir_bucket = TokenBucket(
                fill_per_tick=pir_fill,
                capacity=pbs_bytes,
                tokens=0,
            )
        else:
            self.pir_bucket.reconfigure(
                fill_per_tick=pir_fill,
                capacity=pbs_bytes,
            )

    def refill(self):
        """为本Queue补充一次CIR令牌，并在有限PIR存在时补充PIR令牌。

        两个桶都沿用基础 ``TokenBucket.refill`` 的容量截断规则；uncapped Queue
        不创建也不模拟PIR令牌，因此只执行CIR补充。
        """
        self.cir_bucket.refill()
        if self.pir_bucket is not None:
            self.pir_bucket.refill()

    def has_cir_tokens(self, request_size):
        """返回本Queue的CIR余额是否能够完整覆盖队首请求。

        CIR不允许拆分一个IO扣令牌，因此只有当前余额大于或等于
        请求字节数时，本Queue才能通过CIR路径检查。
        """
        return self.cir_bucket.tokens >= request_size

    def has_pir_tokens(self, request_size):
        """返回本Queue的有限PIR是否允许请求下发。

        uncapped Queue没有峰值上限，直接返回True；有限Queue必须让PIR余额
        完整覆盖请求。
        """
        return (
            self.pir_bucket is None
            or self.pir_bucket.tokens >= request_size
        )

    def consume_cir(self, request_size):
        """在CIR类别请求下发后扣除本Queue的CIR令牌。

        Stage只在CIR eligibility已经通过后调用本函数，所以这里直接
        扣除一个完整请求的字节数，不重复计算资格。
        """
        self.cir_bucket.consume(request_size)

    def consume_pir(self, request_size):
        """从本Queue的有限PIR桶扣除一个完整请求的字节数。

        CIR和EXCESS下发都必须遵守数值PIR。uncapped Queue没有PIR桶，
        因此本函数在该情况下不改变任何状态。
        """
        if self.pir_bucket is not None:
            self.pir_bucket.consume(request_size)

    def enqueue(self, request):
        """把一个已经映射到本queue_id的完整IO追加到FIFO尾部。

        函数保留原请求字典，让后续QoS分类、下发时刻和SSD结果都写回
        同一个对象，不创建会丢失字段的中间副本。
        """
        self.waiting_requests.append(request)

    def head_request(self):
        """返回资格判断和路径扣令牌共同使用的队首IO。

        这是只读查看，不改变FIFO。请求只会在Queue令牌扣除后，由
        ``pop_head`` 真正移除。
        """
        return self.waiting_requests[0]

    def pop_head(self):
        """在Queue令牌扣除后移除并返回队首IO。

        Stage每次仲裁只下发一个完整请求，所以此处只执行一次
        ``popleft``，随后立即回到新的CIR轮。
        """
        return self.waiting_requests.popleft()

    def queue_depth(self):
        """返回当前FIFO中的完整IO数量。

        调度资格用它快速跳过空Queue。
        """
        return len(self.waiting_requests)


class PerQueueTokenBucketStage:
    """连接固定Queue布局、每Queue令牌桶以及每Queue输入FIFO。"""

    def __init__(self, queue_layout, token_config):
        """功能：为布局中全部Queue创建令牌/FIFO控制器和总occupancy状态。

        目的：Group CIR仅用于计算各Queue的CIR份额；运行阶段只保存
        Queue控制器和Queue Bank总占用计数，IO准入与令牌扣除不经过Group级检查。

        输入：
            queue_layout: 已展开的Queue/Group固定布局。
            token_config: Queue CIR/PIR、桶容量和初始状态配置。

        输出：
            None: 初始化全部Queue控制器和为0的总占用计数。
        """
        self.update_period_us = token_config["update_period_us"]
        self.queue_order = list(queue_layout.queue_order)
        self.minimum_cbs_bytes = token_config["queue_cbs_bytes"]
        self.minimum_pbs_bytes = token_config["queue_pbs_bytes"]
        self.max_io_size_bytes = token_config["queue_max_io_size_bytes"]

        group_cir_parameters = build_group_cir_parameters(
            queue_layout,
            token_config,
        )
        queue_parameters = build_queue_parameters(
            queue_layout,
            token_config,
            group_cir_parameters,
        )
        initial_state = token_config["initial_state"]

        self.controllers = {
            queue_id: QueueTokenController(
                queue_parameters[queue_id],
                initial_state,
            )
            for queue_id in self.queue_order
        }

        # 硬件Queue Bank通常维护总occupancy寄存器或非空位图，
        # 不会为了判断“是否有积压”而串行扫描256个FIFO。
        # 这个计数只包含已进入Queue FIFO的请求，不包含未来到达堆。
        self.total_queued_requests = 0

    def set_queue_rate(
        self,
        queue_id,
        cir_fill_bytes_per_tick,
        pir_fill_bytes_per_tick,
    ):
        """功能：根据DPU控制命令动态设置一个Queue的CIR和PIR。

        目的：提供DPU与QoS之间唯一的速率设置入口。这里只修改目标Queue的
        令牌寄存器，不改变Group、分层WRR权重、FIFO内容或SSD后端上限。

        输入：
            queue_id: 当前QoS实例中的合法Queue ID。
            cir_fill_bytes_per_tick: 新CIR的整数周期补充Byte数。
            pir_fill_bytes_per_tick: 新PIR的整数周期补充Byte数；
                None表示uncapped。

        输出：
            None: 原地更新目标Queue令牌桶。
        """
        self.controllers[queue_id].reconfigure_rates(
            cir_fill=cir_fill_bytes_per_tick,
            pir_fill=pir_fill_bytes_per_tick,
            cbs_bytes=_capacity_for_rate(
                self.minimum_cbs_bytes,
                cir_fill_bytes_per_tick,
                self.max_io_size_bytes,
            ),
            pbs_bytes=(
                None
                if pir_fill_bytes_per_tick is None
                else _capacity_for_rate(
                    self.minimum_pbs_bytes,
                    pir_fill_bytes_per_tick,
                    self.max_io_size_bytes,
                )
            ),
        )

    def enqueue(self, request):
        """功能：把完整IO追加到指定Queue FIFO并更新总占用计数。

        目的：模拟硬件Queue Bank在FIFO写入成功时同步更新总occupancy，
        使后续可以O(1)判断是否存在QoS积压。

        输入：
            request: 携带合法 ``queue_id`` 和完整IO字段的请求字典。

        输出：
            None: 原地更新对应FIFO和Queue Bank总占用计数。
        """
        self.controllers[request["queue_id"]].enqueue(request)
        self.total_queued_requests += 1

    def refill(self):
        """在同一80 us令牌事件中补充全部Queue速率桶。

        Group没有运行时令牌桶，因此本事件只遍历Queue控制器。
        """
        for controller in self.controllers.values():
            controller.refill()

    def is_cir_eligible(self, queue_id):
        """判断队首IO是否满足本Queue的CIR准入条件。

        CIR轮要求FIFO非空、Queue CIR余额足够，并且该Queue的有限PIR余额
        也足够。Group只参与WRR选择，不参与这里的令牌检查。
        """
        queue_controller = self.controllers[queue_id]
        if queue_controller.queue_depth() == 0:
            return False

        request = queue_controller.head_request()
        request_size = request["size_bytes"]
        return (
            queue_controller.has_cir_tokens(request_size)
            and queue_controller.has_pir_tokens(request_size)
        )

    def is_excess_eligible(self, queue_id):
        """判断队首IO能否借用保障以外的空闲容量。

        EXCESS轮不查看Queue CIR余额，只检查FIFO和本Queue的有限PIR。
        Queue PIR为uncapped时，只要FIFO非空即可参加第二轮；最终物理总带宽
        仍由下游SSD反压限制。
        """
        queue_controller = self.controllers[queue_id]
        if queue_controller.queue_depth() == 0:
            return False

        request = queue_controller.head_request()
        request_size = request["size_bytes"]
        return queue_controller.has_pir_tokens(request_size)

    def dequeue(self, queue_id, current_time_us, *, rate_class):
        """功能：扣除已选Queue的令牌、取出队首IO并更新总占用计数。

        目的：``rate_class='CIR'`` 扣除Queue CIR和有限PIR；
        ``rate_class='EXCESS'``
        不改变CIR余额，只扣除Queue有限PIR。调用方必须先用对应eligibility
        完成仲裁；FIFO读出成功时同步递减Queue Bank occupancy。

        输入：
            queue_id: 调度器已选中的Queue ID。
            current_time_us: 当前QoS仿真微秒时刻。
            rate_class: 调度类别，只能为 ``CIR`` 或 ``EXCESS``。

        输出：
            dict: 已写入QoS类别和出队时刻的原请求字典。
        """
        queue_controller = self.controllers[queue_id]
        request_size = queue_controller.head_request()["size_bytes"]

        if rate_class == "CIR":
            queue_controller.consume_cir(request_size)

        # Queue PIR是该Queue的硬上限，因此CIR和EXCESS都会扣除有限PIR。
        queue_controller.consume_pir(request_size)

        request = queue_controller.pop_head()
        self.total_queued_requests -= 1
        # 后续事件引擎与SSD继续传递原字典，类别和出队时刻也随请求保留下来。
        request["qos_rate_class"] = rate_class
        request["token_dequeue_time_us"] = current_time_us
        return request

    def has_queued_requests(self):
        """功能：使用Queue Bank总occupancy判断是否存在已到达积压。

        目的：以硬件总占用寄存器的语义替代顺序扫描256个Queue depth，
        在不改变FIFO、令牌或WRR逻辑的前提下将检查降为O(1)。

        输入：
            无。

        输出：
            bool: 至少一个已到达请求仍在Queue FIFO中时返回True。
        """
        return self.total_queued_requests > 0
