"""QoS仿真使用的基础令牌桶。"""


class TokenBucket:
    """保存一个令牌桶的补充量、容量和当前令牌数。"""

    def __init__(self, fill_per_tick, capacity, tokens=0):
        """功能：设置每周期补充量、桶容量和初始令牌。

        目的：建立一个可由静态YAML初始化、也可由DPU状态控制接口在运行时
        重新配置的基础令牌桶。

        输入：
            fill_per_tick: 每个QoS补充周期增加的Byte数，必须非负。
            capacity: 桶最多保存的Byte数，必须非负。
            tokens: 初始令牌Byte数，必须位于 ``[0, capacity]``。

        输出：
            None: 保存经过校验的令牌桶状态。
        """
        self.fill_per_tick = fill_per_tick
        self.capacity = capacity
        self.tokens = tokens

    def refill(self):
        """功能：补充一个周期的令牌，并限制在桶容量以内。

        目的：模拟硬件周期补充逻辑，使CIR/PIR长期速率由
        ``fill_per_tick`` 决定、突发量由 ``capacity`` 决定。

        输入：
            无。

        输出：
            None: 原地更新当前令牌数。
        """
        self.tokens = min(self.capacity, self.tokens + self.fill_per_tick)

    def consume(self, byte_count):
        """功能：完整IO离开队列时扣除相同Byte数的令牌。

        目的：让令牌消耗与实际下发Byte数一致；调用方负责在扣除前完成
        eligibility检查。

        输入：
            byte_count: 当前完整IO需要扣除的Byte数。

        输出：
            None: 原地减少当前令牌数。
        """
        # 基础桶只做数值更新；令牌是否足够由上层队列控制器判断。
        self.tokens -= byte_count

    def reconfigure(self, fill_per_tick, capacity):
        """功能：运行时替换令牌补充量和桶容量。

        目的：承载DPU到QoS的动态CIR/PIR设置，并清除旧需求留下的令牌。

        输入：新的每周期补充Byte数和桶容量。

        输出：
            None: 原地更新速率、容量和当前令牌。
        """
        self.fill_per_tick = fill_per_tick
        self.capacity = capacity
        # Queue速率租约改变时不继承旧租约的突发令牌。
        self.tokens = 0
