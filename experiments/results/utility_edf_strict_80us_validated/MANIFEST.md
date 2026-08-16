# 正式结果代码清单

本文件记录生成同目录 `summary.json` 时的关键环境和源码 SHA-256。
工作树包含用户已有删除及未提交改动，因此不能只用 Git HEAD 重建实验；
下面的内容哈希才是本次结果直接依赖的代码快照标识。

```text
Python 3.10.10
PyYAML 6.0
Git HEAD dea09ed0424aae8d557247d8c9bdf5440f27a4ef

75f34bf3ce0da8f1fe11564a76dd79e80fd4831ea3139527eed331ff04d0cd3f  experiments/results/utility_edf_strict_80us_validated/summary.json
8053f1d545dee97ea0e69e424ab0cb5f28ba807b7dae65413349036781a68ae3  config/simulation_config.yaml
28a4eedafb0ad1b096a5dcd4f3465bb5e92aef25348a0e1b8dc19016c3b0c117  DPU/dispatcher.py
c23444a1812cb54542d9b4b5caa872886d5d2d1210d1577450ea1526a7a37b49  DPU/rate_controller.py
b7378d07dec0be7c2ec0a5aa0923c221f91d19fc9f1ae71da7b7e05770fff4f3  qos/schedulers/weighted_round_robin.py
538bf78ec5807059080451eb37899ab27dc9a2d2315c0ba2450ca4f906fdcb60  qos/schedulers/hierarchical.py
a9461c41e7f422839674254c67801d503fa65b0a9bc69044819e70d8c9331fc4  discrete_simulation/simulator.py
424258a5dab672759d20f5935b9e4ff3a27a6e465705b5435c14dac05bfe46ce  llm_workload/layer_request.py
49393b21c84a20277c5e3409da80cd5d445b1d4de86918f60e01203517c9ef1c  llm_workload/kv_placement_manager.py
b3074cba54c38e5180cc2a244eb1119864dbf6708edffb6c79f2cc53b45e274d  qos_ssd_simulator.py
```

重新核对可使用：

```bash
sha256sum \
  config/simulation_config.yaml \
  DPU/dispatcher.py DPU/rate_controller.py \
  qos/schedulers/weighted_round_robin.py \
  qos/schedulers/hierarchical.py \
  discrete_simulation/simulator.py \
  llm_workload/layer_request.py \
  llm_workload/kv_placement_manager.py \
  qos_ssd_simulator.py
```
