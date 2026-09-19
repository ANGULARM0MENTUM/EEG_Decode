# 测试步骤与结果

本地执行：`pip install -r requirements.txt && pytest -q`

| 用例 | 步骤 | 期望 | 实现 |
| --- | --- | --- | --- |
| 实时 EEG 数据流 | `tests/test_pipeline.py::test_stream_windows_status_and_alignment` 连续注入 20 块多通道数据 | 收到 waveform / entropy / status；窗时间戳单调且步长≈hop | pytest |
| 30 分钟数据流 | `tests/test_acceptance.py::test_thirty_minute_equivalent_stream` 加速注入 256Hz×1800s | 样本数精确、窗数达标、环形缓冲字节数不变 | pytest（标记 slow） |
| 在线熵 | 同上 alignment 测试 | 每通道/全局熵 ∈[0,1]，含 e2e 延迟与吞吐 | pytest |
| 背压 | `test_backpressure_drop_oldest` 队列长度=2 且先不消费 | 出现 BACKPRESSURE，dropped_chunks≥1，会话仍可随后 start | pytest |
| 中断恢复 | `test_gap_then_resume` 停顿 > gap_sec 再继续 | 进入 STREAM_GAP，恢复后状态离开 STREAM_GAP 且继续出窗 | pytest |
| 异常数据 | `test_anomaly_quality_status` 通道饱和 | QUALITY_BAD | pytest |
| 汇总 | `test_mid_session_and_final_summary` | 中途与结束后均有均值/极值/趋势/区间；写出 summary.json | pytest |
| HTTP API | `tests/test_api.py` | health / 创建 / summary / stop | pytest |

墙钟 30 分钟（可选，非 CI）：

```
python scripts/run_long_session.py --seconds 1800
```

浏览器手工：启动 `python -m eeg_decode`，打开 `/`，开始会话，观察波形、熵曲线、KPI，中途点「拉取汇总」，再结束会话。
