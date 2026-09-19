# EEG 实时解码系统

根据笔试要求实现的 **多通道 EEG 实时解码演示**：从 BrainFlow Synthetic Board（或上传流）持续接收数据，在线滤波 / 分窗 / 谱熵计算，Web 客户端实时展示波形、熵值、吞吐与延迟，并在会话中途或结束后生成结构化汇总。

前端保持简单，重点在后端流式架构、有界内存、背压和可复现测试。

## 架构

```
BrainFlow Synthetic Board          浏览器
        |                            |
        v                            v
  producer thread  --->  bounded queue  --->  worker
                              |                  |
                         drop-oldest        状态机 SOS 滤波
                         (backpressure)     滑动窗谱熵 + EMA
                                                |
                                                +--> RAM 环形缓冲 (仅展示窗口)
                                                +--> windows.jsonl (增量落盘)
                                                +--> WebSocket 推送
                                                +--> 在线 Welford 统计 / 区间标注
```

- **实时通信**：FastAPI HTTP + WebSocket。
- **流式处理**：独立生产线程与计算线程，由有界队列解耦。
- **会话状态**：`OK | BACKPRESSURE | STREAM_GAP | QUALITY_BAD | COMPUTE_ERROR`。中断后数据恢复即可继续，不必重启会话。
- **熵值工作流**：按 `fs × window_sec` 取窗、按 `hop_sec` 步进；滤波状态跨窗保持；结果时间戳为窗末端采样时间，与波形共用同一时钟。

不可把成熟开源 EEG 实时分析套件当作核心；本仓库自行实现缓冲、分窗、状态机和熵流水线。BrainFlow **只作为模拟板数据源**。

## 熵指标与参数

默认 **归一化谱熵**（Shannon entropy of band-limited PSD）：

\[
H = -\sum p_i \log_2 p_i \big/ \log_2 K \in [0, 1]
\]

- `0`：能量集中在单个频bin（规则）
- `1`：带内谱接近平坦（噪声样）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `fs` | Synthetic Board 给出，fallback 256 Hz | 采样率 |
| `window_sec` | 2.0 | 窗长 |
| `hop_sec` | 0.25 | 步长（展示刷新与延迟的折中） |
| `bandpass` | 1–40 Hz, Butterworth 4 阶 SOS | 在线带通 |
| `notch` | 50 Hz, Q=30 | 工频陷波 |
| `spectral_fmin/fmax` | 1–40 Hz | 谱熵积分带 |
| `smooth_alpha` | 0.35 | EMA，抑制无意义抖动，保留较大跳变 |
| `entropy_method` | `spectral` | 可改为 `permutation`（order=3, delay=1，范围同样 [0,1]） |

环境变量前缀 `EEG_`，见 `eeg_decode/config.py`。示例：`EEG_WINDOW_SEC=1.0`。

## 使用的基础库（及用途）

- **brainflow**：Synthetic Board 模拟实时 EEG，不参与熵计算。
- **numpy**：缓冲、FFT 谱熵、在线统计。
- **scipy.signal**：Butterworth / IIR notch 设计与 `sosfilt` 状态滤波。
- **fastapi / uvicorn**：HTTP + WebSocket。

## 背压与异常

- 队列满：**丢弃最旧块**（drop-oldest），计数 `dropped_chunks`，状态 `BACKPRESSURE`，会话不中断。
- 超过 `gap_sec`（默认 0.75s）无新数据或时间戳跳变：`STREAM_GAP`，恢复后自动回到 `OK`。
- 饱和 / 平线 / NaN：该通道本窗熵记为 null，状态 `QUALITY_BAD`。
- 计算异常：`COMPUTE_ERROR`，后续窗继续尝试。

长时间会话：RAM 只保留 `display_sec`（默认 8s）环形缓冲；窗结果追加写入 `data/sessions/<id>/windows.jsonl`；汇总统计为 O(通道数) 的 Welford。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
python -m eeg_decode
```

打开 http://127.0.0.1:8000 ，点「开始会话」。没有 BrainFlow 原生库时自动退回 numpy 正弦+噪声发生器（接口相同）。

Docker：

```bash
docker compose up --build
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/sessions` | `{ "source": "brainflow" \| "upload", "accelerated": false }` |
| GET | `/api/sessions/{id}/summary` | 中途或结束后的结构化汇总 |
| POST | `/api/sessions/{id}/stop` | 结束并写 `summary.json` |
| POST | `/api/sessions/{id}/ingest` | 外部上传一块 `(t, data[ch][t])` |
| WS | `/ws/sessions/{id}` | 推送 `waveform` / `entropy` / `status`；可发 `{type:ingest,...}` |

## 测试与验收

```bash
pytest -q
```

覆盖：

1. **实时数据流**：持续注入多通道数据，断言波形帧、窗时间戳、接收状态。
2. **30 分钟当量**：加速注入 `256 Hz × 1800 s` 样本，断言窗数量、内存缓冲上限、会话仍可汇总。
3. **在线熵**：分窗长度、时间对齐、每通道/全局熵、吞吐与 e2e 延迟字段。
4. **流异常**：队列溢出背压；短暂中断后自动续算；饱和/NaN 明确状态。
5. **汇总**：中途 `summary()` 与结束后完整 JSON（均值/极值/波动/趋势/稳定·变化·异常区间/延迟）。

步骤与期望结果见 `tests/` 与 `docs/TEST_REPORT.md`。样例汇总：`examples/summary_sample.json`（测试也会再生成一份）。

### 墙钟 30 分钟（可选）

```bash
python scripts/run_long_session.py --seconds 1800
```

默认测试使用加速模式，避免 CI 真的跑半小时。

## 加分实现摘要

- 有界队列、丢弃策略、超时缺口检测、会话取消（stop）、失败后继续。
- 窗结果持久化 + 环形 RAM，支持断流后续算（同一 session）。
- 结构化事件日志 `events.jsonl`（接收 / 背压 / 缺口 / 计算耗时字段）。
- 在线预处理：陷波、带通、伪迹标记（饱和、平线、肌电代理）。
- EMA 平滑 + 区间分类（STABLE / CHANGING / ANOMALOUS）。
- 谱熵与排列熵离线对比测试。
- `Dockerfile` + `docker-compose.yml` + GitHub Actions。

扩容：会话间无共享可变状态，可按 session 水平扩展 worker；生产故障用 `stop` 落盘汇总后重启进程，从 JSONL 可审计历史窗（实时环形缓冲不恢复，这是展示用热数据）。
