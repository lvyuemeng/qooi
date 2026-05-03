# AGENT.md — qooi 项目约定

## 技术栈

- **量化引擎**: BigQuant SDK (`bigquant[all]`)
  - 数据查询: `from bigquant import dai`
  - 本地回测: `from bigquant import bigtrader`
  - 认证: `bq auth --apikey <AK.SK>` 或代码内 `bigquant.init(ak=..., sk=...)`
  - 参考: https://bigquant.com/wiki/doc/vac4qwmQr4
- **Python 管理**: `uv` (不使用 pip, 不依赖全局 Python 状态)
- **Python 版本**: 3.11–3.13

## 项目结构

```
qooi/
├── .scratch/        # 本地 issue 追踪 (见 docs/agents/)
├── docs/
│   ├── AGENT.md      # 本文
│   ├── adr/          # 架构决策记录
│   └── agents/       # agent 技能配置
├── src/              # 策略源码
├── scripts/          # 回测/运行脚本
├── CONTEXT.md        # 领域术语表
├── AGENTS.md         # agent 技能配置（入口）
└── pyproject.toml    # uv 项目配置
```

## 编码约定

| 项目           | 约定                         |
| -------------- | ---------------------------- |
| Python 安装    | `uv add 'bigquant[all]'`     |
| Python 运行    | `uv run python <script>`     |
| 类型注解       | 所有函数必须标注类型          |
| import         | `from bigquant import dai / bigtrader` |
| 认证方式       | 本地 `bq auth configure`，不提交凭证 |
| DataFrame 库   | 优先 Polars (`result.pl()`)，避免 pandas |

## 工作流

1. `uv sync` — 安装依赖
2. `uv run python scripts/backtest.py` — 运行回测
3. BigQuant AI Studio 仅用于云端分布式计算或数据探索，策略开发优先本地
