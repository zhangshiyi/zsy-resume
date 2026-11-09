# 个人简历网站 · 张世祎

在线访问：**https://zhangshiyi.github.io/zsy-resume/**

一个零框架、零构建的静态个人简历站，外加一个可选的 AI 问答助手。整站纯 HTML / CSS / JavaScript

简历工作方向为投流，AI提效方向为投流

## 页面

| 路径 | 内容 |
|---|---|
| `index.html` | 简历主页面（工作经历、项目、技能、联系方式） |
| `sov-dashboard.html` | SOV 数据看板 · 品牌声量份额可视化（Chart.js） |
| `xhs-ads-project/index.html` | 小红书投放复盘看板（单文件，含图表与交互） |
| `行程演示_杭州.html` | 旅行 AI 项目的行程规划演示页 |

## 本地预览

```bash
git clone https://github.com/zhangshiyi/zsy-resume.git
cd zsy-resume
python3 -m http.server 8000
# 打开 http://localhost:8000
```

## AI 问答助手（可选）

`resume-agent/` 是一个用纯 Python 标准库实现的 RAG + ReAct 服务，**零第三方依赖**，单端口同时提供静态页面和问答接口。

```bash
STATIC_DIR=. python3 resume-agent/server.py
# 打开 http://localhost:8000
```

- `kb.json`：知识库，全部内容取自简历原文，共 16 个 chunk
- 检索：中文 bigram + 英文词，TF-IDF 打分
- 路由：语义分 ≥ 阈值走语义检索，否则走意图分类（联系方式 / 项目 / 指标 / 自我介绍）
- `LLM_API_KEY`：可选。配置后调用 OpenAI 兼容接口作答（仍限定在检索到的原文内）；不配置则使用检索模板模式，答案同样只来自简历原文

在纯静态托管（例如 GitHub Pages）下没有 Python 后端，前端会先探测 `/api/health`，不通则自动隐藏助手入口，不会出现坏掉的按钮。

## 技术说明

- 无打包、无依赖管理，`vendor/` 内置 Chart.js 与 datalabels 插件
- 响应式布局，左侧导航 + 右侧内容，栅格 + sticky
- 部署：GitHub Pages 直接指向 `main` 分支根目录即可

## 说明

本站为个人求职作品，页面内容为本人真实经历。
