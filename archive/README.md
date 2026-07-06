# Archive

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This folder stores historical snapshots, cleanup records, and older workflow
artifacts that are kept for traceability rather than daily execution.

本目录用于保存历史快照、清理记录和旧工作流材料，目的主要是追溯与比较，而不是日常运行。

## 中文

## 归档原则

- 当前维护主线始终优先看 `current/`
- `archive/` 里的内容主要用于：
  - 比较旧逻辑与新逻辑
  - 追溯某个时间点的行为
  - 保存一次性清理或迁移记录
- 除非明确在做历史对照，不建议直接从 `archive/` 运行脚本

## 当前主要内容

- [20260603_code/](./20260603_code/)
  - 2026-06-03 的历史代码快照
- [current_cleanup_20260623/](./current_cleanup_20260623/)
  - 一次与 `current/` 非源码缓存清理相关的记录

## 推荐理解方式

- 如果想知道“现在应该怎么跑”：
  - 看 `README.md`
  - 看 `current/README.md`
  - 看 `linux_workstation/README.md`
  - 看 `windows/README.md`
- 如果想知道“以前为什么这样写过”或“旧逻辑长什么样”：
  - 再进 `archive/`

## English

## Archive Policy

- The maintained execution path is always under `current/`
- Content in `archive/` is kept mainly for:
  - comparing older and newer logic
  - tracing behavior at a previous point in time
  - preserving one-off cleanup or migration records
- Do not treat scripts in `archive/` as the default runnable workflow unless you are intentionally doing historical comparison

## Current Main Contents

- [20260603_code/](./20260603_code/)
  - historical code snapshot from 2026-06-03
- [current_cleanup_20260623/](./current_cleanup_20260623/)
  - record of one cleanup related to non-source cache material under `current/`

## How To Read This Folder

- If you want the current recommended workflow:
  - read `README.md`
  - read `current/README.md`
  - read `linux_workstation/README.md`
  - read `windows/README.md`
- If you want to understand older behavior or why something used to look different:
  - then inspect `archive/`
