# code snapshot 20260603

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This folder stores a historical code snapshot from 2026-06-03 for reference.

本目录保存的是 2026-06-03 的一份历史代码快照，供追溯和比较使用。

## 中文

## 作用

- 保留当时一组旧版脚本
- 方便和当前 `current/` 主线比较
- 不作为当前维护入口

## 使用边界

- 当前应优先使用 `current/` 中注册到 `run_pipeline.py` 的脚本
- 本目录下文件主要用于历史参考、差异对照和问题追溯
- 不建议把这里的脚本当成当前正式工作流直接运行，除非你明确知道自己在比较旧逻辑

## English

## Purpose

- preserve one older set of scripts
- make comparison with the maintained `current/` path easier
- avoid treating this folder as the active entry layer

## Boundary

- the maintained workflow should use the scripts under `current/` that are registered by `run_pipeline.py`
- files here are mainly for historical reference, diffing, and debugging older behavior
- do not treat this folder as the default runnable workflow unless you intentionally need the older logic
