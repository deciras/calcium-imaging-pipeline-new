# Windows Launchers

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This folder holds thin Windows launchers for the shared pipeline and dashboard
code. It is not a separate Windows copy of the analysis logic.

本目录保存的是 Windows 平台的薄启动层，不是另一套独立的 Windows 分析主代码。

## 中文

## 设计边界

Windows 端应该只负责：

- 找到 `conda.exe`
- 选择 dashboard 环境名
- 从仓库根目录启动共享主代码

不应该复制：

- `current/run_pipeline.py`
- `current/pipeline_dashboard_gui.py`
- 各 step 分析脚本

## 当前文件

- [launch_pipeline_dashboard.bat](./launch_pipeline_dashboard.bat)
  - 适合双击或 `cmd.exe` 启动
- [launch_pipeline_dashboard.ps1](./launch_pipeline_dashboard.ps1)
  - 适合 PowerShell 启动和稍强的参数传递

## 默认约定

- 默认 dashboard 环境名：`dashboard_gui`
- 如果未设置 `CONDA_BIN`，会优先尝试这些常见路径：
  - `%USERPROFILE%\anaconda3\Scripts\conda.exe`
  - `%USERPROFILE%\miniconda3\Scripts\conda.exe`
  - `C:\ProgramData\anaconda3\Scripts\conda.exe`
  - `C:\ProgramData\miniconda3\Scripts\conda.exe`

## English

## Boundary

The Windows layer should only:

- locate `conda.exe`
- choose the dashboard environment
- launch the shared repository code from the repo root

It should not copy:

- `current/run_pipeline.py`
- `current/pipeline_dashboard_gui.py`
- the numbered analysis scripts

## Current Files

- [launch_pipeline_dashboard.bat](./launch_pipeline_dashboard.bat)
  - suitable for double-click or `cmd.exe`
- [launch_pipeline_dashboard.ps1](./launch_pipeline_dashboard.ps1)
  - suitable for PowerShell and slightly cleaner argument forwarding

## Default Conventions

- default dashboard environment: `dashboard_gui`
- if `CONDA_BIN` is not set, the launchers try common install paths first
