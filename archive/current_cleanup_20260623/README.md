# current cleanup 20260623

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This archive preserves a cleanup snapshot related to non-source cache material
inside `current/`.

这个归档保存的是一次与 `current/` 目录中非源码缓存材料相关的清理快照。

## 中文

## 用途

本归档只用于记录当时从 `current/` 中移走的非源码缓存文件，不表示这些文件属于当前维护主线。

## 本次移动

- `current/analysis/__pycache__/`
  - 移动到 `archive/current_cleanup_20260623/pycache_backup/current/analysis__pycache__/`

## 未移动

- `current/preprocess/`
- `current/roi/`
- `current/preprocess/__pycache__/`
- `current/roi/__pycache__/`
- `current/run_pipeline.py`
- `current/pipeline_dashboard_gui.py`

原因：

- 当时明确要求 `05` 及之前相关内容不动
- `preprocess`、`roi/manual` 和总入口仍然属于当前维护路径

## 判断

`current/` 中的源码脚本仍然按 `current/run_pipeline.py` 注册或被 wrapper / 文档引用。本归档没有宣称任何源码脚本已经废弃，只是保存一次缓存清理记录。

## English

## Purpose

This archive only records one cleanup pass that moved non-source cache files
out of `current/`. It does not define the maintained source path.

## Moved in This Cleanup

- `current/analysis/__pycache__/`
  - moved to `archive/current_cleanup_20260623/pycache_backup/current/analysis__pycache__/`

## Not Moved

- `current/preprocess/`
- `current/roi/`
- `current/preprocess/__pycache__/`
- `current/roi/__pycache__/`
- `current/run_pipeline.py`
- `current/pipeline_dashboard_gui.py`

Reason:

- the request at that time was to avoid touching content related to step `05` and earlier
- `preprocess`, `roi/manual`, and the main launcher were still part of the maintained path

## Interpretation

The source scripts under `current/` were still registered by
`current/run_pipeline.py` or referenced by wrappers and documentation. This
archive is a cache-cleanup snapshot, not a declaration that source scripts were
retired.
