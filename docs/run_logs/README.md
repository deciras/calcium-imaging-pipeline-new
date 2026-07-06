# Run Log Rules

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This folder stores date-based development and run logs for the repository.

本目录保存按日期归档的开发与运行日志。

## 中文

## 命名规则

日志按日期保存：

```text
docs/run_logs/YYYYMMDD.md
```

## 记录原则

1. 先读取系统日期。
2. 查看已有日志文件。
3. 按任务真实发生日期归档。
4. 如果一次任务跨越多个日期，分别更新对应日期日志。
5. 不把所有内容都堆到当天日志里。
6. 写成实验/开发进度笔记，不写成对话记录。
7. 避免“我今天……”“用户指出……”“Codex 修改……”这类口吻。
8. 优先记录真实运行、代码改动、失败原因、验证结果和下一步计划。

## 建议结构

```text
# YYYYMMDD 运行日志

## 概要
## Progress
## Issues / Notes
## Next
## Future
## 今日结论
```

## English

## Naming

Logs are stored by date:

```text
docs/run_logs/YYYYMMDD.md
```

## Logging Rules

1. Read the system date first.
2. Check whether a log file already exists.
3. File notes under the date when the work actually happened.
4. If one task spans multiple dates, update every affected date file.
5. Do not dump everything into the current day by default.
6. Write these as development or experiment progress notes, not chat transcripts.
7. Avoid phrasing such as “today I...”, “the user said...”, or “Codex changed...”.
8. Prioritize real runs, code changes, failure causes, verification results, and next steps.

## Suggested Structure

```text
# YYYYMMDD run log

## Summary
## Progress
## Issues / Notes
## Next
## Future
## Conclusion
```
