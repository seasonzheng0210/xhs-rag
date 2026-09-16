@echo off
rem M7 计划任务入口，由 xhs_rag.schedule 自动生成，勿手工修改
echo [%date% %time%] M7 sync 被触发 >> "C:\Users\Administrator\.workbuddy\2026-08-29-16-20-26\xhs-rag\data\logs\task_sync.log"
cd /d "C:\Users\Administrator\.workbuddy\2026-08-29-16-20-26\xhs-rag"
"C:\Users\Administrator\.workbuddy\binaries\python\envs\xhs-rag\Scripts\python.exe" -m xhs_rag.cli sync >> "C:\Users\Administrator\.workbuddy\2026-08-29-16-20-26\xhs-rag\data\logs\task_sync.log" 2>&1
echo [%date% %time%] M7 sync 结束, rc=%ERRORLEVEL% >> "C:\Users\Administrator\.workbuddy\2026-08-29-16-20-26\xhs-rag\data\logs\task_sync.log"
