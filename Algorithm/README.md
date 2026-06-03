# Algorithm 工作区

本文件夹包含算法论文原始 PDF 和已转换的文本文件，用于后续分析与中文汇总。

## 目录结构

- `convert_pdfs.py`：将 `Algorithm` 下所有 PDF 转换为纯文本文件。
- `converted_texts/`：转换后生成的文本文件目录，保留每个 PDF 的可读文本。

## 使用方法

1. 进入仓库根目录：
   ```powershell
   Set-Location -Path "d:\ZJULearningFiles\SNN\Projects\Information Reconstruction\SNNInformationReconstruction"
   ```
2. 运行转换脚本：
   ```powershell
   .\.venv\Scripts\python.exe .\Algorithm\convert_pdfs.py
   ```
3. 转换结果保存在 `Algorithm/converted_texts/` 下。

## 当前已处理文件

- `NDOT/9417_NDOT_Neuronal_Dynamics_ba.pdf`
- `NDOT/Bilin_9417_NDOT_Neuronal_Dynamics_ba.pdf`
- `OTTT/2210.04195v2_OTTT.pdf`
- `OTTT/OTTT_穿越时间的在线训练.pdf`
- `S-TLLR/2306.15220v4_S-TLLR.pdf`
- `S-TLLR/STLLR_时间局部学习规则.pdf`
- `TESS/2502.01837v1.pdf`
- `TESS/TESS.pdf`
