# CellularAutomata

## 描述
这是一个用于运行 **ECA** 的 **Python** 程序，分为 **CPU** & **GPU** 版本。

![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 注意事项
1. 程序包含了由 **INFL** 提取的日志模块，会在目录下生成日志文件。
2. 请合理调控模拟规模，否则会炸内存。

## 启动程序
```bash
python main.py
```
```bash
python main_cupy.py
```

## 参数配置
程序使用代码中的 `Config` 类作为参数配置。
```python
class Config:

    Rule: int = 0b00011110        # 规则整数
    Time: int = 2097152           # 迭代次数
    Size: list[int] = [Time * 2]  # 元胞空间尺寸
    Step: int = 1024              # 日志报告间隔
    Save: bool = False            # 是否保存所有迭代状态
    Ppos: list[int] = [Time]      # 初始激活位置
```

## 环境配置
- **CPU** 版本需安装 **numpy** 包。
```bash
pip install numpy
```
- **GPU** 版本需安装 **cupy** 包，及 **Cuda** 工具链。
```bash
pip install cupy
```
- Python版本>=3.8

## 架构特点
- 双引擎设计
- 多维通用性，支持 **1~3 d**
- 内存高效，使用 **as_strided()**
- 向量化计算

## 高级功能
- 异构计算支持
- 多维规则空间映射

## 其他说明
因架构限制，内存占用为 **1 Bytes / Cell**，而非 **1 bit / Cell**。

## 参与贡献
欢迎提交 **Issue** 和 **Pull Request** !