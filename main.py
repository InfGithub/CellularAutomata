"""
基于 NumPy 的初等元胞自动机模型。

提供规则转换、元胞管理和迭代计算功能，支持高效的多维元胞自动机模拟。
包含日志记录功能，便于调试和监控模拟过程。
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from time import time, strftime, localtime
from typing import Literal
from os import makedirs

makedirs("./rule", exist_ok=True)
makedirs("./cell", exist_ok=True)

_log_path: str = "latest.log"
with open(_log_path, mode="w+", encoding="utf-8") as log_file:
    log_file.close()

def log(*texts: object,
        level: Literal["TRACE", "DEBUG", "INFO","WARN", "ERROR", "FATAL"] = "INFO",
        thread: str = "Main", **kwargs) -> tuple[object]:
    global _log_path

    text: str = f"[{strftime("%H:%M:%S", localtime())}] [{thread}/{level\
        }]: {" ".join([t.__str__() for t in texts])}"
    with open(_log_path, mode="a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")

    print(text, **kwargs)
    return texts

class Rulizer:
    """
    规则化器的基类。

    将规则整数转换为多维规则空间的转换器。
    
    用于元胞自动机模拟，将整数规则转换为可执行的多维数组形式。
    """

    def __init__(self, rule: int, dim: int, default: np.ndarray = None):
        """
        初始化方法。

        Parameters
        ----------
        rule : int
            规则整数, 0 <= rule < 2 ^ (2 ^ (3 ^ dim))
        dim : int
            元胞自动机的维度，取值 1 ~ 3
        default: cp.ndarray
            可选、技术性，会将 self.rule_space 赋值为 default
        
        Raises
        ------
        ValueError
            当规则整数超出有效范围时抛出

        Examples
        --------
        >>> Rulizer(0b00011110, dim=1)
        """

        self.pwvalue: int = 3 ** dim

        if default is None:
            rule_repr: bytes = format(rule, f"0{2 ** self.pwvalue}b")[::-1].encode("ascii")
            rule_repr_array: np.ndarray = np.frombuffer(rule_repr, dtype=np.uint8) - 48

            self.rule_space: np.ndarray = rule_repr_array.reshape([2] * self.pwvalue)
        else:
            self.rule_space: np.ndarray = default

    def rule_space_mirror(self):
        """
        镜像变换。

        Examples
        --------
        >>> R = Rulizer(0b00011110, dim=1)
        >>> R.rule_space_mirror()
        """

        self.rule_space = self.rule_space.transpose(range(self.pwvalue)[::-1])

    def rule_space_complement(self):
        """
        补码变换。

        Examples
        --------
        >>> R = Rulizer(0b00011110, dim=1)
        >>> R.rule_space_complement()
        """

        self.rule_space = np.flip(1 - self.rule_space)

class Metatizer:
    """
    元化器的基类。

    元胞自动机的容器和管理器。
    
    维护元胞状态并提供迭代方法，支持多维元胞自动机模拟。
    """

    def __init__(self, size: list[int]):
        """
        初始化方法。

        Parameters
        ----------
        size : list[int]
            空间尺寸
        
        Examples
        --------
        >>> Metatizer([16, 16])
        """

        self.cell: np.ndarray = np.zeros(size, dtype=np.uint8)
        self.ndim: int = len(size)

    def metertion(self, rulizer: Rulizer):
        """
        迭代方法，会修改 self.cell 的值。

        Parameters
        ----------
        rulizer : Rulizer
            迭代使用的规则化器

        Notes
        -----
        - 使用周期边界条件 (wrap mode)
        - 通过 as_strided() 创建内存视图，避免数据复制
        - 将多维窗口展平以应用规则
        
        Examples
        --------
        >>> meta = Metatizer([16, 16])
        >>> rulizer = Rulizer(0b00011110, dim=2)
        >>> meta.Metertion(rulizer)
        """

        padded: np.ndarray = np.pad(self.cell, 1, mode="wrap")
        windows: np.ndarray = as_strided(padded, 
                           shape=self.cell.shape + (3,) * self.ndim,
                           strides=padded.strides * 2)

        flat_windows: np.ndarray = windows.reshape(-1, 3**self.ndim)

        self.cell.ravel()[:] = rulizer.rule_space[tuple(flat_windows.T)]

    def multiteration(self, rulizer: Rulizer, times: int, steps: int = 16, saving: bool = True) -> np.ndarray:
        """
        多重迭代方法。

        Parameters
        ----------
        rulizer : Rulizer
            迭代使用的规则化器
        times : int
            迭代次数
        steps : int = 16
            报告日志的间隔迭代次数
        saving : bool
            是否存储并返回迭代时每刻元胞群的状态
        
        Returns
        -------
        result : np.ndarray
            迭代结果

            当 saving = True 时返回 N+1 维张量，其中包含每刻元胞群的状态。

            当 saving = False 时返回 N 维张量，其为最后一刻元胞群的状态。

        Examples
        --------
        >>> meta = Metatizer([16, 16])
        >>> rulizer = Rulizer(0b00011110, dim=2)
        >>> meta.Multiteration(rulizer, times=64, steps=16)
        """
        if saving:
            result: np.ndarray = np.zeros((times + 1,) + self.cell.shape, dtype=np.uint8)
            result[0] = self.cell.copy()

        for tick in range(1, times + 1):
            self.metertion(rulizer)
            if saving:
                result[tick] = self.cell.copy()
            if tick == times or tick % steps == 0:
                log(f"迭代：{tick} / {times}")
        return result if saving else self.cell

class Config:
    """
    程序运行参数配置的基类。
    """

    Rule: int = 0b00011110
    Time: int = 2097152
    Size: list[int] = [Time * 2]
    Step: int = 1024
    Save: bool = False
    Ppos: list[int] = [Time]

def main():
    """
    主函数。
    
    1. 初始化规则和参数
    2. 创建元胞空间并设置初始状态
    3. 执行多次迭代
    4. 保存结果并输出统计信息
    """

    meta: Metatizer = Metatizer(Config.Size)
    log(f"元胞尺寸: {meta.cell.shape}，迭代次数：{Config.Time}")

    meta.cell[*Config.Ppos] = 1
    rulizer: Rulizer = Rulizer(Config.Rule, dim=meta.ndim)
    rulizer.rule_space.tofile("./rule/result.rule")

    log(f"规则编译完毕，开始迭代。")
    start_time: float = time()
    result = meta.multiteration(rulizer, times=Config.Time, steps=Config.Step, saving=Config.Save)

    log(f"""迭代完毕，用时 {time() - start_time:.4f} s，1/0：{np.sum(result == 1) /
    result.size * 100:.4f} % / {np.sum(result == 0) /
    result.size * 100:.4f} %，内存占用：{result.nbytes / 1048576:,.4f} MB""")
    result.tofile("./cell/result.cell")

if __name__ == "__main__":
    main()