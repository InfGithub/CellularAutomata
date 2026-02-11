# ----------------------------------------------------------------

"""
Cellualr Automata III Author: INF
"""

# ----------------------------------------------------------------

from typing import Literal, TextIO, Callable, TypedDict, Any
from types import ModuleType
from time import strftime, localtime, time
from os import makedirs, environ

# ----------------------------------------------------------------

from sys import set_int_max_str_digits
set_int_max_str_digits(0)

# ----------------------------------------------------------------

log_path: str = "latest.log"

open(log_path, mode="w", encoding="utf-8").close()
file_handle: TextIO = open(log_path, mode="a", encoding="utf-8", buffering=1)

makedirs("./rule", exist_ok=True)
makedirs("./cell", exist_ok=True)

def log(
    *texts: Any,
    level: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"] = "INFO",
    thread: str = "Main",
    **kwargs
):

    info: str = " ".join([item.__str__() for item in texts])
    text: str = f"[{strftime("%H:%M:%S", localtime())}] [{thread}/{level}]: {info}"
    file_handle.write(text + "\n")

    print(text, **kwargs)

# ----------------------------------------------------------------

class Rulizer:
    """
    规则化器的基类。

    将规则整数转换为多维规则空间的转换器。
    
    用于元胞自动机模拟，将整数规则转换为可执行的多维数组形式。
    """

    def __init__(self, rule: int, dim: int, module: ModuleType, default = None):
        """
        初始化方法。

        Parameters
        ----------
        rule : int
            规则整数, 0 <= rule < 2 ^ (2 ^ (3 ^ dim))
        dim : int
            元胞自动机的维度，取值 1 ~ 3
        default: np.ndarray
            可选、技术性，会将 self.rule_space 赋值为 default
        
        Raises
        ------
        ValueError
            当规则整数超出有效范围时抛出

        Examples
        --------
        >>> Rulizer(0b00011110, dim=1)
        """

        self.module: ModuleType = module
        self.pwvalue: int = 3 ** dim

        if default is None:
            rule_repr: bytes = format(rule, f"0{2 ** self.pwvalue}b")[::-1].encode("ascii")
            rule_repr_array: Any = self.module.frombuffer(rule_repr, dtype=self.module.uint8) - 48

            self.rule_space: Any = rule_repr_array.reshape([2] * self.pwvalue)
        else:
            self.rule_space: Any = default

    def rule_space_mirror(self):
        """
        镜像变换。

        Examples
        --------
        >>> R = Rulizer(0b00011110, dim=1)
        >>> R.rule_space_mirror()
        """

        self.rule_space: Any = self.rule_space.transpose(range(self.pwvalue)[::-1])

    def rule_space_complement(self):
        """
        补码变换。

        Examples
        --------
        >>> R = Rulizer(0b00011110, dim=1)
        >>> R.rule_space_complement()
        """

        self.rule_space: Any = self.module.flip(1 - self.rule_space)

# ----------------------------------------------------------------

class Metatizer:
    """
    元化器的基类。

    元胞自动机的容器和管理器。
    
    维护元胞状态并提供迭代方法，支持多维元胞自动机模拟。
    """

    def __init__(
        self,
        size: list[int],
        module: ModuleType,
        is_cupy: bool
    ):
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
        self.module: ModuleType = module
        self.as_strided_func: Callable = self.module.lib.stride_tricks.as_strided
        self.is_cupy: bool = is_cupy

        self.cell: Any = self.module.zeros(size, dtype=self.module.uint8)
        self.ndim: int = len(size)
        self.method_function()

    def method_function(self):
        if self.is_cupy:
            self.metertion: Callable[["Metatizer", Rulizer], None] = self.module.fuse(self._metertion)
        else:
            self.metertion: Callable[["Metatizer", Rulizer], None] = self._metertion

    def _metertion(self, rulizer: Rulizer):
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

        padded: Any = self.module.pad(self.cell, 1, mode="wrap")
        windows: Any = self.as_strided_func(
            padded, 
            shape=self.cell.shape + (3,) * self.ndim,
            strides=padded.strides * 2
        )

        flat_windows: Any = windows.reshape(-1, 3**self.ndim)
        self.cell.ravel()[:] = rulizer.rule_space[tuple(flat_windows.T)]

    def multiteration(
        self,
        rulizer: Rulizer,
        times: int,
        steps: int = 16,
        saving: bool = True
    ):
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
            result: Any = self.module.zeros((times + 1,) + self.cell.shape, dtype=self.module.uint8)
            result[0] = self.cell.copy()

        for tick in range(1, times + 1):
            self.metertion(rulizer)
            if saving:
                result[tick] = self.cell.copy()
            if tick == times or tick % steps == 0:
                log(f"迭代：{tick} / {times}")
        return result if saving else self.cell

# ----------------------------------------------------------------

class ResultType(TypedDict):
    metatizer: Metatizer
    rulizer: Rulizer
    time_taken: float
    one_rate: float
    zero_rate: float
    memory_taken: float
    mark: str

# ----------------------------------------------------------------

class CellularAutomata:
    def __init__(
        self,
        calculate_module: Literal["numpy", "cupy"] = "numpy",
        rule: int = 0b00011110,
        time: int = 32,
        size: list[int] = [64],
        log_rate: int = 4,
        is_save: bool = False,
        start_pos: list[int] = [32],
        call_func: Callable = None,
        default_rule_space: Any = None,
    ):
        self.calculate_module: Literal["numpy", "cupy"] = calculate_module
        self.is_cupy: bool = self.calculate_module == "cupy"

        if self.calculate_module == "numpy":
            import numpy
            self.module: ModuleType = numpy
        elif calculate_module == "cupy":
            import cupy
            self.module: ModuleType = cupy
            environ["CUPY_ACCELERATORS"] = "cub"
            environ["CUPY_CACHE_DIR"] = "/tmp/cupy"
            self.module.cuda.set_allocator(self.module.cuda.MemoryPool().malloc)
        else:
            raise Exception("Module not found")

        self.rule: int = rule
        self.time: int = time
        self.size: list[int] = size
        self.log_rate: int = log_rate
        self.is_save: bool = is_save
        self.start_pos: list[int] = start_pos
        self.call_func: Callable = call_func
        self.default_rule_space: Any = default_rule_space
    
    def run(self, mark: str = "") -> ResultType:
        tick_string: str = strftime(f"%Y-%m-%d-%H-%M-%S-{mark}", localtime())

        meta: Metatizer = Metatizer(self.size, self.module, self.is_cupy)
        log(f"元胞尺寸: {meta.cell.shape}，迭代次数：{self.time}")
        if mark:
            log(f"标记：{mark}")

        if self.start_pos:
            meta.cell[*self.start_pos] = 1
        elif self.call_func:
            meta.cell = self.call_func(meta.cell)

        rulizer: Rulizer = Rulizer(
            self.rule,
            dim=meta.ndim,
            default=self.default_rule_space,
            module=self.module
        )
        self.module.save(
            f"./rule/{tick_string}.rule",
            rulizer.rule_space
        )

        log(f"规则编译完毕，开始迭代。")
        start_time: float = time()
        result: Any = meta.multiteration(
            rulizer,
            times=self.time,
            steps=self.log_rate,
            saving=self.is_save
        )

        time_taken: float = time() - start_time
        one_rate: float = float(self.module.sum(result == 1) / result.size * 100)
        zero_rate: float = float(self.module.sum(result == 0) / result.size * 100)
        memory_taken: float = result.nbytes / 1048576
        log(
            f"迭代完毕，用时 {time_taken:.4f} s，1/0：{one_rate:.4f} % / "
            f" {zero_rate:.4f} %，内存占用：{memory_taken:,.4f} MB"
        )
        self.module.save(
            f"./cell/{tick_string}.cell", result
        )

        return {
            "metatizer": meta,
            "rulizer": rulizer,
            "time_taken": time_taken,
            "one_rate": one_rate,
            "zero_rate": zero_rate,
            "memory_taken": memory_taken,
            "mark": mark
        }

# ----------------------------------------------------------------

if __name__ == "__main__":
    # 示例
    ca = CellularAutomata(
        calculate_module="numpy",
        size=[2560],
        time=1440,
        is_save=False,
        log_rate=128,
        start_pos=[1440],
        rule=30
    )
    ca.run()