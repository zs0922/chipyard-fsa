# FSA + PySpike 集成方案

## 1. 背景与目标

**目标**：在 PySpike（RISC-V 指令集模拟器）中运行软件，将 FSA 作为外设挂接，验证软硬件协同工作流程。

**核心约束**：
- PySpike 使用内部稀疏内存模型（`mem_t`），不暴露物理内存读写 Python API
- FSA 是一个 AXI4 Master 设备，必须通过物理地址读写数据
- 需要让 CPU（PySpike）和 FSA 共享同一块物理内存

## 2. 架构设计

### 2.1 整体架构

```
                      /dev/shm/fsa_phys_mem  (mmap, 物理内存镜像)
                ┌──────────────────────────────────────────────────┐
                │ 0x00000000: boot ROM / reserved                  │
                │ ...                                              │
                │ mem_base (0x80000000):                            │
                │   Q_data  (br × d × fp16)                        │
                │   K_data  (bc × d × fp16)                        │
                │   V_data  (bc × d × fp16)                        │
                │   O_data  (d × br × fp32)                        │
                │ ...                                              │
                │ end of physical memory                           │
                └───┬───────────────────────┬──────────────────────┘
                    │                       │
          ┌─────────┴───────┐    ┌──────────┴─────────┐
          │    PySpike       │    │     FSA 执行引擎    │
          │  (RISC-V CPU)    │    │  (功能模型/Verilator) │
          │                  │    │                     │
          │ sim.read_phys()  │    │ phy_mem.read()      │
          │ sim.write_phys() │    │ phy_mem.write()     │
          │                  │    │                     │
          │ FSAMMIO @ 0x8000 │    │                     │
          └──────────────────┘    └─────────────────────┘
```

### 2.2 同步模式：批量同步（非 cycle-level co-simulation）

FSA 和 PySpike **不同时运行**。同步只发生在：
1. CPU 写完数据并激活 FSA 时（Spike 内存 → mmap）
2. FSA 执行完毕后（mmap → Spike 内存）

这与现有 Verilator 仿真流程一致：`prepare → subprocess.run → read_back`。

## 3. 需要的代码改动

### 3.1 给 PySpike C++ 绑定添加 `read_phys` / `write_phys`

**文件**：PySpike 仓库中的 `src/main/cpp/py_module.cc`

**改动**：在 `py_sim_t` 的 pybind11 绑定中添加两个方法：

```cpp
#include <pybind11/pybind11.h>
#include <string_view>

// 在 py_sim_t 类的 .def 链中添加：

.def("read_phys", [](py_sim_t* pysim, uint64_t addr, size_t size) -> py::bytes {
    auto* ptr = reinterpret_cast<const char*>(pysim->addr_to_mem(addr));
    return py::bytes(ptr, size);
}, py::arg("addr"), py::arg("size"),
   "Read physical memory at given address. Returns bytes object.")

.def("write_phys", [](py_sim_t* pysim, uint64_t addr, py::bytes data) {
    auto sv = static_cast<std::string_view>(data);
    memcpy(pysim->addr_to_mem(addr), sv.data(), sv.size());
}, py::arg("addr"), py::arg("data"),
   "Write bytes to physical memory at given address.")
```

**说明**：`sim_t::addr_to_mem(reg_t)` 是 Spike 已有的 C++ 方法（定义在 `riscv/sim.cc`），返回指向物理内存页的原始指针。只是没有暴露给 Python。加上绑定即可。

### 3.2 共享物理内存模块（纯 Python）

**文件**：`python/shared_phys_mem.py`

```python
"""共享物理内存文件 —— PySpike 与 FSA 之间的数据桥梁"""
import mmap
import os
import struct

class SharedPhysMem:
    def __init__(self, size: int = 256 * 1024 * 1024):
        self.size = size
        self.path = "/dev/shm/fsa_phys_mem"
        if not os.path.exists(self.path):
            with open(self.path, "wb") as f:
                f.truncate(size)
        fd = os.open(self.path, os.O_RDWR)
        self.buf = mmap.mmap(fd, size, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE)

    def read(self, addr: int, size: int) -> bytes:
        return bytes(self.buf[addr : addr + size])

    def write(self, addr: int, data: bytes) -> None:
        self.buf[addr : addr + len(data)] = data

    def read_u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def write_u32(self, addr: int, value: int) -> None:
        self.write(addr, struct.pack("<I", value))
```

### 3.3 FSA 功能模型（纯 Python，可选替代 Verilator）

**文件**：`python/fsa_functional.py`

```python
"""FSA 功能模型 —— 解析指令序列并执行注意力计算 (numpy 实现)"""
import struct
import numpy as np
from shared_phys_mem import SharedPhysMem
from fsa.instructions import (
    InstructionType, FenceInstruction,
    MatrixInstruction, DMAInstruction, DMAFunc, MxFunc
)

class FSAFunctionalModel:
    """纯 Python 实现的 FSA 功能模拟器。

    从共享物理内存文件读取数据，模拟 DMA + 矩阵计算，
    结果写回共享内存。不涉及 RTL / Verilator。
    """

    def __init__(self, mem: SharedPhysMem, sa_rows: int = 4, sa_cols: int = 4):
        self.mem = mem
        self.sa_rows = sa_rows
        self.sa_cols = sa_cols
        # 片上 SRAM 模拟
        self.spad = [None] * 256    # ScratchPad (fp16 elements)
        self.acc  = [None] * 256    # Accumulator (fp32 elements)

    def run(self, instructions: list[int]):
        """执行指令序列"""
        pos = 0
        while pos < len(instructions):
            head = instructions[pos]
            i_type = head >> 29  # 高 3 位 = 指令类型

            if i_type == InstructionType.FENCE.value:
                if (head >> 26) & 1:  # stop bit
                    return
                pos += 1

            elif i_type == InstructionType.DMA.value:
                self._exec_dma(instructions[pos:pos+4])
                pos += 4

            elif i_type == InstructionType.MATRIX.value:
                self._exec_matrix(instructions[pos:pos+3])
                pos += 3

    def _exec_dma(self, inst_words: list[int]):
        """执行 DMA 指令：从共享物理内存读/写数据"""
        # 解析 16 字节 DMA 指令 (header + sram + mem)
        header, sram, mem_lo, mem_hi = inst_words

        func = (header >> 11) & 0xF
        repeat = (header >> 0) & 0x1FF

        sram_addr = (sram >> 12) & 0xFFFFF
        sram_stride = (sram >> 7) & 0x1F
        is_accum = (sram >> 22) & 1
        mem_stride_high = (sram >> 23) & 0x3F

        mem_addr = ((mem_lo & 0xFFFFF) << 0) | ((mem_hi & 0x7FFFF) << 32)
        mem_stride_low = (mem_lo >> 20) & 0x7FFF
        full_stride = ((mem_stride_high << 15) | mem_stride_low)
        if full_stride & (1 << 20):  # sign-extend
            full_stride |= ~0x1FFFFF
        size = (mem_lo >> 35) & 0x3FF  # in bytes

        for r in range(repeat):
            if func == DMAFunc.LD_SRAM.value:
                data = self.mem.read(mem_addr + r * full_stride, size)
                # 写入片上 SRAM (简化：假设 fp16 直接存)
                self.spad[sram_addr + r * sram_stride] = data
            elif func == DMAFunc.ST_SRAM.value:
                # 从 Accumulator SRAM 读，写入共享内存
                data = self.acc[sram_addr + r * sram_stride]
                self.mem.write(mem_addr + r * full_stride, data)

    def _exec_matrix(self, inst_words: list[int]):
        """执行矩阵指令 (SCORE / VALUE / NORM 等)"""
        # 解析 12 字节矩阵指令 (header + spad + acc)
        header, spad, acc = inst_words
        func = (header >> 11) & 0x1F  # MxFunc
        # ... 根据 func 执行对应的注意力操作
        # 此处可以用 numpy 模拟脉动阵列的计算
        raise NotImplementedError("Matrix execution to be implemented per MxFunc")
```

### 3.4 DMA 指令解析（公共工具，两种后端共用）

**文件**：`python/fsa_inst_parser.py`

```python
"""FSA 指令解析 —— 从 uint32 队列中提取 DMA 地址、类型等信息"""
from dataclasses import dataclass
from enum import IntEnum

class InstType(IntEnum):
    MATRIX = 0
    DMA    = 1
    FENCE  = 2

@dataclass
class ParsedDMA:
    """从 4 个 uint32 中解析出的 DMA 指令"""
    func: int          # 0=LD_SRAM, 1=ST_SRAM
    repeat: int        # 行数
    mem_addr: int      # 39-bit 片外物理地址
    mem_stride: int    # 带符号跨步 (字节)
    size: int          # 每行字节数

    @staticmethod
    def parse(words: list[int]) -> "ParsedDMA":
        """
        解析 4-word DMA 指令, 指令编码与硬件严格一致:
          word[0] = mem[31:0]   (addr[31:0])
          word[1] = mem[63:32]  (addr[38:32] | stride2[15] | size[10])
          word[2] = sram[31:0]  (addr | stride | isAccum | mem_stride1[6])
          word[3] = header[31:0] (type | sem | func | repeat)
        """
        mem_lo, mem_hi, sram, header = words

        # 验证指令类型
        assert (header >> 29) & 0x7 == InstType.DMA

        func = (header >> 12) & 0xF       # bits[15:12], 4-bit DMA func
        repeat = (header >> 3) & 0x1FF     # bits[11:3], 9-bit repeat

        # 片外物理地址: addr[38:0] = mem_hi[6:0] << 32 | mem_lo[31:0]
        mem_addr = ((mem_hi & 0x7F) << 32) | (mem_lo & 0xFFFFFFFF)

        # stride2: mem 字段 bits[53:39] = mem_hi[21:7], 15 bits
        stride2 = (mem_hi >> 7) & 0x7FFF

        # mem_stride1: sram 字段 bits[95:90] = sram[31:26], 6 bits
        mem_stride1 = (sram >> 26) & 0x3F

        # 完整跨步 = mem_stride1[5:0] << 15 | stride2[14:0], 21-bit 有符号
        full_stride = (mem_stride1 << 15) | stride2
        if full_stride & (1 << 20):  # sign-extend 21-bit
            full_stride |= ~((1 << 21) - 1)

        # size: mem 字段 bits[63:54] = mem_hi[31:22], 10 bits (字节)
        size = (mem_hi >> 22) & 0x3FF

        return ParsedDMA(func=func, repeat=repeat,
                         mem_addr=mem_addr, mem_stride=full_stride,
                         size=size)

    def is_load(self) -> bool:
        return self.func == 0

    def is_store(self) -> bool:
        return self.func == 1

    def iter_regions(self):
        """遍历该 DMA 指令覆盖的所有内存区域, 生成 (addr, size)"""
        for r in range(self.repeat):
            yield (self.mem_addr + r * self.mem_stride, self.size)


def iter_instructions(inst_queue: list[int]):
    """遍历指令队列, 按指令边界 yield (type, word_list, offset)"""
    pos = 0
    while pos < len(inst_queue):
        head = inst_queue[pos]
        i_type = (head >> 29) & 0x7

        if i_type == InstType.FENCE:
            yield (InstType.FENCE, inst_queue[pos:pos+1], pos)
            pos += 1
        elif i_type == InstType.DMA:
            yield (InstType.DMA, inst_queue[pos:pos+4], pos)
            pos += 4
        elif i_type == InstType.MATRIX:
            yield (InstType.MATRIX, inst_queue[pos:pos+3], pos)
            pos += 3
        else:
            # 未知类型, 跳过
            pos += 1
```

### 3.5 FSA_MMIO 设备（注册在 PySpike 中，支持双后端）

**文件**：`python/fsa_mmio_device.py`

```python
"""FSA MMIO 设备 —— 拦截 CPU 对 0x8000 区域的访问, 协调数据同步.

支持两种后端 (通过 --device 参数指定):
  functional                    → 纯 Python FSA 功能模型
  verilator:/path/to/sim[,arg]  → Verilator RTL 仿真
"""
import struct, threading
from typing import Optional
from riscv import dev
from riscv.sim import sim_t
from shared_phys_mem import SharedPhysMem


class FSAMMIO(dev.MMIO):

    STATE_IDLE   = 0
    STATE_ACTIVE = 1
    STATE_DONE   = 2

    REG_INST       = 0x00
    REG_START      = 0x04
    REG_STATE      = 0x08
    REG_PERF_BASE  = 0x0C   # perf counters start here, up to 0x30

    def __init__(self, sim: sim_t, args: Optional[str] = None):
        super().__init__(sim, args)
        self.state = self.STATE_IDLE
        self.inst_queue: list[int] = []
        self.mem = SharedPhysMem()
        self.perf_counters: dict[int, int] = {}
        self.backend = self._make_backend(sim, args or "")

    def _make_backend(self, sim: sim_t, args: str):
        """解析 args 字符串, 创建对应的后端"""
        if not args or args.startswith("functional"):
            from fsa_functional import FSAFunctionalModel
            return FSAFunctionalModel(self.mem)

        if args.startswith("verilator"):
            from fsa_verilator_backend import VerilatorBackend
            # 格式: verilator:/path/to/sim[,max_cycles=N][,output_dir=/tmp]
            parts = args.split(":", 1)
            config_str = parts[1] if len(parts) > 1 else ""
            return VerilatorBackend(self.mem, config_str)

        raise ValueError(f"Unknown FSA backend: {args}")

    def size(self) -> int:
        return 0x100

    # ─── CPU Store ──────────────────────────────────────────
    def store(self, addr: int, data: bytes) -> None:
        off = addr - 0x8000
        val = struct.unpack("<I", data)[0]

        if off == self.REG_INST:
            self.inst_queue.append(val)

        elif off == self.REG_START:
            if self.state == self.STATE_IDLE and val != 0:
                self.state = self.STATE_ACTIVE
                t = threading.Thread(target=self._execute, daemon=True)
                t.start()

    # ─── CPU Load ───────────────────────────────────────────
    def load(self, addr: int, size: int) -> bytes:
        off = addr - 0x8000

        if off == self.REG_STATE:
            return struct.pack("<I", self.state)

        if self.REG_PERF_BASE <= off <= 0x30:
            return struct.pack("<I", self.perf_counters.get(off, 0))

        return b"\x00" * size

    # ─── 执行 ──────────────────────────────────────────────
    def _execute(self):
        sim = self.sim

        # 1. Spike 内存 → 后端 (输入同步)
        self.backend.sync_input(sim, self.inst_queue)

        # 2. 运行
        self.backend.run(self.inst_queue)

        # 3. 后端 → Spike 内存 (输出同步)
        self.backend.sync_output(sim, self.inst_queue)

        # 4. 性能计数器
        self.perf_counters = {
            self.REG_PERF_BASE + 0x00: getattr(self.backend, "perf_exec_time", 0),
            # ... 更多计数器
        }

        self.state = self.STATE_DONE
```

### 3.6 Verilator 后端实现

**文件**：`python/fsa_verilator_backend.py`

```python
"""Verilator RTL 仿真后端 —— 将 FSA 指令放到子进程 Verilator 中执行。

数据流:
  1. sync_input: 解析 DMA LD 指令 → sim.read_phys() → ELF 文件
  2. run:        inst.bin + 启动 Verilator 子进程
  3. sync_output:解析 DMA ST 指令 → 读 dump-mem 文件 → sim.write_phys()
"""
import os, struct, subprocess, tempfile
from fsa_inst_parser import ParsedDMA, InstType, iter_instructions
from shared_phys_mem import SharedPhysMem


class VerilatorBackend:
    """将 FSA 指令序列放入 Verilator 子进程执行, 负责文件 I/O 和内存同步"""

    def __init__(self, mem: SharedPhysMem, config_str: str = ""):
        self.mem = mem
        self.perf_exec_time = 0

        # 解析配置: "path/to/sim,max_cycles=1000000,output_dir=/tmp"
        defaults = {
            "simulator_path": "",
            "max_cycles": 10000000,
            "output_dir": "/tmp/fsa_verilator",
        }
        for kv in config_str.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                defaults[k] = v
            elif kv and not defaults["simulator_path"]:
                defaults["simulator_path"] = kv

        self.simulator_path = defaults["simulator_path"]
        self.max_cycles = int(defaults["max_cycles"])
        self.output_dir = defaults["output_dir"]

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # 在 sync_input 阶段收集的信息, 供 run 和 sync_output 使用
        self._input_segments: list[tuple[int, int, bytes]] = []   # (addr, size, data)
        self._output_dumps: list[tuple[int, int]] = []             # (addr, size)

    # ═══════════════════════════════════════════════════════════
    # 第一阶段: 输入同步 (Spike 物理内存 → ELF 文件)
    # ═══════════════════════════════════════════════════════════
    def sync_input(self, sim, inst_queue: list[int]):
        """解析 DMA LD 指令, 从 Spike 物理内存读取数据, 准备 ELF 段"""
        self._input_segments.clear()
        self._output_dumps.clear()

        for i_type, words, _ in iter_instructions(inst_queue):
            if i_type != InstType.DMA:
                continue
            dma = ParsedDMA.parse(words)

            if dma.is_load():
                # 从 Spike 物理内存读取 FSA 将要访问的数据
                for addr, size in dma.iter_regions():
                    try:
                        data = sim.read_phys(addr, size)
                        self._input_segments.append((addr, size, data))
                    except Exception as e:
                        print(f"[VerilatorBackend] read_phys({hex(addr)}, {size}) failed: {e}")
                        raise

            elif dma.is_store():
                # 记录输出区域, 等 Verilator 跑完后回读
                for addr, size in dma.iter_regions():
                    self._output_dumps.append((addr, size))

    # ═══════════════════════════════════════════════════════════
    # 第二阶段: 运行 Verilator
    # ═══════════════════════════════════════════════════════════
    def run(self, inst_queue: list[int]):
        """写指令和内存文件, 启动 Verilator 子进程, 等待完成"""

        # 写指令文件 inst.bin (uint32 little-endian)
        inst_file = os.path.join(self.output_dir, "inst.bin")
        inst_bytes = struct.pack(f"{len(inst_queue)}I", *inst_queue)
        with open(inst_file, "wb") as f:
            f.write(inst_bytes)

        # 写内存 ELF 文件 (复用现有 fsa/utils.py 的 ElfWriter)
        mem_file = os.path.join(self.output_dir, "mem.elf")
        self._write_elf(mem_file)

        # 构造 Verilator 命令行
        cmd = [self.simulator_path, inst_file]
        cmd.append(f"+loadmem={mem_file}")
        cmd.append(f"+max-cycles={self.max_cycles}")

        # 为每个输出区域设置 +dump-mem
        self._dump_files = []
        for addr, size in self._output_dumps:
            fname = os.path.join(self.output_dir, f"{addr:016x}.bin")
            self._dump_files.append((fname, addr, size))
            cmd.append(f"+dump-mem={fname}:{addr:#x}:{size:#x}")

        print(f"[VerilatorBackend] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[VerilatorBackend] stderr:\n{result.stderr}")
            raise RuntimeError(f"Verilator exited with code {result.returncode}")

        # 从 Verilator 输出中提取性能计数器
        for line in result.stdout.splitlines():
            if "FSA: perfCnt_execTime" in line:
                self.perf_exec_time = int(line.split("=")[-1].strip())
                break

    # ═══════════════════════════════════════════════════════════
    # 第三阶段: 输出同步 (dump-mem 文件 → Spike 物理内存)
    # ═══════════════════════════════════════════════════════════
    def sync_output(self, sim, inst_queue: list[int]):
        """读 Verilator 输出的 dump 文件, 写回 Spike 物理内存"""
        for fname, addr, size in self._dump_files:
            try:
                data = self.mem.read_from_file(fname, size)  # mmap 文件可直接读
                sim.write_phys(addr, data)
            except Exception as e:
                print(f"[VerilatorBackend] write_phys({hex(addr)}, {size}) failed: {e}")
                raise

    # ═══════════════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════════════
    def _write_elf(self, filename: str):
        """将收集的输入数据段写入 ELF 文件 (格式与 engine.py 一致)"""
        from fsa.utils import ElfWriter
        from fsa.config import get_config
        writer = ElfWriter(self._input_segments, get_config().mem_align)
        writer.write_elf(filename)
```

### 3.7 SharedPhysMem 补充方法

```python
# 在 shared_phys_mem.py 的 SharedPhysMem 类中添加:

def read_from_file(self, path: str, size: int) -> bytes:
    """从文件读取数据并写入共享 mmap (用于 Verilator dump-mem 回读)"""
    with open(path, "rb") as f:
        data = f.read(size)
    return data
```

## 4. Bare-metal 程序（跑在 PySpike 中的 RISC-V 代码）

```c
// attention_test.c — 编译为 RISC-V bare-metal ELF
// 编译: riscv64-unknown-elf-gcc -march=rv64gc -O2 -nostdlib -T link.ld attention_test.c -o attention_test.elf

#define FSA_MMIO_BASE  0x00008000
#define FSA_INST       *(volatile uint32_t *)(FSA_MMIO_BASE + 0x00)
#define FSA_START      *(volatile uint32_t *)(FSA_MMIO_BASE + 0x04)
#define FSA_STATE      *(volatile uint32_t *)(FSA_MMIO_BASE + 0x08)

// 物理地址 (需与 config.json 的 mem_base 对应)
#define Q_ADDR  0x80000000
#define K_ADDR  0x80001000
#define V_ADDR  0x80002000
#define O_ADDR  0x80003000

// 在此处静态嵌入 FSA 指令序列
// 每条 DMA 指令 4 个 32-bit word, 每条 Matrix 指令 3 个 32-bit word
// 实际使用时可通过 Python 脚本生成此文件
static const uint32_t fsa_instructions[] = {
    // DMA LOAD Q: header
    0x0000XXXX,  // [31:29]=DMA, [15:11]=LD_SRAM, [9:0]=repeat
    // DMA LOAD Q: sram (addr, stride, ...)
    0x0000XXXX,
    // DMA LOAD Q: mem_lo (addr[19:0], stride_low, size)
    0x0000XXXX,
    // DMA LOAD Q: mem_hi (addr[38:20], ...)
    0x0000XXXX,

    // ... 更多指令 ...

    // FENCE + STOP
    0x04000000,  // bit[26]=stop
};

void main() {
    // 1. 将输入数据写入物理内存 (简化: 用固定测试数据)
    uint16_t *Q = (uint16_t *)Q_ADDR;
    uint16_t *K = (uint16_t *)K_ADDR;
    uint16_t *V = (uint16_t *)V_ADDR;
    for (int i = 0; i < BR * D; i++) Q[i] = test_data[i];
    for (int i = 0; i < BC * D; i++) K[i] = test_data[i + BR*D];

    // 2. 推送指令队列
    int n_inst = sizeof(fsa_instructions) / sizeof(uint32_t);
    for (int i = 0; i < n_inst; i++) {
        FSA_INST = fsa_instructions[i];
    }

    // 3. 激活 FSA
    FSA_START = 0xFFFFFFFF;

    // 4. 轮询等待完成
    while (FSA_STATE != 2) {
        __asm__ volatile("nop");
    }

    // 5. 验证结果
    uint32_t *O = (uint32_t *)O_ADDR;
    // compare O with expected ...
}
```

## 5. 完整的数据流

### 5.1 功能模型后端

```
CPU (PySpike)                          FSA 功能模型 (同一进程)
────────────                           ────────────────────────
  sw Q_ADDR, q_data                    FSAMMIO.store(0x8004)
  sw K_ADDR, k_data                       │
  sw V_ADDR, v_data                       ├─ sim.read_phys(Q_ADDR) → mmap
  sw 0x8000, inst[0..N]                   ├─ sim.read_phys(K_ADDR) → mmap
  sw 0x8004, 0xFFFFFFFF                   ├─ fsa.run(inst_queue)
       ↓                                  │    ├─ 从 mmap 读数据
  while (lw 0x8008 != 2) ;                │    ├─ DMA + 矩阵计算
       ↓                                  │    └─ 结果写入 mmap
  lw O_ADDR, result                       └─ sim.write_phys(O_ADDR, mmap_data)
                                       STATE = DONE
```

### 5.2 Verilator 后端

```
CPU (PySpike)                          VerilatorBackend (同一进程, 协调)
────────────                           ──────────────────────────────────
  sw Q_ADDR, q_data                    FSAMMIO.store(0x8004)
  sw K_ADDR, k_data                       │
  sw 0x8000, inst[0..N]                   │   ┌── 第一步: 输入同步 ──┐
  sw 0x8004, 0xFFFFFFFF                   ├──→│ 解析 DMA LD 指令      │
       ↓                                  │   │ sim.read_phys(addr)   │
  while (lw 0x8008 != 2) ;                │   │ → mem.elf             │
       ↓                                  │   └──────────────────────┘
  lw O_ADDR, result                       │   ┌── 第二步: 执行 ─────┐
                                          ├──→│ 写 inst.bin          │
                                          │   │ subprocess.run(       │
                                          │   │   verilator,          │
                                          │   │   inst.bin,           │
                                          │   │   +loadmem=mem.elf,   │
                                          │   │   +dump-mem=...       │
                                          │   │ )                     │
                                          │   └──────────────────────┘
                                          │   ┌── 第三步: 输出同步 ──┐
                                          └──→│ 读 dump-mem 输出文件  │
                                              │ sim.write_phys(addr)  │
                                              └──────────────────────┘
                                          STATE = DONE
```

### 5.3 调用命令

```bash
# 功能模型 (无需 RTL 编译)
pyspike \
  --extlib=fsa_mmio_device.py,fsa_functional.py,shared_phys_mem.py,fsa_inst_parser.py \
  --device=fsa_mmio_device.FSAMMIO,0x8000,functional \
  attention_test.elf

# Verilator RTL 仿真 (cycle-accurate)
pyspike \
  --extlib=fsa_mmio_device.py,fsa_verilator_backend.py,fsa_inst_parser.py,shared_phys_mem.py \
  --device=fsa_mmio_device.FSAMMIO,0x8000,verilator:/path/to/simulator,max_cycles=5000000 \
  attention_test.elf
```

`--device` 参数格式：`类名,基址,后端类型[:配置]`。后端类型可以为 `functional` 或不填（默认功能模型），或 `verilator:仿真器路径[,max_cycles=N][,output_dir=/tmp]`。

## 6. 地址空间约定

| 地址范围 | 用途 | 说明 |
|----------|------|------|
| `0x00000000 - 0x7FFFFFFF` | 保留 | Boot ROM, Spike 内部 |
| `0x00008000 - 0x000080FF` | FSA 配置寄存器 | CPU 通过 MMIO 访问 |
| `0x80000000 - 0xBFFFFFFF` | 共享数据内存 | mem_base = 0x80000000, 1GB |
| 片上 SRAM (内部) | ScratchPad + Acc | 仅在 FSA 功能模型内部, 对 CPU 不可见 |

**关键：`mem_base` 必须在 PySpike 的物理内存布局中可访问。** 通过在 `cfg_t.mem_layout` 中配置：

```python
from riscv.config import mem_cfg_t
cfg = sim.cfg
cfg.mem_layout = [mem_cfg_t(base=0x80000000, size=0x40000000)]  # 1GB
```

## 7. 改造清单

| 序号 | 改动项 | 文件 | 类型 | 工作量 |
|------|--------|------|------|--------|
| 1 | 给 PySpike 加 `read_phys`/`write_phys` 绑定 | `src/main/cpp/py_module.cc` | C++ 补丁 | ~20 行 |
| 2 | 共享物理内存模块 | `python/shared_phys_mem.py` | 新增 | ~60 行 |
| 3 | DMA 指令解析工具 | `python/fsa_inst_parser.py` | 新增 | ~80 行 |
| 4 | FSA 功能模型（DMA + 矩阵计算） | `python/fsa_functional.py` | 新增 | ~200 行 |
| 5 | Verilator 后端 | `python/fsa_verilator_backend.py` | 新增 | ~130 行 |
| 6 | FSA_MMIO 设备（PySpike 插件，双后端） | `python/fsa_mmio_device.py` | 新增 | ~120 行 |
| 7 | Bare-metal 测试程序 | `test/attention_test.c` | 新增 | ~80 行 |
| 8 | 集成脚本 (指令生成 + 启动) | `python/run_with_spike.py` | 新增 | ~80 行 |

**总计：~20 行 C++ + ~670 行 Python + ~80 行 C**

## 8. 两种后端对比

| 维度 | 功能模型 | Verilator 后端 |
|------|----------|---------------|
| 精度 | numpy fp16/fp32, 与 RTL 在数值上可能不一致 | 100% RTL 精确 (EasyFloat) |
| 速度 | 秒级 (纯 Python numpy) | 分钟级 (RTL 仿真) |
| 依赖 | 零依赖 | 需编译 Verilator 仿真器 (`make CONFIG=...`) |
| 输入同步 | sim.read_phys → mmap | sim.read_phys → mem.elf |
| 输出同步 | mmap → sim.write_phys | dump-mem 文件 → sim.write_phys |
| 性能计数 | 粗略 | 精确 cycle 计数 |
| 波形 | 不支持 | 支持 `+vcdfile=` |
| 用途 | 快速软件迭代, 指令序列验证 | 硬件精度验证, 性能分析 |

## 9. 与现有流程的对比

| | 现有 Verilator | 现有 FPGA | 本方案 (PySpike + 功能模型) | 本方案 (PySpike + Verilator) |
|---|---|---|---|---|
| CPU | 无 (main.py 模拟) | 无 (Host mmio) | PySpike RISC-V | PySpike RISC-V |
| 数据准备 | numpy → mem.elf | dev_write → HBM | CPU sw 指令写物理内存 | CPU sw 指令写物理内存 |
| 指令推送 | inst.bin 文件 | dev_queue_mmio_write | CPU sw 指令写 0x8000 | CPU sw 指令写 0x8000 |
| FSA 执行 | Verilator 子进程 | FPGA 硬件 | Python 功能模型 | Verilator 子进程 |
| 结果读取 | +dump-mem 文件 | dev_read C2H | mmap → sim.write_phys | dump-mem → sim.write_phys |
| 同步粒度 | 一次 subprocess | 一次 MMIO poll | CPU 轮询 0x8008 | CPU 轮询 0x8008 |
