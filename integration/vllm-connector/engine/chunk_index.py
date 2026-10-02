"""chunk hash → 存储路径 索引 + LRU + pin（engine 通用组件，D-010/D-011）。

索引与介质完全解耦（D-011）：一个 KV chunk 的身份是**文件路径**
（一个 chunk = 一个数据文件），本模块只做 ``chunk_key → 路径``
的映射、LRU 驱逐与 pin 保护；路径的具体形态完全由 backend 决定
并解释，本模块只当 opaque 字符串处理。

hash 体系为父链式（D-007，对齐 vllm fork 的 RFC#45702 体系）：
H_0 = blake2b(b"" ‖ tokens_0)，H_i = blake2b(H_{i-1} ‖ tokens_i)，
H_i 指纹整个前缀 [0, (i+1)×chunk)，将来 chunk 内部分命中直接用细粒度
边界 hash 做 key，无需换 key 体系。

内部状态（写死，供测试断言）：
- free: list[str]                    路径空闲池（栈，尾部为栈顶）
- stored: OrderedDict[bytes, str]    key → 路径，LRU 顺序（头=LRU，尾=MRU）
- pinned: dict[str, int]             路径 → refcount
- pending_store: dict[bytes, str]    已分配未确认落盘的 key → 路径
"""

import hashlib

import numpy as np
from collections import OrderedDict

CHUNK_SIZE = 256  # 对齐 LMCache 默认 chunk_size；构造参数可覆盖


def hash_chunk(tokens: tuple[int, ...], parent: bytes = b"") -> bytes:
    """父链式 blake2b(digest_size=16)：H = blake2b(parent ‖ enc(tokens))。

    D-007：H_i = blake2b(H_{i-1} ‖ enc(tokens_i))，H_0 的 parent 为
    b""（默认值，与无 parent 调用兼容）。同 tokens + parent 必同 key。

    编码：每个 int 的十进制 ASCII 串前加 4 字节小端长度前缀，保证不同
    tuple 一定产生不同字节流（如 (1, 2) 与 (12,) 不可混淆）。
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(parent)
    # Fixed-width little-endian int64 encoding: unambiguous and one C call,
    # instead of a per-token Python loop (dominant admit-path cost at 8k+
    # token prompts: ~3% of EngineCore CPU in profiles). Changes the on-disk
    # key format; stores are rebuilt per server bring-up.
    h.update(np.asarray(tokens, dtype=np.int64).tobytes())
    return h.digest()


class ChunkIndex:
    """token chunk → 存储路径 的索引与分配器（零依赖纯逻辑）。

    构造时注入路径池 ``paths``（backend 生成并解释）；本模块只负责
    在池内分配/驱逐/pin，不知道路径指向什么介质。
    """

    def __init__(self, paths: "list[str] | int", chunk_size: int = CHUNK_SIZE):
        """paths：路径字符串列表（backend 生成）；为兼容旧调用方，
        也接受 int（退化为 ``slot://<i>`` 路径池，仅测试用）。"""
        if isinstance(paths, int):
            if paths < 1:
                raise ValueError(f"paths must be >= 1, got {paths}")
            paths = [f"slot://{i}" for i in range(paths)]
        if not paths:
            raise ValueError("paths must be a non-empty list")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._paths = list(paths)
        self._chunk_size = chunk_size
        # 内部状态（写死）
        self.free: list[str] = list(paths)  # 栈
        self.stored: "OrderedDict[bytes, str]" = OrderedDict()  # 尾 = MRU
        self.pinned: dict[str, int] = {}  # 路径 → refcount
        self.pending_store: dict[bytes, str] = {}

    # ---------- 属性 ----------

    @property
    def capacity(self) -> int:
        return len(self._paths)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    # ---------- 查询 ----------

    def lookup_prefix(self, token_ids: list[int]) -> int:
        """从头逐 chunk 查 stored（pending 不算命中），返回命中 token 数
        （= 命中 chunk 数 × chunk_size，遇首个未命中停止）。

        链式滚动（D-007）：一次遍历维护 parent = H_{i-1}（首 chunk 为
        b""），chunk i 的 key = hash_chunk(tokens_i, parent)。语义不变，
        仍返回命中 token 数。尾部不满一个 chunk 的部分不参与查询。
        """
        cs = self._chunk_size
        hit_chunks = 0
        parent = b""
        i = 0
        while i + cs <= len(token_ids):
            key = hash_chunk(tuple(token_ids[i : i + cs]), parent=parent)
            if key in self.stored:
                hit_chunks += 1
                parent = key
            else:
                break
            i += cs
        return hit_chunks * cs

    def keys_and_last_parent(
        self,
        token_ids: list[int],
        start_chunk: int = 0,
        parent: bytes = b"",
    ) -> "tuple[list[bytes], bytes]":
        """链式计算第 start_chunk 个 chunk 起每个完整 chunk 的 key。

        增量接口（D-007 / Rework-1）：``parent`` 为 H_{start_chunk-1}
        （无前缀时传 b""，即在该处重新起链）。返回
        ``(keys, last_parent)``：keys 与 chunk 同序；last_parent 为最后
        一个产出 chunk 的 H_j（供下次增量续算），keys 为空时原样返回
        传入的 parent。尾部不满一个 chunk 的部分舍弃。
        """
        cs = self._chunk_size
        keys: list[bytes] = []
        cur = bytes(parent)
        i = max(start_chunk, 0) * cs
        while i + cs <= len(token_ids):
            cur = hash_chunk(tuple(token_ids[i : i + cs]), parent=cur)
            keys.append(cur)
            i += cs
        return keys, cur

    def keys_for_tokens(
        self,
        token_ids: list[int],
        start_chunk: int = 0,
        parent: bytes = b"",
    ) -> list[bytes]:
        """第 start_chunk 个 chunk 起每个完整 chunk 的 key（尾部不满
        chunk 舍弃）。

        链式：``keys_and_last_parent`` 的列表视图；需要增量续算时用
        :meth:`keys_and_last_parent` 拿 last_parent。
        """
        return self.keys_and_last_parent(token_ids, start_chunk, parent)[0]

    # ---------- pin 辅助 ----------

    def _pin_path(self, path: str) -> None:
        self.pinned[path] = self.pinned.get(path, 0) + 1

    def _unpin_path(self, path: str) -> None:
        rc = self.pinned.get(path, 0)
        if rc <= 0:
            raise ValueError(f"path {path!r} is not pinned")
        if rc == 1:
            del self.pinned[path]
        else:
            self.pinned[path] = rc - 1

    # ---------- 分配 / 提交 ----------

    def allocate(self, keys: list[bytes]) -> "tuple[list[str], list[bytes]] | None":
        """为不在 stored/pending 的 keys 分配路径。

        free 不足时从 stored 头部（LRU）驱逐不在 pinned 的项直到够；
        仍不够 → None（无副作用）。成功 → (paths 与新分配的 keys
        同序, evicted_keys)，并记 pending + pin。全已存在 → ([], [])。
        """
        new_keys: list[bytes] = []
        seen: set = set()
        for k in keys:
            if k in self.stored or k in self.pending_store or k in seen:
                continue
            seen.add(k)
            new_keys.append(k)
        if not new_keys:
            return ([], [])

        need = len(new_keys)
        evicted_keys: list[bytes] = []
        if len(self.free) < need:
            # 先无副作用地选出待驱逐 key（LRU 头起，跳过 pinned）
            to_evict: list[bytes] = []
            for k in self.stored:
                if len(self.free) + len(to_evict) >= need:
                    break
                if self.pinned.get(self.stored[k], 0) == 0:
                    to_evict.append(k)
            if len(self.free) + len(to_evict) < need:
                return None
            evicted_keys = to_evict
            for k in to_evict:
                p = self.stored.pop(k)
                self.free.append(p)

        paths: list[str] = []
        for k in new_keys:
            p = self.free.pop()
            self.pending_store[k] = p
            self._pin_path(p)
            paths.append(p)
        return (paths, evicted_keys)

    def complete_store(self, keys: list[bytes], success: bool = True) -> None:
        """success → pending 移入 stored（MRU 端）解 pin；
        False → 路径回 free 解 pin。"""
        for k in keys:
            p = self.pending_store.pop(k)
            if success:
                self.stored[k] = p
                self.stored.move_to_end(k)  # MRU 端
            else:
                self.free.append(p)
            self._unpin_path(p)

    # ---------- pin / LRU ----------

    def pin(self, keys: list[bytes]) -> list[str]:
        """查 stored 并 pin（每个 key 的路径 refcount+1），返回路径与
        keys 同序；任一 key 不在 stored → KeyError（不产生部分 pin 的
        半截状态）。"""
        paths: list[str] = []
        for k in keys:
            if k not in self.stored:
                raise KeyError(k)
            paths.append(self.stored[k])
        for p in paths:
            self._pin_path(p)
        return paths

    def unpin(self, keys: list[bytes]) -> None:
        """对 stored 中的 key 解 pin；key 不在 stored → KeyError。"""
        for k in keys:
            p = self.stored[k]
            self._unpin_path(p)

    def touch(self, keys: list[bytes]) -> None:
        """把存在的 key 移到 stored 尾部（MRU）；不存在的忽略。"""
        for k in keys:
            if k in self.stored:
                self.stored.move_to_end(k)

    def reset(self) -> None:
        """清空 stored/pinned/pending，free 恢复满。"""
        self.free = list(self._paths)
        self.stored.clear()
        self.pinned.clear()
        self.pending_store.clear()


# ---------------------------------------------------------------------------
# kado_native backend: the same index implemented in Rust (kado/native/).
# Interface-compatible, including read-only .stored / .pending_store views.
try:  # pragma: no cover - exercised when the wheel is installed
    import kado_native as _kn
except ImportError:
    _kn = None

PyChunkIndex = ChunkIndex

if _kn is not None:
    import logging as _logging
    _logging.getLogger("kado").info("chunk index: kado_native (Rust) backend active")

    class _View:
        def __init__(self, contains, length, get=None):
            self._c, self._l, self._g = contains, length, get
        def __contains__(self, k): return self._c(bytes(k))
        def __len__(self): return self._l()
        def get(self, k, default=None):
            v = self._g(bytes(k)) if self._g is not None else None
            return default if v is None else v

    class ChunkIndex:  # noqa: F811 - native replacement
        def __init__(self, paths, chunk_size: int = CHUNK_SIZE):
            if isinstance(paths, int):
                paths = [f"slot://{i}" for i in range(paths)]
            self._n = _kn.ChunkIndex(list(paths), chunk_size)
            self.stored = _View(self._n.stored_contains, self._n.stored_len, self._n.stored_get)
            self.pending_store = _View(self._n.pending_contains, self._n.pending_len)
            self.pinned = _View(lambda k: False, self._n.pinned_len)
            self.free = _View(lambda k: False, self._n.free_len)
        @property
        def capacity(self): return self._n.capacity
        @property
        def chunk_size(self): return self._n.chunk_size
        def lookup_prefix(self, token_ids): return self._n.lookup_prefix(list(token_ids))
        def keys_and_last_parent(self, token_ids, start_chunk=0, parent=b""):
            ks, lp = self._n.keys_and_last_parent(list(token_ids), start_chunk, bytes(parent))
            return list(ks), lp
        def keys_for_tokens(self, token_ids, start_chunk=0, parent=b""):
            return list(self._n.keys_for_tokens(list(token_ids), start_chunk, bytes(parent)))
        def allocate(self, keys):
            r = self._n.allocate([bytes(k) for k in keys])
            return None if r is None else (r[0], list(r[1]))
        def complete_store(self, keys, success=True): self._n.complete_store([bytes(k) for k in keys], success)
        def pin(self, keys): return self._n.pin([bytes(k) for k in keys])
        def unpin(self, keys): self._n.unpin([bytes(k) for k in keys])
        def touch(self, keys): self._n.touch([bytes(k) for k in keys])
        def reset(self): self._n.reset()
