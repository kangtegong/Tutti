"""WorkerImpl — vllm worker 侧回调 → TuttiEngine 执行态翻译（T-114，D-005）。

ARCHITECTURE.md §4 映射（本模块的全部职责）：

    start_load_kv        → engine.load_layer(plan, 0)          发起第 0 层
    wait_for_layer_load  → 等 L 层句柄；发 L+1（逐层预取）；
                           末层后 engine.complete_load(plan)
    save_kv_layer        → engine.store_layer(plan, L)          与计算并行
    wait_for_save        → no-op（逐层句柄已在 save_kv_layer 管理）
    get_finished         → 收割：store 全部层句柄 done →
                           engine.complete_store(plan, success)

并发模型（§2b，写明不改行为）：请求粒度 = chunk 层段；批量并发由
engine/backend 的批量提交承担；逐层回调的动机是 store 数据依赖（L 层
算完才发，保 overlap）。

背压（§2a 第 2 级）：``max_in_flight_layers > 0`` 时，在途层句柄数达
上限 → 最老句柄 wait 后再发新层；0 = 不限（第 1 级 = engine staging
槽位背压，第 3 级 = backend 拒收重发，均不在本层）。

波次拆分：engine 限制单次 load_layer/store_layer 的 chunk 数 ≤
max_chunks_per_wave（"adapter must split waves"）——本层负责把大 plan
按 max_chunks_per_wave 切波，complete_* 用完整 plan（keys 全集）。

术语纪律（§0/§0a 铁律）：本模块只出现 chunk / chunk_id(opaque) / layer /
slot_mapping 语义；staging 注册与 IO 提交是 backend 内部事务，worker
永远看不到。

chunk 身份重建（D-007 链式哈希的确定性）：load 侧 LoadSpec 只携带
scheduler 分配的 chunk_ids（opaque）；complete_load 需要完整 keys 才能
解除 pin——keys 由同一 token 序列经 engine.index.keys_for_tokens 滚动
重算（纯函数，两侧必然一致），不依赖 scheduler 侧额外传参。
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from engine.core import LoadPlan, StorePlan

if TYPE_CHECKING:  # pragma: no cover - 类型标注专用
    from adapter.connector import ReqMeta, TuttiConnectorMetadata
    from engine.core import TuttiEngine


class _LoadState:
    """单请求的加载状态（逐层预取推进）。"""

    __slots__ = ("plan", "issued", "handles")

    def __init__(self, plan: LoadPlan):
        self.plan = plan
        self.issued = 0      # 已发起的最高层号
        self.handles: dict = {}  # layer_idx → [句柄]（波次拆分可多条）


class _StoreState:
    """单请求的存储状态（逐层提交；收割在 get_finished）。"""

    __slots__ = ("plan", "issued", "handles")

    def __init__(self, plan: StorePlan):
        self.plan = plan
        self.issued = -1     # 已提交的最高层号（-1 = 尚未开始）
        self.handles: dict = {}


class WorkerImpl:
    """worker 侧编排薄层：vllm 逐层回调 → engine 执行态调用。

    cfg（connector 注入的 extra_config 副本，新增键 ``block_size``）：
    - ``block_size``（必填，int）：vllm paged KV 的块大小（token 数），
      与 engine.chunk_tokens 一起决定 blocks_per_chunk。
    engine：由 connector 构造注入（与 scheduler 侧同一 config 来源）。
    gather_fn / scatter_fn：透传 engine（非 None 时覆盖 engine 钩子；
    默认 None = engine 自带钩子——config 注入或 noop 记录。真 kernel
    接线由集成卡完成，不属本卡）。
    """

    def __init__(
        self,
        cfg: dict,
        engine: "TuttiEngine",
        max_in_flight_layers: int = 0,
        gather_fn=None,
        scatter_fn=None,
    ):
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be dict, got {type(cfg).__name__}")
        if not isinstance(max_in_flight_layers, int) or max_in_flight_layers < 0:
            raise ValueError(
                f"max_in_flight_layers must be int >= 0, got {max_in_flight_layers!r}"
            )
        self._cfg = cfg
        self._engine = engine
        self._max_in_flight = max_in_flight_layers
        if gather_fn is not None:
            engine.gather_fn = gather_fn
        if scatter_fn is not None:
            engine.scatter_fn = scatter_fn

        # register_kv_caches 后有效
        self._layer_index: "dict[str, int] | None" = None
        self._num_layers: "int | None" = None
        self._blocks_per_chunk: "int | None" = None

        # 每步状态
        self._loads: dict = {}   # req_id → _LoadState
        self._stores: dict = {}  # req_id → _StoreState
        # 背压第 2 级：在途层句柄（max_in_flight_layers > 0 时启用）
        self._inflight: deque = deque()

    # -------------------------------------------------- vllm 回调（§4）

    def register_kv_caches(self, kv_caches: dict) -> None:
        """层表翻译 + engine.bind（零注册：staging 是唯一注册对象，且
        注册动作在 backend 内部——KV 池从不被直接访问）。"""
        if self._layer_index is not None:
            raise RuntimeError("register_kv_caches already called")
        if not isinstance(kv_caches, dict) or not kv_caches:
            raise TypeError(f"kv_caches must be a non-empty dict, got {kv_caches!r}")

        block_size = self._cfg.get("block_size")
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError(
                "cfg['block_size'] must be a positive int "
                f"(vllm paged KV block size), got {block_size!r}"
            )
        chunk_tokens = self._engine.chunk_tokens
        if chunk_tokens % block_size != 0:
            raise ValueError(
                f"chunk_tokens ({chunk_tokens}) must be a multiple of "
                f"block_size ({block_size}) to map chunks to paged blocks"
            )

        self._layer_index = {name: i for i, name in enumerate(kv_caches)}
        self._num_layers = len(kv_caches)
        self._blocks_per_chunk = chunk_tokens // block_size
        self._engine.bind(kv_caches, self._num_layers, self._blocks_per_chunk)

    def start_load_kv(self, metadata: "TuttiConnectorMetadata", forward_context, **kwargs) -> None:
        """对本步 metadata 的每个 load 请求：发起第 0 层。"""
        self._require_registered()
        # 上一步残留的 load（异常中断）：先 complete_load 释放 pin
        for state in self._loads.values():
            self._engine.complete_load(state.plan)
        self._loads.clear()
        for req_meta in metadata.requests:
            spec = req_meta.load_spec
            if spec is None or not spec.chunk_ids:
                continue
            n = len(spec.chunk_ids)
            # D-007 链式哈希确定性：keys 由 token 序列重算（见模块 docstring）
            keys = tuple(
                self._engine.index.keys_for_tokens(req_meta.token_ids.tolist())[:n]
            )
            dst = self._chunk_first_blocks(req_meta.block_ids, start_chunk=0, n=n)
            plan = LoadPlan(keys=keys, chunk_ids=tuple(spec.chunk_ids), dst_first_blocks=dst)
            state = _LoadState(plan)
            self._loads[req_meta.req_id] = state
            self._issue_load_waves(state, layer_idx=0)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """等 L 层句柄 → 发 L+1（逐层预取）；末层后 complete_load。"""
        self._require_registered()
        layer_idx = self._layer_index_of(layer_name)
        for req_id in list(self._loads):
            state = self._loads[req_id]
            for handle in state.handles.get(layer_idx, ()):
                self._wait_and_retire(handle)
            if layer_idx != state.issued:
                continue  # 该请求此层未发起（异常时序）：不推进
            nxt = layer_idx + 1
            if nxt < self._num_layers:
                self._issue_load_waves(state, layer_idx=nxt)
                state.issued = nxt
            else:
                # 末层：解除 pin（计划态回收）
                self._engine.complete_load(state.plan)
                del self._loads[req_id]

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata,
                      metadata: "TuttiConnectorMetadata", **kwargs) -> None:
        """对本层待存请求：engine.store_layer（与计算并行）。"""
        self._require_registered()
        layer_idx = self._layer_index_of(layer_name)
        for req_meta in metadata.requests:
            spec = req_meta.save_spec
            if spec is None or not spec.can_save or not spec.chunk_ids:
                continue
            # One request can save several times (chunked prefill, decode crossing a
            # chunk boundary) while earlier saves are still in flight -> key per save event.
            store_key = (req_meta.req_id, spec.skip_leading_tokens)
            state = self._stores.get(store_key)
            if state is None:
                skip_chunk = spec.skip_leading_tokens // self._engine.chunk_tokens
                n = len(spec.chunk_ids)
                src = self._chunk_first_blocks(
                    req_meta.block_ids, start_chunk=skip_chunk, n=n
                )
                state = _StoreState(
                    StorePlan(
                        keys=tuple(spec.chunk_keys),
                        chunk_ids=tuple(spec.chunk_ids),
                        src_first_blocks=src,
                        evicted=(),
                    )
                )
                self._stores[store_key] = state
            if layer_idx <= state.issued:
                continue  # 本层已提交（重复回调）
            self._issue_store_waves(state, layer_idx=layer_idx)
            state.issued = layer_idx

    def wait_for_save(self) -> None:
        """no-op：逐层句柄已在 save_kv_layer 管理，收割在 get_finished。"""

    def get_finished(self, finished_req_ids) -> "tuple[set, set]":
        """收割已完成请求（store 全部层句柄 done → complete_store）。

        返回 vllm 要求的 (sending, receiving)；R1 恒 (None, None)
        （本 connector 不做跨进程异步收发声明）。
        """
        self._require_registered()
        for req_id in list(self._stores):
            state = self._stores[req_id]
            if state.issued != self._num_layers - 1:
                continue  # 还有层未提交（正常每步全层都会到）
            if not all(
                handle.query()
                for handles in state.handles.values()
                for handle in handles
            ):
                continue  # 在途：留待后续步收割
            self._engine.complete_store(state.plan, success=True)
            for handles in state.handles.values():
                for handle in handles:
                    self._retire(handle)
            del self._stores[req_id]
        return None, None

    # -------------------------------------------------- 内部

    def _require_registered(self) -> None:
        if self._layer_index is None:
            raise RuntimeError("register_kv_caches not called yet")

    def _layer_index_of(self, layer_name: str) -> int:
        try:
            return self._layer_index[layer_name]
        except KeyError:
            raise ValueError(f"unknown layer name {layer_name!r}") from None

    def _chunk_first_blocks(self, block_ids, start_chunk: int, n: int) -> tuple:
        """第 start_chunk+j 个 chunk 的首块号（chunk j 覆盖 token
        [g*cs,(g+1)*cs) → block_ids[g*cs//bs] = block_ids[g*bpc]）。

        离散块与连续块同路径：块号只是透传给 gather/scatter 钩子的
        paged 定位，本层不做任何变换。
        """
        bpc = self._blocks_per_chunk
        ids = block_ids.tolist() if hasattr(block_ids, "tolist") else list(block_ids)
        need = (start_chunk + n) * bpc
        if len(ids) < need:
            raise ValueError(
                f"block_ids has {len(ids)} blocks < {(start_chunk + n) * bpc} "
                f"needed for {n} chunks from chunk #{start_chunk}"
            )
        return tuple(ids[(start_chunk + j) * bpc] for j in range(n))

    def _issue_load_waves(self, state: _LoadState, layer_idx: int) -> None:
        plan, fb = state.plan, state.plan.dst_first_blocks
        handles = []
        for keys, chunk_ids, first_blocks in self._split_waves(
            plan.keys, plan.chunk_ids, fb
        ):
            sub = LoadPlan(keys=keys, chunk_ids=chunk_ids, dst_first_blocks=first_blocks)
            self._acquire_inflight_slot()  # 背压：先等最老，再发新层
            handle = self._engine.load_layer(sub, layer_idx)
            handles.append(handle)
            self._inflight.append(handle)
        state.handles[layer_idx] = handles

    def _issue_store_waves(self, state: _StoreState, layer_idx: int) -> None:
        plan, fb = state.plan, state.plan.src_first_blocks
        handles = []
        for keys, chunk_ids, first_blocks in self._split_waves(
            plan.keys, plan.chunk_ids, fb
        ):
            sub = StorePlan(
                keys=keys, chunk_ids=chunk_ids,
                src_first_blocks=first_blocks, evicted=(),
            )
            self._acquire_inflight_slot()  # 背压：先等最老，再发新层
            handle = self._engine.store_layer(sub, layer_idx)
            handles.append(handle)
            self._inflight.append(handle)
        state.handles[layer_idx] = handles

    def _split_waves(self, keys, chunk_ids, first_blocks):
        """按 engine.max_chunks_per_wave 切波（engine 拒绝超限单批）。"""
        m = self._engine.max_chunks_per_wave
        for i in range(0, len(chunk_ids), m):
            yield (
                tuple(keys[i : i + m]),
                tuple(chunk_ids[i : i + m]),
                tuple(first_blocks[i : i + m]),
            )

    # ---------- 背压第 2 级（§2a） ----------

    def _acquire_inflight_slot(self) -> None:
        """发起新层前取在途名额：在途层句柄数达上限 → 最老句柄
        wait 后再发（行为契约：wait 严格先于新层发起）。"""
        if self._max_in_flight <= 0:
            return
        while len(self._inflight) >= self._max_in_flight:
            self._inflight.popleft().wait()

    def _wait_and_retire(self, handle) -> None:
        handle.wait()
        self._retire(handle)

    def _retire(self, handle) -> None:
        try:
            self._inflight.remove(handle)
        except ValueError:
            pass  # 未启用背压或已被背压 wait 收割
