"""TuttiConnectorV1 — vLLM v1 connector 的 scheduler 侧 adapter（T-113，D-005）。

结构平移自 tutti-legacy ``engine/pkg/integration/vllm/vllm_v1_adapter.py``
（L166-191 LoadSpec/SaveSpec、L259-347 RequestTracker、L353-484 ReqMeta、
L1154-1226 get_num_new_matched_tokens、L1229-1275 update_state_after_alloc、
L1276-1386 build_connector_meta），其中 legacy 的 lookup_client（RPC）与
GPU 文件元数据替换为 TuttiEngine 计划态调用（ARCHITECTURE.md §3.1）：

- ``get_num_new_matched_tokens``：``engine.lookup_prefix`` 命中前缀；
  ``min_retrieve_tokens`` / ``max_tokens_per_load``（chunk 对齐 cap）与
  全命中 -1 重算的生产细节对齐 LMCache
  ``lmcache/integration/vllm/vllm_v1_adapter.py:1388-1531``。
- ``build_connector_meta``：``engine.plan_load`` / ``engine.plan_store``。
  chunk 身份 = chunk_key（链式哈希，D-007）；chunk 路径（chunk_id）由
  engine 的 backend 生成并解释——adapter 只 opaque 透传（§0a 铁律）。

铁律（§0/§0a）：本模块只出现 chunk / chunk_key / chunk_id / layer /
KVBlock 语义，不出现任何介质概念；不 import tutti_runtime。
worker 侧传输（T-114）：WORKER 角色构造 WorkerImpl（adapter/worker.py，
逐层编排翻译 + 背压），本类的 worker 回调转发给它；scheduler 角色行为
不变（T-113）。

新版 vllm fork（third_pkgs/vllm，RFC#45702）签名适配：
- ``get_num_new_matched_tokens`` 返回 ``tuple[int | None, bool]``
  （同步 lookup，恒非 None；本 connector 非 between-steps 异步加载，
  第二元素恒 False）。
- ``update_state_after_alloc(request, blocks, num_external_tokens)`` 三参
  （blocks 暂不使用；CoW pending copy 语义留给 T-114 接线）。
- ``scheduled_new_reqs`` 元素为 ``NewRequestData``；
  ``scheduled_cached_reqs`` 为 ``CachedRequestData``（并行数组，
  ``all_token_ids`` 传播全 token 序列，``resumed_req_ids`` 标记
  preemption 恢复——恢复时 ``new_block_ids`` 替换而非追加）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)

from adapter.worker import WorkerImpl
from engine.core import TuttiEngine

if TYPE_CHECKING:  # pragma: no cover - 类型标注专用
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

#: extra_config 可选键的默认值（对齐 LMCache 生产语义）
DEFAULT_CHUNK_SIZE = 256  # 对齐 LMCache 默认（lmcache/v1/config.py）

# ---------------------------------------------------------------------------
# 共享 TuttiEngine 注册表（R2-A 上抛裁决，2026-08-20）：
# scheduler 与 worker 两个角色各构造一个 TuttiConnectorV1；chunk 生命周期
# 索引（pin/pending/resident）必须是同一实例，否则 plan_* 记在 scheduler
# 侧、complete_* 结算在 worker 侧 → ChunkIndex KeyError + 泄漏。
# 单进程内按 (id(vllm_config), engine_config 规范化键) 共享同一实例；
# 测试可经 extra_config["tutti_engine_instance"] 直传实例完全绕过注册表。
# 多进程（调度/worker 分进程）下的索引状态同步属集成轮问题，另行设计。
# ---------------------------------------------------------------------------
_ENGINE_REGISTRY: dict = {}


def _engine_registry_key(vllm_config, engine_config: dict):
    """规范化注册键：同 config 对象 + 同内容 → 同一 engine 实例。"""
    items = tuple(
        (k, f"backend@{id(v)}" if k == "backend" else repr(v))
        for k, v in sorted(engine_config.items())
    )
    return (id(vllm_config), items)


# ---------------------------------------------------------------- Load/SaveSpec


@dataclass
class LoadSpec:
    """一次外部命中的计划态记录（legacy :168-176 平移，去介质字段）。

    - vllm_cached_tokens：调度查询时 vLLM 本地已算 token 数
    - tutti_cached_tokens：engine 命中前缀 token 数（cap 后，chunk 对齐）
    - can_load：update_state_after_alloc 确认可加载后置 True
    - chunk_ids：plan_load 分配的存储路径（opaque，backend 解释）
    """

    req_id: str
    vllm_cached_tokens: int
    tutti_cached_tokens: int
    can_load: bool = False
    chunk_ids: tuple = ()


@dataclass
class SaveSpec:
    """一次存储计划的请求侧描述（legacy :181-191 平移）。

    - skip_leading_tokens：已保存前缀 token 数（含外部命中前缀）
    - can_save：本步是否有新完整 chunk 需要写盘
    - chunk_keys / chunk_ids：plan_store 结果（新分配需要写盘的 chunk）
    """

    req_id: str
    skip_leading_tokens: int
    can_save: bool
    chunk_keys: tuple = ()
    chunk_ids: tuple = ()


# ---------------------------------------------------------------- RequestTracker


@dataclass
class RequestTracker:
    """请求生命周期追踪（legacy :259-347 平移，结构参考 LMCache :111）。

    token_ids 为"已调度前缀"序列（preempted 恢复时由 all_token_ids
    重建），allocated_block_ids 为 vLLM 已分配块号（多 KV cache group
    时取 group 0——vllm connector 单 group 约定）。
    """

    req_id: str
    token_ids: list[int]
    allocated_block_ids: list[int]
    # The number of tokens that has been saved (incl. external hits)
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request,
        num_tokens_to_compute: int,
        external_cached_tokens: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request: vLLM ``NewRequestData``（scheduled_new_reqs 元素）。
            num_tokens_to_compute: 将被"计算"的 token 数（含 vLLM 本地
                命中 num_computed_tokens 与本步新调度 token）。
            external_cached_tokens: engine 命中前缀 token 数（外部命中
                视为已保存，避免重复写盘）。
        """
        # vLLM 0.9.0 update: block_ids changed from list[int] to
        # tuple[list[int], ...]; connector 只支持单 KV cache group。
        block_ids = new_request.block_ids
        if block_ids and isinstance(block_ids[0], (list, tuple)):
            unfolded_block_ids = list(block_ids[0])
        else:
            unfolded_block_ids = list(block_ids)

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=list(
                new_request.prompt_token_ids[:num_tokens_to_compute]
            ),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=external_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: "Sequence[Sequence[int]] | None",
    ) -> None:
        """Update the request tracker when a running request is scheduled
        again（legacy :327-347 平移）。"""
        self.token_ids.extend(new_token_ids)
        if new_block_ids is None or len(new_block_ids) == 0:
            return
        if not isinstance(new_block_ids[0], (list, tuple)):
            # 平铺列表（旧版 vllm 兼容）
            self.allocated_block_ids.extend(new_block_ids)
            return
        # 单 group：取 group 0
        self.allocated_block_ids.extend(new_block_ids[0])


# ---------------------------------------------------------------- ReqMeta


@dataclass
class ReqMeta:
    """worker 侧单请求操作描述（legacy :353-484 平移）。"""

    # Request id
    req_id: str
    # Request tokens（本步 save 范围内的 token 序列）
    token_ids: torch.Tensor
    # Slot mapping of current request, 由 allocated_block_ids 计算出
    slot_mapping: torch.Tensor

    block_ids: torch.Tensor

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        external_chunk_size: int = DEFAULT_CHUNK_SIZE,
        load_spec: Optional[LoadSpec] = None,
        skip_save: bool = False,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker: the request tracker.
            block_size: the block size in vLLM.
            external_chunk_size: the chunk size（token 数）。
            load_spec: the load spec for KV cache loading.
            skip_save: whether to skip the save operation.
            discard_partial_chunks: 尾部不满一个 chunk 是否舍弃。

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """

        def cdiv(a: int, b: int) -> int:
            return -(a // -b)

        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens

        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, external_chunk_size)
            * external_chunk_size
        )
        skip_save = skip_save or (input_token_len < chunk_boundary)

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting（partial chunk 舍弃）
        num_tokens_to_save = (
            (input_token_len // external_chunk_size * external_chunk_size)
            if discard_partial_chunks
            else input_token_len
        )

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(
            tracker.req_id, skip_leading_tokens, not skip_save
        )

        # Calculate the token ids and slot mappings for load and save
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]

        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.tutti_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            block_ids=block_ids,
            save_spec=save_spec,
            load_spec=load_spec,
        )


# ---------------------------------------------------------------- metadata


class TuttiConnectorMetadata:
    """scheduler → worker 的每步元数据（T-114 worker 侧消费）。"""

    def __init__(self) -> None:
        self.requests: list[ReqMeta] = []

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)


# ---------------------------------------------------------------- connector


class TuttiConnectorV1(KVConnectorBase_V1):
    """Tutti 外部 KV connector 的双角色适配层（D-005 方案一）。

    scheduler 角色（T-113）：命中查询/分配确认/metadata 组装，调
    engine 计划态；worker 角色（T-114）：逐层回调转发 WorkerImpl，调
    engine 执行态（ARCHITECTURE.md §4 映射表）。

    config（vllm_config.kv_transfer_config.kv_connector_extra_config）：
    - ``tutti_engine_config``（必填，dict）：透传 TuttiEngine(config=...)；
      其 ``chunk_tokens`` 缺省时取本层 ``chunk_size``，显式给定且不一致
      → ValueError。
    - ``chunk_size``（可选，默认 256）：chunk 的 token 数。
    - ``min_retrieve_tokens``（可选，默认 0）：低于该值不检索（仍记录
      命中前缀供 save skip，对齐 LMCache）。
    - ``max_tokens_per_load``（可选，默认 0 = 不限）：单次加载 token 上
      限，chunk 对齐截断。
    - ``force_skip_save``（可选，默认 False）：全局禁写。
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        # kv_cache_config 缺省为 None：真 vllm KVConnectorFactory 以
        # 3 参 (config, role, kv_cache_config) 实例化（base.py）；
        # 2 参直调（测试）同样合法。
        super().__init__(vllm_config, role, kv_cache_config)

        transfer_config = getattr(
            vllm_config, "kv_transfer_config", None
        )
        extra_config = getattr(
            transfer_config, "kv_connector_extra_config", None
        )
        if not isinstance(extra_config, dict):
            raise ValueError(
                "TuttiConnectorV1 requires a dict "
                "kv_transfer_config.kv_connector_extra_config, got "
                f"{extra_config!r}"
            )
        if "tutti_engine_config" not in extra_config:
            raise ValueError(
                "TuttiConnectorV1 extra_config is missing required key "
                "'tutti_engine_config' (dict for TuttiEngine)"
            )

        engine_config = dict(extra_config["tutti_engine_config"])
        chunk_size = int(extra_config.get("chunk_size", DEFAULT_CHUNK_SIZE))
        if "chunk_tokens" in engine_config:
            if int(engine_config["chunk_tokens"]) != chunk_size:
                raise ValueError(
                    "chunk_tokens mismatch: engine "
                    f"{engine_config['chunk_tokens']!r} vs adapter "
                    f"chunk_size {chunk_size!r}"
                )
        else:
            engine_config["chunk_tokens"] = chunk_size

        self._chunk_size = chunk_size
        self._block_size = vllm_config.cache_config.block_size

        explicit_engine = extra_config.get("tutti_engine_instance")
        if explicit_engine is not None:
            self.engine = explicit_engine
        else:
            key = _engine_registry_key(vllm_config, engine_config)
            shared = _ENGINE_REGISTRY.get(key)
            if shared is None:
                shared = TuttiEngine(config=engine_config)
                _ENGINE_REGISTRY[key] = shared
            self.engine = shared

        # worker 角色：双角色壳的 worker 侧（T-114）——与 scheduler 同一
        # config 来源构造 engine，回调转发 WorkerImpl（§4 映射表）。
        if role == KVConnectorRole.WORKER:
            self._worker = WorkerImpl(
                cfg=dict(extra_config, block_size=self._block_size),
                engine=self.engine,
                max_in_flight_layers=int(
                    extra_config.get("max_in_flight_layers", 0)
                ),
                gather_fn=extra_config.get("gather_fn"),
                scatter_fn=extra_config.get("scatter_fn"),
            )
            return

        # ---- scheduler 角色（T-113）----
        self._worker = None
        self._live_requests: dict = {}
        self._min_retrieve_tokens = int(
            extra_config.get("min_retrieve_tokens", 0)
        )
        self._max_tokens_per_load = int(
            extra_config.get("max_tokens_per_load", 0)
        )
        self._force_skip_save = bool(extra_config.get("force_skip_save", False))

        # 请求级状态
        self.load_specs: dict[str, LoadSpec] = {}
        self._request_trackers: dict[str, RequestTracker] = {}

    # -------------------------------------------------- scheduler: 命中查询

    def on_new_request(self, request) -> None:
        """Keep the live Request so build_connector_meta can read new token ids
        (vLLM >= 0.29 no longer ships them in CachedRequestData)."""
        self._live_requests[request.request_id] = request

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> "tuple[Optional[int], bool]":
        """Check for external KV cache hit（新签名 tuple 返回）。

        token 序列取 ``request.all_token_ids``（覆盖 preemption 恢复）。
        """
        token_ids = list(request.all_token_ids)

        # 外部命中前缀（链式 chunk key 滚动，engine 计划态）
        num_external_hit_tokens = self.engine.lookup_prefix(token_ids)

        # When prompt length is divisible by the chunk size and all
        # chunks are cached, we need to recompute the last token.
        # (legacy :1196-1204 / LMCache :1444-1452)
        need_to_allocate = num_external_hit_tokens - num_computed_tokens
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        # (LMCache :1454-1458)
        below_min_retrieve = (
            self._min_retrieve_tokens > 0
            and need_to_allocate < self._min_retrieve_tokens
        )

        # Chunked KV loading: cap the number of tokens reported to the
        # scheduler; cap aligned to chunk boundaries (LMCache :1483-1507)
        capped_hit_tokens = num_external_hit_tokens
        if (
            self._max_tokens_per_load > 0
            and need_to_allocate > self._max_tokens_per_load
        ):
            cap = (
                self._max_tokens_per_load
                // self._chunk_size
                * self._chunk_size
            )
            need_to_allocate = cap
            capped_hit_tokens = (
                (num_computed_tokens + cap)
                // self._chunk_size
                * self._chunk_size
            )

        self.load_specs[request.request_id] = LoadSpec(
            req_id=request.request_id,
            vllm_cached_tokens=num_computed_tokens,
            tutti_cached_tokens=capped_hit_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0, False
        return need_to_allocate, False

    # -------------------------------------------------- scheduler: 分配确认

    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ) -> None:
        """Update KVConnector state after block allocation.

        blocks（新版 vllm KVCacheBlocks）暂不使用——CoW pending copy
        语义由 T-114 worker 侧接线时处理。
        """
        spec = self.load_specs.get(request.request_id)
        if spec is None:
            return

        if num_external_tokens == 0:
            # No need to load anything
            spec.can_load = False
            return

        # full-hit 请求需重算最后一个 token（get_num_new_matched_tokens
        # 已从 need 中扣除；对齐 LMCache :1585-1607 的一致性断言）
        recalc_last = (
            1 if spec.tutti_cached_tokens == request.num_tokens else 0
        )
        expected = (
            spec.tutti_cached_tokens
            - spec.vllm_cached_tokens
            - recalc_last
        )
        assert num_external_tokens == expected, (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{spec.tutti_cached_tokens} (tokens in tutti) - "
            f"{spec.vllm_cached_tokens} (tokens in vllm) - "
            f"{recalc_last} (full tutti hits subtracts last token to "
            f"recalculate logits) for request {request.request_id}"
        )
        spec.can_load = True

    # -------------------------------------------------- scheduler: 组装 meta

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> TuttiConnectorMetadata:
        """Build the connector metadata for this step.

        遍历 finished/new/cached 三类请求（legacy :1276-1386 平移），
        对 load 请求 ``engine.plan_load``（pin），对 save 请求
        ``engine.plan_store``（分配路径）；chunk_id opaque 透传给 worker。
        调用后本 connector 的步级状态被重置（load_specs 已消费）。
        """
        meta = TuttiConnectorMetadata()

        # 处理计算完成的请求
        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self.load_specs.pop(finished_req_id, None)
            self._live_requests.pop(finished_req_id, None)

        # 处理新调度的请求：tracker 建档 + load/save 计划
        for request in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            external_cached_tokens = 0
            if load_spec is not None:
                external_cached_tokens = load_spec.tutti_cached_tokens

            request_tracker = RequestTracker.from_new_request(
                request,
                num_tokens_to_compute,
                external_cached_tokens,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._chunk_size,
                load_spec=load_spec,
                skip_save=self._force_skip_save,
                discard_partial_chunks=True,
            )
            if req_meta is not None:
                self._plan_request(req_meta, request_tracker)
                meta.add_request(req_meta)

        # 处理已调度过的请求：tracker 增量更新 + save 计划
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers.get(req_id)
            if request_tracker is None:
                # preemption 恢复的首步可能未经 scheduled_new_reqs；
                # 此时 all_token_ids 给出全序列，重建建档
                all_token_ids = cached_reqs.all_token_ids.get(req_id)
                assert all_token_ids is not None, (
                    f"Request {req_id} not in _request_trackers and no "
                    "all_token_ids propagated for recovery"
                )
                request_tracker = RequestTracker(
                    req_id=req_id,
                    token_ids=[],
                    allocated_block_ids=[],
                    num_saved_tokens=0,
                )
                self._request_trackers[req_id] = request_tracker

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(
                req_id, 0
            )
            new_block_ids = cached_reqs.new_block_ids[i]
            all_token_ids = cached_reqs.all_token_ids.get(req_id)

            if req_id in cached_reqs.resumed_req_ids:
                # preemption 恢复：token 序列重建，块号整体替换
                if all_token_ids is not None:
                    request_tracker.token_ids = list(all_token_ids)
                if new_block_ids:
                    request_tracker.allocated_block_ids = list(
                        new_block_ids[0]
                    )
            else:
                if all_token_ids is not None:
                    base = len(request_tracker.token_ids)
                    new_token_ids = all_token_ids[
                        base : base + num_new_tokens
                    ]
                else:
                    # vLLM >= 0.29: new_token_ids is PP-only (empty otherwise) and
                    # all_token_ids only covers requests skipped last step -> read the
                    # live Request recorded by on_new_request() (same as LMCache does).
                    live = self._live_requests.get(req_id)
                    if live is not None:
                        base = len(request_tracker.token_ids)
                        new_token_ids = list(live.all_token_ids[base : base + num_new_tokens])
                    elif i < len(cached_reqs.new_token_ids):
                        new_token_ids = cached_reqs.new_token_ids[i]
                    else:
                        new_token_ids = []
                request_tracker.update(new_token_ids, new_block_ids)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._chunk_size,
                load_spec=None,
                skip_save=self._force_skip_save,
                discard_partial_chunks=True,
            )
            if req_meta is not None:
                self._plan_request(req_meta, request_tracker)
                meta.add_request(req_meta)

        return meta

    # -------------------------------------------------- worker 侧接线（T-114）

    def register_kv_caches(self, kv_caches: dict) -> None:
        """层表 → WorkerImpl（engine.bind；scheduler 角色无操作）。"""
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context, **kwargs) -> None:
        self._require_worker().start_load_kv(
            self._worker_metadata(), forward_context, **kwargs
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._require_worker().wait_for_layer_load(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kwargs
    ) -> None:
        self._require_worker().save_kv_layer(
            layer_name, kv_layer, attn_metadata,
            self._worker_metadata(), **kwargs,
        )

    def wait_for_save(self) -> None:
        self._require_worker().wait_for_save()

    def get_finished(self, finished_req_ids) -> "tuple[set, set]":
        """收割（worker 角色）；scheduler 角色走基类默认 (None, None)。"""
        if self._worker is not None:
            return self._worker.get_finished(finished_req_ids)
        return None, None

    def _require_worker(self) -> WorkerImpl:
        if self._worker is None:
            raise NotImplementedError(
                "worker-side callbacks are only available on a WORKER-role "
                "connector (scheduler-only 进程不实现 worker 侧传输)"
            )
        return self._worker

    def _worker_metadata(self) -> "TuttiConnectorMetadata":
        """本步 metadata（model runner 经 bind_connector_metadata 注入）。

        未绑定时返回空 metadata（防御直调；真实生命周期恒先 bind）。
        """
        meta = getattr(self, "_connector_metadata", None)
        return meta if meta is not None else TuttiConnectorMetadata()

    # -------------------------------------------------- 内部：engine 计划

    def _plan_request(
        self, req_meta: ReqMeta, tracker: RequestTracker
    ) -> None:
        """对单个 ReqMeta 执行 engine 计划态调用（load pin / save 分配）。

        chunk_key 用链式哈希从 tracker.token_ids 全序列滚动计算
        （engine.index.keys_for_tokens），再按区间切片——保证与
        lookup_prefix / 索引存储完全一致（D-007 链式体系）。
        """
        cs = self._chunk_size

        # load 计划：can_load 时 pin 命中前缀的 chunk 路径
        if req_meta.load_spec is not None:
            hit_chunks = req_meta.load_spec.tutti_cached_tokens // cs
            all_keys = self.engine.index.keys_for_tokens(tracker.token_ids)
            load_keys = all_keys[:hit_chunks]
            try:
                plan = self.engine.plan_load(load_keys)
            except KeyError:
                # lookup 与 plan 之间被并发驱逐：放弃本步加载
                logger.warning(
                    "Reqid %s: planned load keys evicted before plan_load; "
                    "falling back to local compute",
                    req_meta.req_id,
                )
                req_meta.load_spec = None
            else:
                req_meta.load_spec.chunk_ids = tuple(plan.chunk_ids)

        # save 计划：为"已保存前缀之后、新完整 chunk 边界之内"的
        # chunk 分配路径（skip_leading 对齐 legacy SaveSpec）
        if req_meta.save_spec is not None and req_meta.save_spec.can_save:
            skip_chunk = req_meta.save_spec.skip_leading_tokens // cs
            end_chunk = tracker.num_saved_tokens // cs
            all_keys = self.engine.index.keys_for_tokens(tracker.token_ids)
            save_keys = all_keys[skip_chunk:end_chunk]
            if save_keys:
                plan = self.engine.plan_store(save_keys)
                if plan is None:
                    # 容量耗尽且无可驱逐：放弃本步写盘
                    logger.warning(
                        "Reqid %s: engine out of capacity for %d chunks; "
                        "skipping save this step",
                        req_meta.req_id,
                        len(save_keys),
                    )
                    req_meta.save_spec.can_save = False
                else:
                    req_meta.save_spec.chunk_keys = tuple(plan.keys)
                    req_meta.save_spec.chunk_ids = tuple(plan.chunk_ids)
