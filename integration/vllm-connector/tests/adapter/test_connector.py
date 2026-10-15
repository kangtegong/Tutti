# SPDX-License-Identifier: Apache-2.0
"""T-113 tests: TuttiConnectorV1 scheduler-side adapter (D-005 / D-007).

Run:
    source /data/home/ryeqiu/env-tutti.sh
    cd integration/vllm-connector && python -m pytest tests/adapter/ -v
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# integration/vllm-connector（engine 顶层包 + adapter 所在根）
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData  # noqa: E402

from adapter.connector import (  # noqa: E402
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
    TuttiConnectorV1,
)

CS = 128          # chunk_size（token 数）
BS = 128          # vllm block_size
CHUNK_KV_BYTES = 1024
CAPACITY = 1024 * 64   # 64 chunks


# ---------------------------------------------------------------- helpers


def make_extra(**overrides) -> dict:
    extra = {
        "tutti_engine_config": {
            "backend": "memory",
            "chunk_kv_bytes": CHUNK_KV_BYTES,
            "capacity_bytes": CAPACITY,
            "max_chunks_per_wave": 8,
        },
        "chunk_size": CS,
    }
    extra.update(overrides)
    return extra


def make_connector(extra: dict) -> TuttiConnectorV1:
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BS),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra
        ),
    )
    return TuttiConnectorV1(vllm_config, KVConnectorRole.SCHEDULER)


def fake_request(req_id: str, tokens) -> object:
    tokens = list(tokens)
    return SimpleNamespace(
        request_id=req_id, all_token_ids=tokens, num_tokens=len(tokens)
    )


def make_new_request(req_id, prompt_tokens, block_ids, num_computed=0):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt_tokens),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def make_sched_output(
    new_reqs=(),
    cached: "CachedRequestData | None" = None,
    num_scheduled_tokens: dict | None = None,
    finished=(),
):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens or {},
        total_num_scheduled_tokens=sum(
            (num_scheduled_tokens or {}).values()
        ),
        finished_req_ids=set(finished),
    )


def store_prefix(connector: TuttiConnectorV1, tokens, num_chunks: int):
    """用 engine 计划态直接预填前 num_chunks 个 chunk（模拟历史写盘）。"""
    keys = connector.engine.index.keys_for_tokens(list(tokens))
    plan = connector.engine.plan_store(keys[:num_chunks])
    assert plan is not None
    connector.engine.complete_store(plan, success=True)
    return keys


# 5-chunk / 640-token prompt（token 序列写死）
PROMPT = list(range(1000, 1000 + 5 * CS))


# ---------------------------------------------------------------- import / 角色


def test_import_smoke():
    """验收点 5：import 冒烟。"""
    from adapter.connector import TuttiConnectorV1 as C

    assert callable(C)
    assert TuttiConnectorV1 is C


def test_scheduler_role_instantiable_with_real_vllm_base():
    """验收点 4：SCHEDULER 可实例化，且 isinstance 真 vllm 基类。"""
    connector = make_connector(make_extra())
    assert isinstance(connector, KVConnectorBase_V1)
    assert isinstance(connector, TuttiConnectorV1)


def test_worker_role_instantiable_with_real_vllm_base():
    """WORKER 角色可实例化（T-114 接线后），isinstance 真 vllm 基类；
    worker 回调在 scheduler 角色实例上仍被拒绝（占位语义保留）。"""
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BS),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=make_extra()
        ),
    )
    connector = TuttiConnectorV1(vllm_config, KVConnectorRole.WORKER)
    assert isinstance(connector, KVConnectorBase_V1)
    assert isinstance(connector, TuttiConnectorV1)
    assert connector._worker is not None
    assert connector.engine is not None

    # scheduler 角色实例：worker 回调拒绝（占位语义保留）
    sched = make_connector(make_extra())
    assert sched._worker is None
    with pytest.raises(NotImplementedError):
        sched.start_load_kv(None)
    with pytest.raises(NotImplementedError):
        sched.wait_for_layer_load("l0")
    with pytest.raises(NotImplementedError):
        sched.save_kv_layer("l0", None, None)
    with pytest.raises(NotImplementedError):
        sched.wait_for_save()
    # scheduler 角色沿用基类默认收割语义
    assert sched.get_finished(set()) == (None, None)


# ---------------------------------------------------------------- config 校验


def test_missing_engine_config_raises():
    """验收点 5：缺 tutti_engine_config → ValueError（消息含键名）。"""
    with pytest.raises(ValueError, match="tutti_engine_config"):
        make_connector({"chunk_size": CS})


def test_chunk_tokens_mismatch_raises():
    extra = make_extra()
    extra["tutti_engine_config"]["chunk_tokens"] = 64
    with pytest.raises(ValueError, match="chunk_tokens"):
        make_connector(extra)


def test_engine_config_default_chunk_tokens_injected():
    """engine_config 未给 chunk_tokens 时注入 adapter 的 chunk_size。"""
    connector = make_connector(make_extra())
    assert connector.engine.chunk_tokens == CS


# ---------------------------------------------------------------- slot_mapping（legacy :390-469）


def test_tracker_of_resumed_request_covers_generated_tokens():
    """A request that resumes after a preemption reaches the connector as a new
    request (vLLM's V2 model runner); its whole sequence (prompt + generated
    tokens) is in prefill_token_ids. The hit lookup runs over that sequence, so
    the tracker - the load plan - must cover it, not the prompt alone."""
    prompt = list(range(5000, 5000 + 200))
    generated = list(range(9000, 9000 + 70))
    nr = make_new_request("r_resumed", prompt, [3, 7, 9])
    nr.prefill_token_ids = prompt + generated
    tracker = RequestTracker.from_new_request(nr, 270, 2 * CS)
    assert tracker.token_ids == prompt + generated
    # a request without prefill_token_ids (V1 runner, or a first admission) keeps the prompt
    nr2 = make_new_request("r_new", prompt, [3, 7])
    assert RequestTracker.from_new_request(nr2, 200, 0).token_ids == prompt


def test_slot_mapping_values_written_out():
    """验收点 1：block_ids=[3,7]、block_size=128、200 token。

    partial chunk（200-128=72 个尾部）舍弃 → num_tokens_to_save=128；
    slot_mapping 仅覆盖 block 3：[3*128 + 0 .. 3*128 + 127]（写死）。
    """
    tokens = list(range(5000, 5000 + 200))
    nr = make_new_request("r1", tokens, [3, 7])
    tracker = RequestTracker.from_new_request(nr, 200, 0)
    assert tracker.token_ids == tokens
    assert tracker.allocated_block_ids == [3, 7]

    req_meta = ReqMeta.from_request_tracker(tracker, BS, CS)
    assert req_meta is not None
    # partial chunk 舍弃：只保留 1 个完整 chunk 的 token
    assert len(req_meta.token_ids) == CS
    assert tracker.num_saved_tokens == CS
    # slot_mapping 逐项断言（legacy :456-462 公式）
    expected = torch.tensor(
        [3 * BS + i for i in range(BS)], dtype=torch.long
    )
    assert torch.equal(req_meta.slot_mapping, expected)
    assert req_meta.slot_mapping.dtype == torch.long
    # 第二个 chunk 边界之外（block 7）不出现
    assert int(req_meta.slot_mapping.min()) == 3 * BS
    assert int(req_meta.slot_mapping.max()) == 3 * BS + BS - 1


def test_slot_mapping_multi_block():
    """256 token、blocks=[3,7]：恰好 2 chunk，slot_mapping 跨两块写死。"""
    tokens = list(range(7000, 7000 + 2 * CS))
    nr = make_new_request("r2", tokens, [3, 7])
    tracker = RequestTracker.from_new_request(nr, 2 * CS, 0)
    req_meta = ReqMeta.from_request_tracker(tracker, BS, CS)
    expected = torch.tensor(
        [3 * BS + i for i in range(BS)] + [7 * BS + i for i in range(BS)],
        dtype=torch.long,
    )
    assert torch.equal(req_meta.slot_mapping, expected)


def test_skip_leading_tokens_chunk_alignment():
    """外部命中前缀跳过：num_saved_tokens=128 时新请求 384 token。

    skip_leading=128（已存 1 chunk）；chunk_boundary=cdiv(129,128)*128=256
    ≤ 384 → save；num_tokens_to_save=384；token_ids 为全部 384（load/save
    共用），slot_mapping 覆盖 3 块。
    """
    tokens = list(range(8000, 8000 + 3 * CS))
    nr = make_new_request("r3", tokens, [0, 1, 2])
    tracker = RequestTracker.from_new_request(nr, 3 * CS, CS)  # 外部命中 128
    req_meta = ReqMeta.from_request_tracker(tracker, BS, CS)
    assert req_meta is not None
    assert req_meta.save_spec.skip_leading_tokens == CS
    assert req_meta.save_spec.can_save
    assert len(req_meta.token_ids) == 3 * CS
    assert tracker.num_saved_tokens == 3 * CS


def test_below_chunk_boundary_skips_save():
    """num_saved=128、新序列 200 token（boundary=256>200）→ skip_save。"""
    tokens = list(range(9000, 9000 + 200))
    nr = make_new_request("r4", tokens, [0, 1])
    tracker = RequestTracker.from_new_request(nr, 200, CS)
    req_meta = ReqMeta.from_request_tracker(tracker, BS, CS)
    # skip_save 且无 load → None
    assert req_meta is None


# ---------------------------------------------------------------- 命中查询


def test_get_num_new_matched_tokens_partial_hit():
    """验收点 2：engine 预填 3 chunk，5-chunk prompt 查询。

    返回 (3*cs - computed, False) = (384, False)。
    """
    connector = make_connector(make_extra())
    store_prefix(connector, PROMPT, 3)
    req = fake_request("q1", PROMPT)
    assert connector.get_num_new_matched_tokens(req, 0) == (3 * CS, False)
    # spec 已记录（vllm_cached=0, tutti_cached=384, can_load=False）
    spec = connector.load_specs["q1"]
    assert spec.vllm_cached_tokens == 0
    assert spec.tutti_cached_tokens == 3 * CS
    assert not spec.can_load


def test_get_num_new_matched_tokens_full_hit_recalc_last():
    """全命中 -1 路径：hit == num_tokens → need -= 1（legacy :1196-1204）。"""
    connector = make_connector(make_extra())
    store_prefix(connector, PROMPT, 5)  # 640 == num_tokens
    req = fake_request("q2", PROMPT)
    assert connector.get_num_new_matched_tokens(req, 0) == (5 * CS - 1, False)


def test_get_num_new_matched_tokens_min_retrieve_gate():
    """min_retrieve_tokens 门限：命中低于阈值 → (0, False)，仍记录 spec。"""
    extra = make_extra(min_retrieve_tokens=500)
    connector = make_connector(extra)
    store_prefix(connector, PROMPT, 3)  # hit 384 < 500
    req = fake_request("q3", PROMPT)
    assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
    # below-min 仍记录命中（供 save skip，LMCache :1454-1458）
    assert connector.load_specs["q3"].tutti_cached_tokens == 3 * CS


def test_get_num_new_matched_tokens_max_tokens_per_load_cap():
    """max_tokens_per_load：chunk 对齐 cap（LMCache :1483-1507）。"""
    extra = make_extra(max_tokens_per_load=200)  # cap 对齐 → 128
    connector = make_connector(extra)
    store_prefix(connector, PROMPT, 3)  # hit 384
    req = fake_request("q4", PROMPT)
    assert connector.get_num_new_matched_tokens(req, 0) == (CS, False)
    spec = connector.load_specs["q4"]
    # capped_hit = (0 + 128) // 128 * 128
    assert spec.tutti_cached_tokens == CS


def test_get_num_new_matched_tokens_no_hit():
    connector = make_connector(make_extra())
    req = fake_request("q5", PROMPT)
    assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
    assert connector.load_specs["q5"].tutti_cached_tokens == 0


# ---------------------------------------------------------------- 分配确认


def test_update_state_after_alloc_can_load():
    """验收点 3：update_state_after_alloc 后 LoadSpec.can_load=True。"""
    connector = make_connector(make_extra())
    store_prefix(connector, PROMPT, 3)
    req = fake_request("q6", PROMPT)
    connector.get_num_new_matched_tokens(req, 0)
    spec = connector.load_specs["q6"]
    assert not spec.can_load

    connector.update_state_after_alloc(req, blocks=None, num_external_tokens=3 * CS)
    assert spec.can_load

    # num_external_tokens == 0 → 不加载
    connector.get_num_new_matched_tokens(req, 0)
    spec2 = connector.load_specs["q6"]
    connector.update_state_after_alloc(req, blocks=None, num_external_tokens=0)
    assert not spec2.can_load


def test_update_state_after_alloc_full_hit_consistency():
    """全命中：num_external = tutti - vllm - 1（重算最后一个 token）。"""
    connector = make_connector(make_extra())
    store_prefix(connector, PROMPT, 5)
    req = fake_request("q7", PROMPT)
    matched, _ = connector.get_num_new_matched_tokens(req, 0)
    assert matched == 5 * CS - 1
    # vllm 按返回值分配：num_external == 639
    connector.update_state_after_alloc(req, blocks=None, num_external_tokens=matched)
    assert connector.load_specs["q7"].can_load

    # 不一致 → AssertionError（LMCache :1593-1607）
    connector.get_num_new_matched_tokens(req, 0)
    with pytest.raises(AssertionError):
        connector.update_state_after_alloc(
            req, blocks=None, num_external_tokens=12345
        )


def test_update_state_after_alloc_unknown_request_noop():
    connector = make_connector(make_extra())
    connector.update_state_after_alloc(
        fake_request("ghost", PROMPT), blocks=None, num_external_tokens=8
    )  # 不应抛错


# ---------------------------------------------------------------- build_connector_meta


def test_build_connector_meta_load_and_save():
    """验收点 4：真 TuttiEngine（MemoryBackend）全链路。

    engine 预填 3 chunk → 查询命中 384 → 分配确认 → build_connector_meta：
    - plan_load：load_spec.chunk_ids 与 engine 索引内路径一致（opaque 透传）
    - plan_store：save 段（chunk 3..4）分配新路径，进 pending_store
    """
    connector = make_connector(make_extra())
    keys = store_prefix(connector, PROMPT, 3)

    req = fake_request("m1", PROMPT)
    connector.get_num_new_matched_tokens(req, 0)
    connector.update_state_after_alloc(req, blocks=None, num_external_tokens=3 * CS)

    # vllm 视角：384 已算（外部加载），本步调度剩余 256 → compute 640
    nr = make_new_request("m1", PROMPT, [0, 1, 2, 3, 4], num_computed=3 * CS)
    out = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"m1": 2 * CS}
    )
    meta = connector.build_connector_meta(out)

    assert len(meta.requests) == 1
    rm = meta.requests[0]
    assert rm.req_id == "m1"

    # ---- load 计划：chunk_ids 与 engine 索引内路径一致（opaque 透传）
    assert rm.load_spec is not None
    expected_paths = tuple(
        connector.engine.index.stored[k] for k in keys[:3]
    )
    assert rm.load_spec.chunk_ids == expected_paths
    # plan_load 已 pin（每路径 refcount 1）
    for p in expected_paths:
        assert connector.engine.index.pinned[p] == 1

    # ---- save 计划：chunk 3..4 两个新 chunk
    assert rm.save_spec.can_save
    assert rm.save_spec.skip_leading_tokens == 3 * CS
    assert rm.save_spec.chunk_keys == tuple(keys[3:5])
    for k in keys[3:5]:
        assert k in connector.engine.index.pending_store
    # save chunk_ids 也在 pending 表内（同一 opaque 路径）
    for k, cid in zip(rm.save_spec.chunk_keys, rm.save_spec.chunk_ids):
        assert connector.engine.index.pending_store[k] == cid

    # token 序列与 slot_mapping 覆盖 640 token（5 块）
    assert len(rm.token_ids) == 5 * CS
    assert len(rm.slot_mapping) == 5 * CS

    # tracker 状态推进
    assert connector._request_trackers["m1"].num_saved_tokens == 5 * CS


def test_build_connector_meta_new_request_save_only():
    """无命中新请求：仅 save 计划（5 个 chunk 全新）。"""
    connector = make_connector(make_extra())
    nr = make_new_request("s1", PROMPT, [0, 1, 2, 3, 4])
    out = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"s1": 5 * CS}
    )
    meta = connector.build_connector_meta(out)
    assert len(meta.requests) == 1
    rm = meta.requests[0]
    assert rm.load_spec is None
    assert rm.save_spec.can_save
    assert len(rm.save_spec.chunk_keys) == 5
    assert len(rm.save_spec.chunk_ids) == 5
    assert len(connector.engine.index.pending_store) == 5


def test_build_connector_meta_force_skip_save():
    """force_skip_save 且无 load → ReqMeta None → meta 空。"""
    connector = make_connector(make_extra(force_skip_save=True))
    nr = make_new_request("s2", PROMPT, [0, 1, 2, 3, 4])
    out = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"s2": 5 * CS}
    )
    meta = connector.build_connector_meta(out)
    assert meta.requests == []


def test_build_connector_meta_finished_requests_cleanup():
    """finished_req_ids 清理 tracker 与残留 load_spec。"""
    connector = make_connector(make_extra())
    nr = make_new_request("f1", PROMPT, [0, 1, 2, 3, 4])
    out = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"f1": 5 * CS}
    )
    connector.build_connector_meta(out)
    assert "f1" in connector._request_trackers
    connector.load_specs["f1"] = LoadSpec("f1", 0, 0)

    out2 = make_sched_output(finished=["f1"])
    meta = connector.build_connector_meta(out2)
    assert meta.requests == []
    assert "f1" not in connector._request_trackers
    assert "f1" not in connector.load_specs


def test_build_connector_meta_cached_request_incremental():
    """cached 请求增量：第二步追加 200 token（840 总量）→ 新存 1 chunk。"""
    connector = make_connector(make_extra())
    keys = store_prefix(connector, PROMPT, 5)  # 历史已存 5 chunk

    # 第一步：new request 全量（外部命中 640 = 全部）
    req = fake_request("c1", PROMPT)
    connector.get_num_new_matched_tokens(req, 0)
    nr = make_new_request("c1", PROMPT, [0, 1, 2, 3, 4])
    out1 = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"c1": 5 * CS}
    )
    connector.build_connector_meta(out1)  # skip_leading=640，无新存

    # 第二步：decode 追加 200 token（840 总量，boundary 768 ≤ 840）
    grown = PROMPT + list(range(2000, 2000 + 200))
    cached = CachedRequestData.make_empty()
    cached.req_ids = ["c1"]
    cached.resumed_req_ids = set()
    cached.new_token_ids = [[]]
    cached.all_token_ids = {"c1": grown}
    cached.new_block_ids = [([5],)]      # 追加一个 block
    cached.num_computed_tokens = [5 * CS]
    cached.num_output_tokens = [200]
    out2 = make_sched_output(
        cached=cached, num_scheduled_tokens={"c1": 200}
    )
    meta = connector.build_connector_meta(out2)

    assert len(meta.requests) == 1
    rm = meta.requests[0]
    tracker = connector._request_trackers["c1"]
    assert tracker.token_ids == grown            # all_token_ids 增量合并
    assert tracker.allocated_block_ids == [0, 1, 2, 3, 4, 5]
    # save：skip_leading=640（5 chunk），840 → 768 边界 → 新存 1 chunk
    # （增量 chunk key 在增长后序列上链式滚动）
    grown_keys = connector.engine.index.keys_for_tokens(grown)
    assert rm.save_spec.skip_leading_tokens == 5 * CS
    assert rm.save_spec.chunk_keys == tuple(grown_keys[5:6])
    assert len(rm.save_spec.chunk_ids) == 1
    assert tracker.num_saved_tokens == 6 * CS


def test_build_connector_meta_preempted_recovery():
    """resumed_req_ids：token 序列恢复 + 块号整体替换。"""
    connector = make_connector(make_extra())
    base = list(range(3000, 3000 + 2 * CS))
    keys = store_prefix(connector, base, 2)  # 2 chunk 已存

    req = fake_request("p1", base)
    connector.get_num_new_matched_tokens(req, 0)
    nr = make_new_request("p1", base, [0, 1])
    out1 = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"p1": 2 * CS}
    )
    connector.build_connector_meta(out1)

    # preemption 后恢复：追加 128 token（384 总量），块号替换为 [9,10,11]
    grown = base + list(range(4000, 4000 + CS))
    cached = CachedRequestData.make_empty()
    cached.req_ids = ["p1"]
    cached.resumed_req_ids = {"p1"}
    cached.new_token_ids = [[]]
    cached.all_token_ids = {"p1": grown}
    cached.new_block_ids = [([9, 10, 11],)]
    cached.num_computed_tokens = [2 * CS]
    cached.num_output_tokens = [CS]
    out2 = make_sched_output(
        cached=cached, num_scheduled_tokens={"p1": CS}
    )
    meta = connector.build_connector_meta(out2)

    tracker = connector._request_trackers["p1"]
    assert tracker.token_ids == grown          # 恢复全序列
    assert tracker.allocated_block_ids == [9, 10, 11]  # 替换而非追加

    assert len(meta.requests) == 1
    rm = meta.requests[0]
    # save：skip_leading=256，384 → boundary 384 → 新存 1 chunk（第 3 个，
    # key 在恢复后的全序列上链式滚动）
    grown_keys = connector.engine.index.keys_for_tokens(grown)
    assert rm.save_spec.chunk_keys == tuple(grown_keys[2:3])
    # slot_mapping 用新块号（9,10,11）：token 0 / 128 / 256 分别落块 9/10/11
    assert int(rm.slot_mapping[0]) == 9 * BS
    assert int(rm.slot_mapping[BS]) == 10 * BS
    assert int(rm.slot_mapping[2 * BS]) == 11 * BS


def test_build_connector_meta_load_key_missing_falls_back():
    """plan_load 前被驱逐 → load 放弃（不崩溃，回退本地计算）。"""
    connector = make_connector(make_extra())
    keys = store_prefix(connector, PROMPT, 3)

    req = fake_request("e1", PROMPT)
    connector.get_num_new_matched_tokens(req, 0)
    connector.update_state_after_alloc(req, blocks=None, num_external_tokens=3 * CS)

    # 模拟 lookup 与 plan 之间被驱逐：直接从索引移除
    victim = keys[2]
    path = connector.engine.index.stored.pop(victim)
    connector.engine.index.free.append(path)

    nr = make_new_request("e1", PROMPT, [0, 1, 2, 3, 4], num_computed=3 * CS)
    out = make_sched_output(
        new_reqs=[nr], num_scheduled_tokens={"e1": 2 * CS}
    )
    meta = connector.build_connector_meta(out)
    rm = meta.requests[0]
    assert rm.load_spec is None   # 已回退：只保留 save 计划


# ---------------------------------------------------------------- 术语铁律


def test_no_forbidden_storage_terms():
    """验收点 6：源码不出现介质词（extent/shard/nvme/prp/fiap/fallocate/_dma）。"""
    src = (_ROOT / "adapter" / "connector.py").read_text(
        encoding="utf-8"
    )
    forbidden = re.compile(
        r"extent|shard|nvme|\bprp\b|fiemap|fallocate|_dma"
    )
    m = forbidden.search(src)
    assert m is None, f"forbidden storage term {m.group()!r} in vllm_adapter.py"


# ---------------------------------------------------------------- 构造契约（主 session 验收补丁）


def test_factory_three_arg_instantiation():
    """真 vllm KVConnectorFactory 以 3 参实例化（base.py __init__ 签名）。"""
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BS),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=make_extra()
        ),
    )
    conn = TuttiConnectorV1(
        vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=None
    )
    assert isinstance(conn, KVConnectorBase_V1)


def test_scheduler_and_worker_share_engine():
    """同 vllm_config 的双角色共享同一 TuttiEngine（pin/pending 一致性）。"""
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BS),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=make_extra()
        ),
    )
    sched = TuttiConnectorV1(vllm_config, KVConnectorRole.SCHEDULER)
    worker = TuttiConnectorV1(vllm_config, KVConnectorRole.WORKER)
    assert sched.engine is worker.engine


def test_explicit_engine_instance_bypasses_registry():
    """extra_config['tutti_engine_instance'] 直传实例，绕过共享注册表。"""
    from engine.core import TuttiEngine

    mine = TuttiEngine(
        config={
            "backend": "memory",
            "chunk_tokens": CS,
            "chunk_kv_bytes": CHUNK_KV_BYTES,
            "capacity_bytes": CAPACITY,
            "max_chunks_per_wave": 8,
        }
    )
    conn = make_connector(make_extra(tutti_engine_instance=mine))
    assert conn.engine is mine
