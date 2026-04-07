# Vortex Components: Next Session High-Signal Context

## 1) Purpose
This file is a compact handoff for continuing implementation with low chat context cost.
Primary focus is building DAG-oriented scheduling internals in vortex_components, aligned with DesignDoc.pdf and current repository architecture.

## 2) Repo Landmarks

### Core project roots
- `vortex/`
- `cascade/`
- `derecho/`

### Vortex areas relevant to this work
- `vortex/src/vortex_components/`
  - `CMakeLists.txt`
  - `include/vortex_components/arena.hpp`
  - `include/vortex_components/arena_impl.hpp`
  - `include/vortex_components/core.hpp`
  - `tests/arena_tests.cpp`
  - `benchmarks/arena_bench.cpp`
  - `DesignDoc.pdf`
- `vortex/example_pipeline/`
  - `example_udl.cpp`
  - `console_printer_udl.cpp`
  - `messages/example_message.hpp`

### Cascade config flow (where dfgs.json enters)
- `cascade/src/service/data_flow_graph.cpp`
  - File load entry point for DFG definitions.
- `cascade/include/cascade/data_flow_graph.hpp`
  - Parsed DFG vertex + UDL configuration storage.
- `cascade/include/cascade/detail/service_impl.hpp`
  - During startup, parsed UDL config is passed to UDL observer creation.
- `cascade/include/cascade/detail/user_defined_logic_manager_impl.hpp`
  - UDL symbol dispatch and observer acquisition.

Important implication:
- UDLs and components should consume config JSON passed into observer initialization.
- Do not add runtime filesystem reads of dfgs.json in ocdpo hot path.

## 3) Key outcomes from DesignDoc (captured from provided pages)

### System model
- Central scheduler for DAG-structured jobs.
- Two message classes:
  - Control messages: scheduler intent and worker status.
  - Data messages: task output payloads and intermediate values.

### Worker internal boundary
- ocdpo handler is ingress ownership boundary.
- Payload that outlives callback must be copied/moved into worker-owned memory before callback returns.

### A: Join responsibilities
- Decode enough metadata at ingress.
- Move/copy payload into Input Arena and produce BlobHandle.
- Enqueue lightweight ingress metadata + handle.
- Join thread drains queue and performs dependency assembly.

### Required internal pieces
- Input Arena
- Ingress Queue
- Join Table
- DAG Registry

### DAG Registry contract
For each (graph_id, task_id):
- expected_inputs
- invoke_handler
- downstream

### Message sketches from doc to preserve
- TaskRef
- Decision
- SchedulerCommand
- WorkerStatus
- TaskOutputHeader
- IngressRecord
- BlobHandle
- TaskBinding
- TaskSpec

## 4) Messaging approach decided

### Serialization strategy
- Use mutils-compatible structs for control and headers.
- Keep large data payload opaque for ingress; decode only header in ocdpo.

### Helper pattern adopted
- `into_blob(T&&)` helper in `example_pipeline/messages/example_message.hpp` to avoid intermediate vector copy and serialize via Blob generator.

### Producer identity strategy
- Sender node id is insufficient to distinguish UDL producer.
- Use key namespace and/or payload metadata field to identify producing path.

## 5) Current components include layout decision
- Goal: include style should be `<vortex_components/xyz.hpp>`.
- Wrapper headers created under `include/vortex_components/`.
- CMake include path adjusted for tests/benchmarks to point at include root.

## 6) Implementation plan (phased)

### Phase 1: Message contracts in vortex_components
Add message definitions for control and data headers:
- TaskRef, Decision, SchedulerCommand, WorkerStatus
- TaskOutputHeader
- Runtime-only structs: IngressRecord, TaskBinding

### Phase 2: Static DAGRegistry
- Parse DAG config JSON passed through UDL config.
- Validate required fields and graph consistency.
- Provide lookup API by (graph_id, task_id).
- Prefer immutable snapshot model after load.

### Phase 3: Dynamic JoinTable
- Track dependency slots by destination TaskRef.
- Emit TaskBinding when all slots satisfied.
- Guard duplicate slot writes and malformed dependency ids.

### Phase 4: Ingress glue helpers
- Decode TaskOutputHeader from object blob.
- Resolve dependency slot from ingress bindings.
- Copy payload to arena and enqueue lightweight IngressRecord.

### Phase 5: Tests
- Parser and schema validation tests.
- Registry lookup and validation tests.
- JoinTable completion semantics tests.
- Message codec decode/encode tests.

### Phase 6: UDL integration
- Load registry in initialize/get_observer from config JSON.
- Keep ocdpo path minimal and non-blocking.

## 7) Planned file changes

### Modify
- `vortex/src/vortex_components/CMakeLists.txt`
- `vortex/src/vortex_components/include/vortex_components/core.hpp`
- `vortex/src/vortex_components/README.md`

### Add
- `vortex/src/vortex_components/include/vortex_components/messages.hpp`
- `vortex/src/vortex_components/include/vortex_components/dag_registry.hpp`
- `vortex/src/vortex_components/include/vortex_components/join_table.hpp`
- `vortex/src/vortex_components/src/message_codec.cpp`
- `vortex/src/vortex_components/src/dag_registry.cpp`
- `vortex/src/vortex_components/src/join_table.cpp`
- `vortex/src/vortex_components/tests/message_codec_tests.cpp`
- `vortex/src/vortex_components/tests/dag_registry_tests.cpp`
- `vortex/src/vortex_components/tests/join_table_tests.cpp`

## 8) Build/test/benchmark iteration commands

Assume build dir: `/home/jq54/workspace/vortex/build-Debug`

### Fast target builds
```bash
cmake --build /home/jq54/workspace/vortex/build-Debug --target dag_registry_tests -j32
cmake --build /home/jq54/workspace/vortex/build-Debug --target join_table_tests -j32
cmake --build /home/jq54/workspace/vortex/build-Debug --target message_codec_tests -j32
```

### Component test sweep
```bash
ctest --test-dir /home/jq54/workspace/vortex/build-Debug --output-on-failure -R "arena|dag_registry|join_table|message_codec"
```

### Single-test debug run
```bash
/home/jq54/workspace/vortex/build-Debug/src/vortex_components/dag_registry_tests
/home/jq54/workspace/vortex/build-Debug/src/vortex_components/join_table_tests
/home/jq54/workspace/vortex/build-Debug/src/vortex_components/message_codec_tests
```

### Full project compile before checkpoint
```bash
cmake --build /home/jq54/workspace/vortex/build-Debug -j32
```

### Benchmarks
```bash
cmake --build /home/jq54/workspace/vortex/build-Debug --target arena_benchmarks -j32
/home/jq54/workspace/vortex/build-Debug/src/vortex_components/arena_benchmarks \
  --benchmark_min_time=0.3 \
  --benchmark_repetitions=8 \
  --benchmark_report_aggregates_only=true
```

### Benchmark result capture
```bash
/home/jq54/workspace/vortex/build-Debug/src/vortex_components/arena_benchmarks \
  --benchmark_out=/tmp/arena_bench_after.json \
  --benchmark_out_format=json
```

## 9) Guardrails and pitfalls
- Keep ocdpo callback short: parse minimal metadata, copy payload to arena, enqueue, return.
- Avoid direct dfg file I/O from ocdpo or join path.
- Use immutable DAGRegistry snapshots by default for lock-free reads.
- Do not rely on sender node id alone for producer identity; key/payload metadata should encode producer/path intent.

## 10) Immediate next coding step
Implement Phase 1 scaffolding (messages.hpp + tests) and wire CMake targets so unit tests run independently before adding DAGRegistry and JoinTable logic.

## 11) Current implementation status (updated)

Completed:
- Phase 1 message contracts and codec tests.
- Phase 2 DAG registry parser/validator and unit tests.
- Phase 3 join table implementation and unit tests.
- DFG adapter path: DAG registry can now be derived from DFG graph JSON to avoid duplicate DAG encoding.

New files now present:
- `include/vortex_components/messages.hpp`
- `include/vortex_components/dag_registry.hpp`
- `include/vortex_components/join_table.hpp`
- `tests/message_codec_tests.cpp`
- `tests/dag_registry_tests.cpp`
- `tests/join_table_tests.cpp`

Build/test status:
- `message_codec_tests`: passing (4 test cases, 33 assertions)
- `dag_registry_tests`: passing (8 test cases, 34 assertions)
- `join_table_tests`: passing (6 test cases, 30 assertions)

Next immediate step:
- Phase 4: ingress glue helpers (header decode + dependency-slot mapping + ingress record creation path).

Additional notes:
- `example_pipeline/example_udl.cpp` now prefers registry init from explicit config, then from provided DFG JSON, then from fallback files (`cfg/dfgs.json.tmp`, `example_pipeline/cfg/dfgs.json.tmp`) using `DAGRegistry::from_dfg_json(..., MY_UUID)`.
- Duplicate inline `dag_registry` block was removed from `example_pipeline/cfg/dfgs.json.tmp` to keep DFG structure as single source of truth.
