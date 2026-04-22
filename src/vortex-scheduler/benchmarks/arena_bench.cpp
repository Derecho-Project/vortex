#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <vortex_scheduler/arena.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

std::vector<std::byte> make_payload(std::size_t size, unsigned int seed) {
  std::vector<std::byte> payload(size);
  for (std::size_t i = 0; i < size; ++i) {
    payload[i] = static_cast<std::byte>((seed + i * 29u) & 0xFFu);
  }
  return payload;
}

void BM_PutUniqueSmall(benchmark::State &state) {
  ArenaAllocator<std::uint64_t> arena;
  const auto payload = make_payload(static_cast<std::size_t>(state.range(0)), 7);
  std::uint64_t key = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        arena.put(key++, std::span<const std::byte>(payload)));
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * state.range(0));
}

void BM_PutOverwriteSameKey(benchmark::State &state) {
  ArenaAllocator<std::uint64_t> arena;
  const auto payload =
      make_payload(static_cast<std::size_t>(state.range(0)), 19);
  constexpr std::uint64_t key = 42;

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        arena.put(key, std::span<const std::byte>(payload)));
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * state.range(0));
}

void BM_GetHot(benchmark::State &state) {
  ArenaAllocator<std::uint64_t> arena;
  const auto payload = make_payload(64, 31);

  for (std::uint64_t key = 0; key < 4096; ++key) {
    arena.put(key, std::span<const std::byte>(payload));
  }

  std::uint64_t key = 0;
  for (auto _ : state) {
    const auto handle = arena.get(key++ & 4095u);
    if (handle.has_value()) {
      auto allocation_id = handle->allocation_id;
      benchmark::DoNotOptimize(allocation_id);
    }
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_ResolveHot(benchmark::State &state) {
  ArenaAllocator<std::uint64_t> arena;
  const auto payload =
      make_payload(static_cast<std::size_t>(state.range(0)), 47);
  std::vector<BufferHandle> handles;
  handles.reserve(4096);

  for (std::uint64_t key = 0; key < 4096; ++key) {
    handles.push_back(arena.put(key, std::span<const std::byte>(payload)));
  }

  std::size_t index = 0;
  for (auto _ : state) {
    const auto resolved = arena.resolve_const(handles[index++ & 4095u]);
    if (resolved.has_value()) {
      benchmark::DoNotOptimize(resolved->data());
      benchmark::DoNotOptimize(resolved->size());
    }
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * state.range(0));
}

void BM_PutTakeChurn(benchmark::State &state) {
  ArenaAllocator<std::uint64_t> arena;
  const auto payload =
      make_payload(static_cast<std::size_t>(state.range(0)), 61);
  constexpr std::uint64_t key_space = 1024;
  std::uint64_t key = 0;

  for (auto _ : state) {
    const std::uint64_t current = key++ % key_space;
    benchmark::DoNotOptimize(
        arena.put(current, std::span<const std::byte>(payload)));
    if ((current & 1u) == 0u) {
      benchmark::DoNotOptimize(arena.take(current));
    }
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * state.range(0));
}

void BM_LongRunningWindowedChurn(benchmark::State &state) {
  const std::size_t payload_size = static_cast<std::size_t>(state.range(0));
  const std::size_t window = static_cast<std::size_t>(state.range(1));
  ArenaAllocator<std::uint64_t> arena;
  const auto payload = make_payload(payload_size, 83);

  std::uint64_t key = 0;
  for (auto _ : state) {
    const std::uint64_t current = key++;
    benchmark::DoNotOptimize(
        arena.put(current, std::span<const std::byte>(payload)));
    if (current >= window) {
      benchmark::DoNotOptimize(arena.take(current - window));
    }
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(payload_size));
}

void BM_LongRunningWindowedChurnReserved(benchmark::State &state) {
  const std::size_t payload_size = static_cast<std::size_t>(state.range(0));
  const std::size_t window = static_cast<std::size_t>(state.range(1));
  const std::size_t tuned_segment_capacity =
      std::max<std::size_t>(1 << 20, payload_size * 1024);
  ArenaAllocator<std::uint64_t> arena(tuned_segment_capacity);
  arena.reserve(window, window, (window * payload_size) / tuned_segment_capacity + 8);
  const auto payload = make_payload(payload_size, 97);

  std::uint64_t key = 0;
  for (auto _ : state) {
    const std::uint64_t current = key++;
    benchmark::DoNotOptimize(
        arena.put(current, std::span<const std::byte>(payload)));
    if (current >= window) {
      benchmark::DoNotOptimize(arena.take(current - window));
    }
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(payload_size));
}

BENCHMARK(BM_PutUniqueSmall)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_PutOverwriteSameKey)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_GetHot);
BENCHMARK(BM_ResolveHot)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_PutTakeChurn)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_LongRunningWindowedChurn)->Args({64, 1 << 16})->Args({256, 1 << 16});
BENCHMARK(BM_LongRunningWindowedChurnReserved)
    ->Args({64, 1 << 16})
    ->Args({256, 1 << 16});

} // namespace

BENCHMARK_MAIN();
