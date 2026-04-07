#include <catch2/catch.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <vortex_scheduler/arena.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

std::vector<std::byte> make_bytes(std::initializer_list<unsigned int> values) {
  std::vector<std::byte> bytes;
  bytes.reserve(values.size());
  for (unsigned int value : values) {
    bytes.push_back(static_cast<std::byte>(value));
  }
  return bytes;
}

std::vector<std::byte> make_patterned_bytes(std::size_t size,
                                            unsigned int seed) {
  std::vector<std::byte> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::byte>((seed + i * 17u) & 0xFFu);
  }
  return bytes;
}

void require_span_equals(std::span<const std::byte> actual,
                         std::span<const std::byte> expected) {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == expected[i]);
  }
}

} // namespace

TEST_CASE("put/get/resolve returns stored payload", "[arena]") {
  ArenaAllocator<int> arena;
  const auto payload = make_bytes({1, 2, 3, 4});

  const BufferHandle handle = arena.put(7, std::span<const std::byte>(payload));

  const auto stored = arena.get(7);
  REQUIRE(stored.has_value());
  CHECK(stored->allocation_id == handle.allocation_id);

  const auto resolved = arena.resolve_const(handle);
  REQUIRE(resolved.has_value());
  REQUIRE(resolved->size() == payload.size());
  CHECK((*resolved)[0] == payload[0]);
  CHECK((*resolved)[1] == payload[1]);
  CHECK((*resolved)[2] == payload[2]);
  CHECK((*resolved)[3] == payload[3]);
}

TEST_CASE("duplicate put invalidates previous handle and replaces bytes",
          "[arena]") {
  ArenaAllocator<int> arena;
  const auto first_payload = make_bytes({1, 2, 3});
  const auto second_payload = make_bytes({9, 8, 7, 6});

  const BufferHandle first = arena.put(1, std::span<const std::byte>(first_payload));
  const BufferHandle second =
      arena.put(1, std::span<const std::byte>(second_payload));

  CHECK_FALSE(arena.resolve_const(first).has_value());

  const auto latest = arena.get(1);
  REQUIRE(latest.has_value());
  CHECK(latest->allocation_id == second.allocation_id);

  const auto resolved = arena.resolve_const(second);
  REQUIRE(resolved.has_value());
  REQUIRE(resolved->size() == second_payload.size());
  CHECK((*resolved)[0] == second_payload[0]);
  CHECK((*resolved)[1] == second_payload[1]);
  CHECK((*resolved)[2] == second_payload[2]);
  CHECK((*resolved)[3] == second_payload[3]);
}

TEST_CASE("take removes key and invalidates token even if segment stays live",
          "[arena]") {
  ArenaAllocator<int> arena;
  const auto payload_a = make_bytes({1, 2});
  const auto payload_b = make_bytes({3, 4});

  const BufferHandle first = arena.put(1, std::span<const std::byte>(payload_a));
  const BufferHandle sibling = arena.put(2, std::span<const std::byte>(payload_b));

  const auto taken = arena.take(1);
  REQUIRE(taken.has_value());
  CHECK(taken->allocation_id == first.allocation_id);

  CHECK_FALSE(arena.get(1).has_value());
  CHECK_FALSE(arena.resolve_const(first).has_value());
  CHECK(arena.resolve_const(sibling).has_value());
}

TEST_CASE("fully freed latest segment can be reused while stale handles stay invalid",
          "[arena]") {
  ArenaAllocator<int> arena;
  std::vector<std::byte> payload(32, std::byte{0x2A});

  const BufferHandle first = arena.put(1, std::span<const std::byte>(payload));
  REQUIRE(arena.take(1).has_value());
  CHECK_FALSE(arena.resolve_const(first).has_value());

  const BufferHandle next = arena.put(2, std::span<const std::byte>(payload));

  CHECK(next.segment_id == first.segment_id);
  CHECK(next.generation > first.generation);
  CHECK(next.offset == 0);
  CHECK(arena.resolve_const(next).has_value());
}

TEST_CASE("allocation larger than default capacity gets its own segment",
          "[arena]") {
  ArenaAllocator<int> arena;
  std::vector<std::byte> large_payload(5000, std::byte{0x5A});

  const BufferHandle handle =
      arena.put(99, std::span<const std::byte>(large_payload));

  const auto resolved = arena.resolve_const(handle);
  REQUIRE(resolved.has_value());
  CHECK(resolved->size() == large_payload.size());
  CHECK((*resolved)[0] == std::byte{0x5A});
  CHECK((*resolved)[4999] == std::byte{0x5A});
}

TEST_CASE("zero length payloads round trip and invalidate correctly", "[arena]") {
  ArenaAllocator<int> arena;

  const BufferHandle handle = arena.put(5, std::span<const std::byte>{});

  const auto stored = arena.get(5);
  REQUIRE(stored.has_value());
  CHECK(stored->len == 0);

  const auto resolved = arena.resolve_const(handle);
  REQUIRE(resolved.has_value());
  CHECK(resolved->empty());

  REQUIRE(arena.take(5).has_value());
  CHECK_FALSE(arena.resolve_const(handle).has_value());
}

TEST_CASE("mutable resolve exposes in place bytes for live handles", "[arena]") {
  ArenaAllocator<int> arena;
  const auto payload = make_bytes({1, 2, 3, 4});

  const BufferHandle handle = arena.put(11, std::span<const std::byte>(payload));

  auto writable = arena.resolve(handle);
  REQUIRE(writable.has_value());
  REQUIRE(writable->len == payload.size());

  writable->bytes[1] = std::byte{0xAA};
  writable->bytes[3] = std::byte{0xBB};

  const auto resolved = arena.resolve_const(handle);
  REQUIRE(resolved.has_value());
  CHECK((*resolved)[0] == std::byte{0x01});
  CHECK((*resolved)[1] == std::byte{0xAA});
  CHECK((*resolved)[2] == std::byte{0x03});
  CHECK((*resolved)[3] == std::byte{0xBB});
}

TEST_CASE("latest segment is append only until capacity is exhausted", "[arena]") {
  ArenaAllocator<int> arena;

  const auto first_payload = make_patterned_bytes(1500, 1);
  const auto second_payload = make_patterned_bytes(1500, 2);
  const auto third_payload = make_patterned_bytes(1500, 3);

  const BufferHandle first =
      arena.put(1, std::span<const std::byte>(first_payload));
  const BufferHandle second =
      arena.put(2, std::span<const std::byte>(second_payload));
  const BufferHandle third =
      arena.put(3, std::span<const std::byte>(third_payload));

  CHECK(first.segment_id == second.segment_id);
  CHECK(second.segment_id != third.segment_id);
  CHECK(first.offset == 0);
  CHECK(second.offset == first_payload.size());
  CHECK(third.offset == 0);

  REQUIRE(arena.resolve_const(first).has_value());
  REQUIRE(arena.resolve_const(second).has_value());
  REQUIRE(arena.resolve_const(third).has_value());
}

TEST_CASE("older freed gaps are not reused while latest segment still has room",
          "[arena]") {
  ArenaAllocator<int> arena;

  const auto payload_a = make_patterned_bytes(256, 10);
  const auto payload_b = make_patterned_bytes(256, 20);
  const auto payload_c = make_patterned_bytes(128, 30);

  const BufferHandle first =
      arena.put(1, std::span<const std::byte>(payload_a));
  const BufferHandle second =
      arena.put(2, std::span<const std::byte>(payload_b));
  REQUIRE(arena.take(1).has_value());

  const BufferHandle third =
      arena.put(3, std::span<const std::byte>(payload_c));

  CHECK(second.segment_id == third.segment_id);
  CHECK(third.offset == payload_a.size() + payload_b.size());
  CHECK_FALSE(arena.resolve_const(first).has_value());
}

TEST_CASE("repeated overwrite churn invalidates every superseded handle",
          "[arena][stress]") {
  ArenaAllocator<std::string> arena;
  std::vector<BufferHandle> stale_handles;
  std::vector<std::byte> expected;
  std::optional<BufferHandle> previous;

  for (std::size_t i = 0; i < 512; ++i) {
    auto payload = make_patterned_bytes((i % 97) + 1, static_cast<unsigned int>(i));
    if (previous.has_value()) {
      stale_handles.push_back(*previous);
    }
    const BufferHandle handle =
        arena.put("shared-key", std::span<const std::byte>(payload));
    expected = std::move(payload);
    previous = handle;

    const auto latest = arena.get("shared-key");
    REQUIRE(latest.has_value());
    const auto resolved = arena.resolve_const(*latest);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, expected);
  }

  for (const BufferHandle &stale : stale_handles) {
    CHECK_FALSE(arena.resolve_const(stale).has_value());
  }
}

TEST_CASE("stress mix of many keys preserves live values and invalidates taken ones",
          "[arena][stress]") {
  ArenaAllocator<int> arena;
  std::vector<std::optional<BufferHandle>> live_handles(64);
  std::vector<std::vector<std::byte>> expected_payloads(64);
  std::vector<BufferHandle> retired_handles;

  for (int iteration = 0; iteration < 2000; ++iteration) {
    const int key = iteration % 64;

    if (iteration % 5 == 0) {
      if (live_handles[key].has_value()) {
        retired_handles.push_back(*live_handles[key]);
        const auto taken = arena.take(key);
        REQUIRE(taken.has_value());
        live_handles[key].reset();
        expected_payloads[key].clear();
      } else {
        CHECK_FALSE(arena.take(key).has_value());
      }
      continue;
    }

    if (live_handles[key].has_value()) {
      retired_handles.push_back(*live_handles[key]);
    }

    expected_payloads[key] = make_patterned_bytes(
        static_cast<std::size_t>((iteration % 113) + 1),
        static_cast<unsigned int>(iteration + key * 13));
    live_handles[key] =
        arena.put(key, std::span<const std::byte>(expected_payloads[key]));

    const auto stored = arena.get(key);
    REQUIRE(stored.has_value());
    REQUIRE(stored->allocation_id == live_handles[key]->allocation_id);

    const auto resolved = arena.resolve_const(*stored);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, expected_payloads[key]);
  }

  for (std::size_t key = 0; key < live_handles.size(); ++key) {
    const auto current = arena.get(static_cast<int>(key));
    if (!live_handles[key].has_value()) {
      CHECK_FALSE(current.has_value());
      continue;
    }

    REQUIRE(current.has_value());
    CHECK(current->allocation_id == live_handles[key]->allocation_id);
    const auto resolved = arena.resolve_const(*current);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, expected_payloads[key]);
  }

  for (const BufferHandle &retired : retired_handles) {
    CHECK_FALSE(arena.resolve_const(retired).has_value());
  }
}

TEST_CASE("handles from retired segments stay invalid across continued growth",
          "[arena][stress]") {
  ArenaAllocator<int> arena;
  std::vector<BufferHandle> retired;

  for (int segment = 0; segment < 12; ++segment) {
    std::vector<int> keys;
    for (int i = 0; i < 3; ++i) {
      const int key = segment * 10 + i;
      auto payload = make_patterned_bytes(1700, static_cast<unsigned int>(segment * 10 + i));
      retired.push_back(arena.put(key, std::span<const std::byte>(payload)));
      keys.push_back(key);
    }

    for (int key : keys) {
      REQUIRE(arena.take(key).has_value());
    }
  }

  for (int i = 0; i < 64; ++i) {
    auto payload = make_patterned_bytes(96, static_cast<unsigned int>(1000 + i));
    const BufferHandle live =
        arena.put(1000 + i, std::span<const std::byte>(payload));
    REQUIRE(arena.resolve_const(live).has_value());
  }

  for (const BufferHandle &handle : retired) {
    CHECK_FALSE(arena.resolve_const(handle).has_value());
  }
}

TEST_CASE("forged handles are rejected even when fields look plausible",
          "[arena][edge]") {
  ArenaAllocator<int> arena;
  const auto payload = make_patterned_bytes(64, 77);
  const BufferHandle real = arena.put(1, std::span<const std::byte>(payload));

  BufferHandle wrong_allocation = real;
  ++wrong_allocation.allocation_id;
  CHECK_FALSE(arena.resolve_const(wrong_allocation).has_value());

  BufferHandle wrong_generation = real;
  ++wrong_generation.generation;
  CHECK_FALSE(arena.resolve_const(wrong_generation).has_value());

  BufferHandle wrong_offset = real;
  ++wrong_offset.offset;
  CHECK_FALSE(arena.resolve_const(wrong_offset).has_value());

  BufferHandle wrong_len = real;
  ++wrong_len.len;
  CHECK_FALSE(arena.resolve_const(wrong_len).has_value());

  BufferHandle wrong_segment = real;
  wrong_segment.segment_id += 100;
  CHECK_FALSE(arena.resolve_const(wrong_segment).has_value());

  REQUIRE(arena.resolve_const(real).has_value());
}

TEST_CASE("allocation ids increase monotonically across puts and rewrites",
          "[arena][edge]") {
  ArenaAllocator<int> arena;
  std::vector<BufferHandle> handles;

  for (int i = 0; i < 50; ++i) {
    auto payload = make_patterned_bytes(static_cast<std::size_t>((i % 7) + 1),
                                        static_cast<unsigned int>(i));
    handles.push_back(arena.put(i % 5, std::span<const std::byte>(payload)));
  }

  REQUIRE(handles.size() == 50);
  for (std::size_t i = 1; i < handles.size(); ++i) {
    CHECK(handles[i].allocation_id > handles[i - 1].allocation_id);
  }
}

TEST_CASE("segment generation advances each time a segment is retired",
          "[arena][edge]") {
  ArenaAllocator<int> arena;
  const auto payload = make_patterned_bytes(128, 91);

  const BufferHandle first = arena.put(1, std::span<const std::byte>(payload));
  REQUIRE(arena.take(1).has_value());

  const BufferHandle second = arena.put(2, std::span<const std::byte>(payload));
  REQUIRE(arena.take(2).has_value());

  const BufferHandle third = arena.put(3, std::span<const std::byte>(payload));

  CHECK(second.segment_id == first.segment_id);
  CHECK(third.segment_id == second.segment_id);
  CHECK(second.generation == first.generation + 1);
  CHECK(third.generation == second.generation + 1);
  CHECK_FALSE(arena.resolve_const(first).has_value());
  CHECK_FALSE(arena.resolve_const(second).has_value());
  REQUIRE(arena.resolve_const(third).has_value());
}

TEST_CASE("exact fill forces next allocation onto a fresh segment",
          "[arena][edge]") {
  ArenaAllocator<int> arena;
  const auto first_payload = make_patterned_bytes(2048, 1);
  const auto second_payload = make_patterned_bytes(2048, 2);
  const auto third_payload = make_patterned_bytes(1, 3);

  const BufferHandle first =
      arena.put(1, std::span<const std::byte>(first_payload));
  const BufferHandle second =
      arena.put(2, std::span<const std::byte>(second_payload));
  const BufferHandle third =
      arena.put(3, std::span<const std::byte>(third_payload));

  CHECK(first.segment_id == second.segment_id);
  CHECK(second.offset == first_payload.size());
  CHECK(third.segment_id != second.segment_id);
  CHECK(third.offset == 0);
}

TEST_CASE("taking a missing key is harmless during churn", "[arena][edge]") {
  ArenaAllocator<int> arena;

  CHECK_FALSE(arena.take(404).has_value());

  for (int i = 0; i < 100; ++i) {
    auto payload = make_patterned_bytes(static_cast<std::size_t>((i % 33) + 1),
                                        static_cast<unsigned int>(200 + i));
    const int key = i % 8;
    arena.put(key, std::span<const std::byte>(payload));
    CHECK_FALSE(arena.take(1000 + i).has_value());
  }
}

TEST_CASE("string keys remain isolated under heavy overwrite churn",
          "[arena][stress]") {
  ArenaAllocator<std::string> arena;
  std::vector<std::string> keys = {"alpha", "beta", "gamma", "delta", "omega"};
  std::vector<std::vector<std::byte>> expected(keys.size());
  std::vector<std::optional<BufferHandle>> handles(keys.size());

  for (int iteration = 0; iteration < 1500; ++iteration) {
    const std::size_t index = static_cast<std::size_t>(iteration % keys.size());
    expected[index] = make_patterned_bytes(
        static_cast<std::size_t>((iteration % 61) + 4),
        static_cast<unsigned int>(iteration * 3));
    handles[index] =
        arena.put(keys[index], std::span<const std::byte>(expected[index]));

    if (iteration % 11 == 0) {
      const std::size_t victim =
          static_cast<std::size_t>((iteration / 11) % keys.size());
      if (handles[victim].has_value()) {
        REQUIRE(arena.take(keys[victim]).has_value());
        handles[victim].reset();
        expected[victim].clear();
      }
    }
  }

  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto current = arena.get(keys[i]);
    if (!handles[i].has_value()) {
      CHECK_FALSE(current.has_value());
      continue;
    }

    REQUIRE(current.has_value());
    CHECK(current->allocation_id == handles[i]->allocation_id);
    const auto resolved = arena.resolve_const(*current);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, expected[i]);
  }
}

TEST_CASE("long deterministic workload keeps all surviving handles valid",
          "[arena][stress]") {
  ArenaAllocator<int> arena;
  struct ExpectedRecord {
    BufferHandle handle;
    std::vector<std::byte> payload;
    bool live = false;
  };

  std::vector<ExpectedRecord> records(128);
  std::vector<BufferHandle> retired;

  for (int iteration = 0; iteration < 5000; ++iteration) {
    const int key = (iteration * 17) % 128;

    if (records[key].live && (iteration % 9 == 0)) {
      retired.push_back(records[key].handle);
      REQUIRE(arena.take(key).has_value());
      records[key].live = false;
      records[key].payload.clear();
      continue;
    }

    if (records[key].live) {
      retired.push_back(records[key].handle);
    }

    records[key].payload = make_patterned_bytes(
        static_cast<std::size_t>((iteration % 127) + 1),
        static_cast<unsigned int>(iteration + key));
    records[key].handle =
        arena.put(key, std::span<const std::byte>(records[key].payload));
    records[key].live = true;

    const auto resolved = arena.resolve_const(records[key].handle);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, records[key].payload);
  }

  for (int key = 0; key < 128; ++key) {
    const auto current = arena.get(key);
    if (!records[key].live) {
      CHECK_FALSE(current.has_value());
      continue;
    }

    REQUIRE(current.has_value());
    CHECK(current->allocation_id == records[key].handle.allocation_id);
    const auto resolved = arena.resolve_const(*current);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, records[key].payload);
  }

  for (const BufferHandle &handle : retired) {
    CHECK_FALSE(arena.resolve_const(handle).has_value());
  }
}

TEST_CASE("reserved allocator survives long churn without losing correctness",
          "[arena][stress]") {
  ArenaAllocator<int> arena(1 << 20);
  arena.reserve(2048, 2048, 32);

  std::vector<std::optional<BufferHandle>> handles(2048);
  std::vector<std::vector<std::byte>> expected(2048);
  std::vector<BufferHandle> retired;

  for (int iteration = 0; iteration < 12000; ++iteration) {
    const int key = iteration % 2048;

    if (handles[key].has_value()) {
      retired.push_back(*handles[key]);
    }

    expected[key] = make_patterned_bytes(
        static_cast<std::size_t>((iteration % 48) + 16),
        static_cast<unsigned int>(500 + iteration));
    handles[key] = arena.put(key, std::span<const std::byte>(expected[key]));

    if ((iteration % 7) == 0) {
      const int victim = (iteration / 7) % 2048;
      if (handles[victim].has_value()) {
        retired.push_back(*handles[victim]);
        REQUIRE(arena.take(victim).has_value());
        handles[victim].reset();
        expected[victim].clear();
      }
    }
  }

  for (int key = 0; key < 2048; ++key) {
    const auto current = arena.get(key);
    if (!handles[key].has_value()) {
      CHECK_FALSE(current.has_value());
      continue;
    }

    REQUIRE(current.has_value());
    CHECK(current->allocation_id == handles[key]->allocation_id);
    const auto resolved = arena.resolve_const(*current);
    REQUIRE(resolved.has_value());
    require_span_equals(*resolved, expected[key]);
  }

  for (const BufferHandle &retired_handle : retired) {
    CHECK_FALSE(arena.resolve_const(retired_handle).has_value());
  }
}

TEST_CASE("slot reuse still rejects stale handles after repeated take and put",
          "[arena][edge]") {
  ArenaAllocator<int> arena;
  arena.reserve(1, 1, 1);

  const auto first_payload = make_patterned_bytes(32, 1);
  const auto second_payload = make_patterned_bytes(32, 2);
  const auto third_payload = make_patterned_bytes(32, 3);

  const BufferHandle first =
      arena.put(9, std::span<const std::byte>(first_payload));
  REQUIRE(arena.take(9).has_value());
  const BufferHandle second =
      arena.put(9, std::span<const std::byte>(second_payload));
  REQUIRE(arena.take(9).has_value());
  const BufferHandle third =
      arena.put(9, std::span<const std::byte>(third_payload));

  CHECK(first.slot_id == second.slot_id);
  CHECK(second.slot_id == third.slot_id);
  CHECK(first.slot_version < second.slot_version);
  CHECK(second.slot_version < third.slot_version);
  CHECK_FALSE(arena.resolve_const(first).has_value());
  CHECK_FALSE(arena.resolve_const(second).has_value());

  const auto resolved = arena.resolve_const(third);
  REQUIRE(resolved.has_value());
  require_span_equals(*resolved, third_payload);
}
