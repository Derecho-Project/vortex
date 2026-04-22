
#include <catch2/catch.hpp>
#include <vortex_scheduler/arena.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>

namespace scheduler {
namespace {

std::vector<std::byte> make_patterned_bytes(std::size_t size, std::uint64_t seed) {
	std::vector<std::byte> bytes(size);
	std::uint64_t state = seed;
	for(std::size_t i = 0; i < size; ++i) {
		state = state * 6364136223846793005ULL + 1;
		bytes[i] = std::byte{ static_cast<unsigned char>((state >> 32) & 0xFFU) };
	}
	return bytes;
}

void require_span_equals(std::span<const std::byte> actual, std::span<const std::byte> expected) {
	REQUIRE(actual.size() == expected.size());
	CHECK(std::ranges::equal(actual, expected));
}

} // namespace

TEST_CASE("Simple arena put and view", "[arena]") {
	Arena arena(20'971'520);
	std::vector<std::byte> payload1{ std::byte(0xDE), std::byte(0xAD), std::byte(0xBE), std::byte(0xEF) };
	std::vector<std::byte> payload2{ std::byte(0xEF), std::byte(0xBE), std::byte(0xAD), std::byte(0xDE) };

	auto h1 = arena.put(payload1);
	auto h2 = arena.put(payload2);

	REQUIRE(std::ranges::equal(arena.view(h1), payload1));
	REQUIRE(std::ranges::equal(arena.view(h2), payload2));
}

TEST_CASE("Bad put with payload larger than segment capacity", "[arena]") {
	Arena arena(1);
	std::vector<std::byte> payload1{ std::byte(0xDE), std::byte(0xAD), std::byte(0xBE), std::byte(0xEF) };
	REQUIRE_THROWS_AS(arena.put(payload1), std::runtime_error);
}

TEST_CASE("Take invalidates handle and frees space", "[arena]") {
	Arena arena(20'971'520);
	std::vector<std::byte> payload{ std::byte(0xDE), std::byte(0xAD), std::byte(0xBE), std::byte(0xEF) };

	auto h1 = arena.put(payload);
	auto h2 = arena.put(payload);

	arena.take(std::move(h1));

	REQUIRE(arena.view(h1).empty());
	REQUIRE(std::ranges::equal(arena.view(h2), payload));
}

TEST_CASE("Multiple takes on same handle are safe", "[arena]") {
	Arena arena(20'971'520);
	std::vector<std::byte> payload{ std::byte(0xDE), std::byte(0xAD), std::byte(0xBE), std::byte(0xEF) };

	auto h1 = arena.put(payload);

	arena.take(std::move(h1));
	arena.take(std::move(h1)); // should not crash or double free

	REQUIRE(arena.view(h1).empty());
}

TEST_CASE("Zero-length payloads round trip and invalidate correctly", "[arena][edge]") {
	Arena arena(8);
	std::vector<std::byte> payload;

	auto handle = arena.put(payload);
	REQUIRE(arena.view(handle).empty());

	arena.take(std::move(handle));
	REQUIRE(arena.view(handle).empty());
}

TEST_CASE("Largest payload that fits in a segment round trips", "[arena][edge]") {
	Arena arena(8);
	auto payload = make_patterned_bytes(7, 0xA11CEULL);

	auto handle = arena.put(payload);
	require_span_equals(arena.view(handle), payload);
}

TEST_CASE("Payload equal to segment capacity is rejected", "[arena][edge]") {
	Arena arena(8);
	auto payload = make_patterned_bytes(8, 0xBAD5EEDULL);
	REQUIRE_THROWS_AS(arena.put(payload), std::runtime_error);
}

TEST_CASE("Freed segments are reused with a bumped generation", "[arena][edge]") {
	Arena arena(8);
	auto payload = make_patterned_bytes(3, 0x1234ULL);

	auto first = arena.put(payload);
	arena.take(std::move(first));

	auto second = arena.put(payload);

	REQUIRE(second.segment == first.segment);
	REQUIRE(second.generation > first.generation);
	REQUIRE(arena.view(first).empty());
	require_span_equals(arena.view(second), payload);
}

TEST_CASE("Multiple segments", "[arena]") {
	Arena arena(sizeof(int) * 5);
	std::vector<int> payload{ 1, 2, 3, 4};
	std::span<std::byte> payload_bytes{ reinterpret_cast<std::byte*>(payload.data()), sizeof(int) * payload.size() };

	auto h1 = arena.put(payload_bytes);
	auto h2 = arena.put(payload_bytes);

	REQUIRE(std::ranges::equal(arena.view(h1), payload_bytes));
	REQUIRE(std::ranges::equal(arena.view(h2), payload_bytes));

	arena.take(std::move(h1));

	REQUIRE(arena.view(h1).empty());
	REQUIRE(std::ranges::equal(arena.view(h2), payload_bytes));

	auto h3 = arena.put(payload_bytes);
	REQUIRE(std::ranges::equal(arena.view(h3), payload_bytes));
	REQUIRE(std::ranges::equal(arena.view(h2), payload_bytes));
	REQUIRE(arena.view(h1).empty());
	REQUIRE(h2.segment != h3.segment);
}

TEST_CASE("Deterministic churn fuzz preserves live payloads", "[arena][fuzz]") {
	constexpr std::size_t kSegmentCapacity = 96;
	constexpr std::size_t kSlots = 32;
	constexpr std::size_t kIterations = 2'000;

	struct SlotState {
		std::optional<Arena::BufferHandle> handle;
		std::vector<std::byte> payload;
	};

	Arena arena(kSegmentCapacity);
	std::vector<SlotState> slots(kSlots);
	std::vector<Arena::BufferHandle> retired_handles;
	std::mt19937_64 rng(0xC0FFEEULL);
	std::uniform_int_distribution<std::size_t> slot_dist(0, kSlots - 1);
	std::uniform_int_distribution<int> action_dist(0, 2);
	std::uniform_int_distribution<std::size_t> size_dist(0, kSegmentCapacity - 1);

	for(std::size_t iteration = 0; iteration < kIterations; ++iteration) {
		const std::size_t slot_index = slot_dist(rng);
		SlotState& slot = slots[slot_index];
		INFO("iteration=" << iteration << ", slot=" << slot_index);

		if(action_dist(rng) == 0 && slot.handle.has_value()) {
			retired_handles.push_back(*slot.handle);
			arena.take(std::move(*slot.handle));
			REQUIRE(arena.view(retired_handles.back()).empty());
			slot.handle.reset();
			slot.payload.clear();
		} else {
			if(slot.handle.has_value()) {
				retired_handles.push_back(*slot.handle);
				arena.take(std::move(*slot.handle));
				REQUIRE(arena.view(retired_handles.back()).empty());
			}

			slot.payload = make_patterned_bytes(size_dist(rng), rng());
			slot.handle = arena.put(slot.payload);
			require_span_equals(arena.view(*slot.handle), slot.payload);
		}

		for(const SlotState& live_slot : slots) {
			if(!live_slot.handle.has_value()) {
				continue;
			}
			require_span_equals(arena.view(*live_slot.handle), live_slot.payload);
		}
	}

	for(const Arena::BufferHandle& retired_handle : retired_handles) {
		CHECK(arena.view(retired_handle).empty());
	}
}

TEST_CASE("Deterministic growth fuzz keeps handles valid across many segments", "[arena][fuzz]") {
	constexpr std::size_t kSegmentCapacity = 64;
	constexpr std::size_t kAllocations = 256;

	Arena arena(kSegmentCapacity);
	std::vector<Arena::BufferHandle> handles;
	std::vector<std::vector<std::byte>> payloads;
	std::mt19937_64 rng(0x5EED1234ULL);
	std::uniform_int_distribution<std::size_t> size_dist(0, kSegmentCapacity - 1);

	handles.reserve(kAllocations);
	payloads.reserve(kAllocations);

	for(std::size_t i = 0; i < kAllocations; ++i) {
		payloads.push_back(make_patterned_bytes(size_dist(rng), rng()));
		handles.push_back(arena.put(payloads.back()));
	}

	for(std::size_t i = 0; i < handles.size(); ++i) {
		INFO("initial verification index=" << i);
		require_span_equals(arena.view(handles[i]), payloads[i]);
	}

	for(std::size_t i = 0; i < handles.size(); i += 2) {
		arena.take(std::move(handles[i]));
		REQUIRE(arena.view(handles[i]).empty());
	}

	for(std::size_t i = 1; i < handles.size(); i += 2) {
		INFO("post-take verification index=" << i);
		require_span_equals(arena.view(handles[i]), payloads[i]);
	}
}

} // namespace scheduler