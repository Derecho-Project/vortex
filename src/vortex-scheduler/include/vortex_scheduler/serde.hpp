#pragma once
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace scheduler::serde {

// ~~~ Reflection between view and owned types ~~~
// A type-family / HKD trait: parameterize a message struct on `Family` and use
// `reflection_t<Family, T>::type` for fields whose representation differs
// between an owning copy (`OwnedFamily`) and a zero-copy view (`ViewFamily`).
//
// Contract for specializations:
//   - `type` is the concrete representation
//   - `make(span<const T>)` constructs `type` from a contiguous range of `T`
//   - `view(const type&)` returns a `span<const T>` valid for the lifetime of
//     its argument
// `make(view(x))` must round-trip equal to `x`.
struct ViewFamily;
struct OwnedFamily;

template <typename Family, typename T>
struct string_repr_t {
	static_assert(sizeof(Family) == 0,
		"string_repr_t: no specialization for this (Family, T) pair");
};

template <typename T>
struct string_repr_t<OwnedFamily, T> {
	using type = std::basic_string<T>;

	// Allocates; can throw std::bad_alloc.
	static type make(std::span<const T> s) {
		return type(s.data(), s.size());
	}

	static std::span<const T> view(const type& v) noexcept {
		return std::span<const T>(v.data(), v.size());
	}
};

template <typename T>
struct string_repr_t<ViewFamily, T> {
	using type = std::basic_string_view<T>;

	static type make(std::span<const T> s) noexcept {
		return type(s.data(), s.size());
	}

	static std::span<const T> view(const type& v) noexcept {
		return std::span<const T>(v.data(), v.size());
	}
};

// ~~~ Array repr (vector vs span) ~~~
// Same family pattern, but for variable-length arrays of arbitrary trivially-
// copyable element types: OwnedFamily -> std::vector<T>, ViewFamily -> std::span<const T>.
//
// Contract is identical to reflection_t: `make(view(x)) == x`, and the span
// returned by `view` is valid for the lifetime of its argument.
template <typename Family, typename T>
struct array_repr_t {
	static_assert(sizeof(Family) == 0,
		"array_repr_t: no specialization for this Family");
};

template <typename T>
struct array_repr_t<OwnedFamily, T> {
	using type = std::vector<T>;

	// Allocates; can throw std::bad_alloc.
	static type make(std::span<const T> s) {
		return type(s.begin(), s.end());
	}

	static std::span<const T> view(const type& v) noexcept {
		return std::span<const T>(v.data(), v.size());
	}
};

template <typename T>
struct array_repr_t<ViewFamily, T> {
	using type = std::span<const T>;

	static type make(std::span<const T> s) noexcept { return s; }

	static std::span<const T> view(const type& v) noexcept { return v; }
};


/// @brief concept for types that can be serialized by our custom serde implementation of slice types.
template <typename T>
concept Serializable = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

template <typename T>
concept BufferSerializable =
	requires(const T& t, std::span<const std::byte> payload, std::span<std::byte> buffer) {
		// from_buffer may allocate (e.g. OwnedFamily) so it is NOT required to be noexcept.
		{ T::from_buffer(payload) } -> std::same_as<T>;
		{ t.to_buffer(buffer) } noexcept -> std::same_as<std::size_t>;
		{ t.size_estimate() } noexcept -> std::same_as<std::size_t>;
	} && !std::is_pointer_v<T>;

/// @brief returns padding in num_bytes to align the next write to the alignment requirements of T (base_ptr + offset + <return value>)
///	@note assumes that the base_ptr is suitably aligned for all types being written, meaning that the first write must be with offset = 0
template <Serializable T>
static constexpr std::size_t pad_aligned_to(const std::size_t offset) noexcept;

/// @brief computes the size of a serialized message without actually serializing it
/// @note constexpr if all components of the message are constexpr-constructible, otherwise it is a runtime function
struct Sizer {
    std::size_t count = 0;

	[[nodiscard]] constexpr std::size_t bytes_written() const noexcept;

	template <Serializable T>
	constexpr void put() noexcept;

	template <Serializable T>
	constexpr void put_span(const std::span<T>& v) noexcept;
};

/// @brief writes to a pre-allocated buffer, ensuring proper alignment and padding for each type
struct Writer {
	std::span<std::byte> buf;
	std::size_t pos = 0;
	
	[[nodiscard]] std::size_t remaining() const noexcept;
	[[nodiscard]] std::size_t bytes_written() const noexcept;

	template <Serializable T>
	void put(const T& v) noexcept;

	template <Serializable T>
	void put_span(const std::span<T>& v) noexcept;
};


/// @brief helper for deserialization, similar to rust nom's "cursor" pattern
struct Reader {
	std::span<const std::byte> buf;
	std::size_t pos = 0;

	[[nodiscard]] std::size_t remaining() const noexcept;

	template <Serializable T>
	[[nodiscard]] T get() noexcept;

	template <Serializable T>
	[[nodiscard]] std::span<const T> get_span(std::size_t count) noexcept;
};

} // namespace scheduler::serde

#include "detail/serde_impl.hpp"