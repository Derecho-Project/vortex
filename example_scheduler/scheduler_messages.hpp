#pragma once
#include <cstdint>
#include <cstring>
#include <span>

#include <vortex_scheduler/serde.hpp>

namespace derecho::cascade {

struct MetadataA {
	double example_field4 = 0.0;
	float example_field3 = 0.0f;
	std::uint32_t example_field1 = 0;
	char example_field2 = '\0';
	bool example_field5 = false;
	void* random_ptr_value = (void*)0x239832;
};

struct StepATag {};
struct StepBTag {};
struct StepCTag {};
struct StepDTag {};

template <typename Family, typename Tag = StepATag>
struct StepAMessageT {
	using string_repr_t = scheduler::serde::string_repr_t<Family, char>;
	using message_t = typename string_repr_t::type;

	// NOTE: this struct is memcpy'd onto the wire, so:
	//  - field order is chosen to minimize padding holes (largest align first)
	//  - the in-class default initializers below ensure padding bytes are
	//    well-defined (zero) instead of leaking arbitrary stack memory.
	//  - all participants are assumed to share endianness, FP representation,
	//    and ABI padding.

	MetadataA metadata{ };
	message_t message{ };

	/// @brief Deserialize from a wire-format payload.
	/// @note  For ViewFamily, `message` aliases `payload`; the returned object
	///        is only valid for the lifetime of `payload`.
	/// @note  For OwnedFamily, allocates and may throw std::bad_alloc; not noexcept.
	[[nodiscard]] static StepAMessageT<Family, Tag> from_buffer(std::span<const std::byte> payload);
	[[nodiscard]] std::size_t to_buffer(const std::span<std::byte>& buffer) const noexcept;
	[[nodiscard]] std::size_t size_estimate() const noexcept;
};

using StepAMessageView = StepAMessageT<scheduler::serde::ViewFamily,  StepATag>;
using StepAMessage     = StepAMessageT<scheduler::serde::OwnedFamily, StepATag>;
using StepBMessageView = StepAMessageT<scheduler::serde::ViewFamily,  StepBTag>;
using StepBMessage     = StepAMessageT<scheduler::serde::OwnedFamily, StepBTag>;
using StepCMessageView = StepAMessageT<scheduler::serde::ViewFamily,  StepCTag>;
using StepCMessage     = StepAMessageT<scheduler::serde::OwnedFamily, StepCTag>;
using StepDMessageView = StepAMessageT<scheduler::serde::ViewFamily,  StepDTag>;
using StepDMessage     = StepAMessageT<scheduler::serde::OwnedFamily, StepDTag>;

static_assert(scheduler::serde::BufferSerializable<StepAMessage>);
static_assert(scheduler::serde::BufferSerializable<StepAMessageView>);
static_assert(scheduler::serde::BufferSerializable<StepBMessage>);
static_assert(scheduler::serde::BufferSerializable<StepBMessageView>);
static_assert(scheduler::serde::BufferSerializable<StepCMessage>);
static_assert(scheduler::serde::BufferSerializable<StepCMessageView>);
static_assert(scheduler::serde::BufferSerializable<StepDMessage>);
static_assert(scheduler::serde::BufferSerializable<StepDMessageView>);

} // namespace derecho::cascade

#include "scheduler_messages_impl.tpp"