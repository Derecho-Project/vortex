#include <cstdint>
#include <string>
#include <derecho/mutils-serialization/SerializationSupport.hpp>
#include <vortex_scheduler/prelude.hpp>

struct StepAMessage : public mutils::ByteRepresentable,
					  public scheduler::VortexSerde<StepAMessage> {
	std::string message;

	StepAMessage() = default;
	StepAMessage(std::string message)
		: message(std::move(message)) { }

	DEFAULT_SERIALIZATION_SUPPORT(StepAMessage, message);
};

struct StepBMessage : public mutils::ByteRepresentable,
					  public scheduler::VortexSerde<StepBMessage> {
	std::string message;

	StepBMessage() = default;
	explicit StepBMessage(std::string message)
		: message(std::move(message)) { }

	DEFAULT_SERIALIZATION_SUPPORT(StepBMessage, message);
};

struct StepCMessage : public mutils::ByteRepresentable,
					  public scheduler::VortexSerde<StepCMessage> {
	std::string message;

	StepCMessage() = default;
	explicit StepCMessage(std::string message)
		: message(std::move(message)) { }

	DEFAULT_SERIALIZATION_SUPPORT(StepCMessage, message);
};
