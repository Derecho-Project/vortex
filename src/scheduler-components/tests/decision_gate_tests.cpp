#include <catch2/catch.hpp>

#include <vector>

#include <vortex_scheduler/decision_gate.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

struct Packet {
	int id = 0;
};

DecisionGateKey make_key(uint16_t graph_id, uint16_t job_id, uint16_t source_task_id, uint16_t target_task_id) {
	return DecisionGateKey {
		.task = TaskRef {graph_id, job_id, source_task_id},
		.target_task_id = target_task_id,
	};
}

} // namespace

TEST_CASE("DecisionGate flushes only credited key", "[decision_gate]") {
	DecisionGate<Packet> gate;
	const auto key_a = make_key(0, 10, 1, 2);
	const auto key_b = make_key(0, 10, 1, 3);

	gate.enqueue(key_a, Packet {1});
	gate.enqueue(key_b, Packet {2});
	gate.add_credit(key_a, 1);

	std::vector<int> sent;
	auto sender = [&sent](Packet&& p) { sent.push_back(p.id); };

	const auto flushed = gate.flush_key(key_a, sender);
	CHECK(flushed == 1);
	REQUIRE(sent.size() == 1);
	CHECK(sent[0] == 1);
	CHECK(gate.pending_packet_count() == 1);
}

TEST_CASE("DecisionGate retains result until credit arrives", "[decision_gate]") {
	DecisionGate<Packet> gate;
	const auto key = make_key(0, 11, 2, 5);
	gate.enqueue(key, Packet {7});

	std::vector<int> sent;
	auto sender = [&sent](Packet&& p) { sent.push_back(p.id); };

	CHECK(gate.flush_key(key, sender) == 0);
	CHECK(sent.empty());
	CHECK(gate.pending_packet_count() == 1);

	gate.add_credit(key, 1);
	CHECK(gate.flush_key(key, sender) == 1);
	REQUIRE(sent.size() == 1);
	CHECK(sent[0] == 7);
	CHECK(gate.pending_packet_count() == 0);
}

TEST_CASE("DecisionGate cancel drops pending and credit", "[decision_gate]") {
	DecisionGate<Packet> gate;
	const auto key = make_key(0, 20, 4, 9);
	gate.enqueue(key, Packet {3});
	gate.add_credit(key, 1);

	gate.cancel_task(TaskRef {0, 20, 4});

	std::vector<int> sent;
	auto sender = [&sent](Packet&& p) { sent.push_back(p.id); };
	CHECK(gate.flush_key(key, sender) == 0);
	CHECK(sent.empty());
	CHECK(gate.pending_packet_count() == 0);
}

TEST_CASE("DecisionGate respects credit count", "[decision_gate]") {
	DecisionGate<Packet> gate;
	const auto key = make_key(0, 30, 6, 8);

	gate.enqueue(key, Packet {1});
	gate.enqueue(key, Packet {2});
	gate.enqueue(key, Packet {3});
	gate.add_credit(key, 2);

	std::vector<int> sent;
	auto sender = [&sent](Packet&& p) { sent.push_back(p.id); };

	CHECK(gate.flush_key(key, sender) == 2);
	REQUIRE(sent.size() == 2);
	CHECK(sent[0] == 1);
	CHECK(sent[1] == 2);
	CHECK(gate.pending_packet_count() == 1);
}
