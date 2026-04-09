#include <catch2/catch.hpp>

#include <filesystem>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <vortex_scheduler/task_join_service.hpp>
#include <vortex_scheduler/messages.hpp>
#include <diamond_messages.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

std::string write_temp_dfg() {
  const std::string json = R"({
    "tasks": [
      {"task_id":0, "pathname":"/A", "udl_uuid":"u"},
      {"task_id":1, "pathname":"/B", "udl_uuid":"u"},
      {"task_id":2, "pathname":"/C", "udl_uuid":"u"},
      {"task_id":3, "pathname":"/D", "udl_uuid":"u"}
    ],
    "graphs": [
      {"graph_id":0,
       "task_list":[0,1,2,3],
       "upstream_by_task":[
         {"task_id":1, "from":[0]},
         {"task_id":2, "from":[0]},
         {"task_id":3, "from":[1,2]}
       ]}
    ]
  })";

  auto path = std::filesystem::temp_directory_path() / "dfg_test.json";
  std::ofstream out(path);
  out << json;
  return path.string();
}

std::vector<std::byte> make_packet(const scheduler::TaskOutput& header,
                                   const std::vector<std::byte>& payload) {
  const auto hsize = header.size_estimate();
  std::vector<std::byte> buf(hsize + payload.size());
  header.to_bytes(reinterpret_cast<uint8_t*>(buf.data()));
  std::memcpy(buf.data() + hsize, payload.data(), payload.size());
  return buf;
}

}  // namespace

TEST_CASE("TaskJoinService emits binding when all deps present", "[task_join_service]") {
  const auto dfg_path = write_temp_dfg();
  auto dag = scheduler::DagRegistry::from_dfg_file(dfg_path, "u");
  scheduler::TaskJoinService service(dag);

  // input from B (task 1)
  StepBMessage msgB{"msgB"};
  std::vector<std::byte> pB(msgB.size_estimate());
  msgB.to_bytes(reinterpret_cast<uint8_t*>(pB.data()));

  scheduler::TaskOutput hB{7, 42, 3, 1, 0, static_cast<uint32_t>(pB.size())};
  auto pktB = make_packet(hB, pB);

  auto first = service.recv("/D", std::span<const std::byte>(pktB.data(), pktB.size()));
  CHECK_FALSE(first.has_value());

  // input from C (task 2)
  StepCMessage msgC{"msgC"};
  std::vector<std::byte> pC(msgC.size_estimate());
  msgC.to_bytes(reinterpret_cast<uint8_t*>(pC.data()));

  scheduler::TaskOutput hC{8, 42, 3, 2, 0, static_cast<uint32_t>(pC.size())};
  auto pktC = make_packet(hC, pC);

  auto second = service.recv("/D", std::span<const std::byte>(pktC.data(), pktC.size()));
  REQUIRE(second.has_value());

  const auto& binding = *second;
  CHECK(binding.task.graph_id == 0);
  CHECK(binding.task.job_id == 42);
  CHECK(binding.task.task_id == 3);
  REQUIRE(binding.inputs.size() == 2);

  auto payload0 = service.resolve(binding.inputs[0]);
  auto payload1 = service.resolve(binding.inputs[1]);
  REQUIRE(payload0.has_value());
  REQUIRE(payload1.has_value());

  StepBMessage outB = *StepBMessage::from_bytes(nullptr, reinterpret_cast<const uint8_t*>(payload0->data()));
  StepCMessage outC = *StepCMessage::from_bytes(nullptr, reinterpret_cast<const uint8_t*>(payload1->data()));
  CHECK(outB.message == "msgB");
  CHECK(outC.message == "msgC");
}

TEST_CASE("TaskJoinService emits binding for root task ingress", "[task_join_service]") {
  const auto dfg_path = write_temp_dfg();
  auto dag = scheduler::DagRegistry::from_dfg_file(dfg_path, "u");
  scheduler::TaskJoinService service(dag);

  StepAMessage msgA{"seedA"};
  std::vector<std::byte> pA(msgA.size_estimate());
  msgA.to_bytes(reinterpret_cast<uint8_t*>(pA.data()));

  // Root task A has no upstream dependencies in this DAG.
  scheduler::TaskOutput hA{1, 500, 0, 0, 0, static_cast<uint32_t>(pA.size())};
  auto pktA = make_packet(hA, pA);

  auto binding = service.recv("/A/500", std::span<const std::byte>(pktA.data(), pktA.size()));
  REQUIRE(binding.has_value());
  CHECK(binding->task.graph_id == 0);
  CHECK(binding->task.job_id == 500);
  CHECK(binding->task.task_id == 0);
  REQUIRE(binding->inputs.size() == 1);

  auto payload = service.resolve(binding->inputs[0]);
  REQUIRE(payload.has_value());
  auto outA = StepAMessage::from_bytes(nullptr, reinterpret_cast<const uint8_t*>(payload->data()));
  REQUIRE(outA != nullptr);
  CHECK(outA->message == "seedA");
}
