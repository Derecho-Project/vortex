#include <catch2/catch.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include <vortex_scheduler/dag_registry.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

nlohmann::json make_valid_jobs_config() {
  return nlohmann::json{
      {"tasks",
       nlohmann::json::array(
           {{{"task_id", 0}, {"pathname", "/A"}, {"udl_uuid", "uuid-a"}},
            {{"task_id", 1}, {"pathname", "/B"}, {"udl_uuid", "uuid-b"}},
            {{"task_id", 2}, {"pathname", "/C"}, {"udl_uuid", "uuid-c"}},
            {{"task_id", 3}, {"pathname", "/D"}, {"udl_uuid", "uuid-d"}}})},
      {"graphs",
       nlohmann::json::array(
           {{{"graph_id", 0},
             {"description", "canonical diamond DFG"},
             {"task_list", nlohmann::json::array({0, 1, 2, 3})},
             {"upstream_by_task",
              nlohmann::json::array({{{"task_id", 0}, {"from", nlohmann::json::array()}},
                                      {{"task_id", 1}, {"from", nlohmann::json::array({0})}},
                                      {{"task_id", 2}, {"from", nlohmann::json::array({0})}},
                                      {{"task_id", 3}, {"from", nlohmann::json::array({1, 2})}}})}}})}};
}

std::filesystem::path write_temp_json(const nlohmann::json& json,
                                      const std::string& filename) {
  namespace fs = std::filesystem;
  const auto path = fs::temp_directory_path() / filename;
  std::ofstream out(path);
  if(!out.is_open()) {
    throw std::runtime_error("failed to open temp file for writing: " + path.string());
  }
  out << json.dump(2);
  return path;
}

}  // namespace

TEST_CASE("DagRegistry parses jobs config and derives downstream", "[dag_registry]") {
  namespace fs = std::filesystem;
  const auto path = write_temp_json(make_valid_jobs_config(), "vortex_dag_registry_jobs_valid.json");

  const auto registry = DagRegistry::from_dfg_file(path.string(), "uuid-a");
  REQUIRE(registry.graph_count() == 1);

  const auto* t0 = registry.find_task(0, 0);
  const auto* t1 = registry.find_task(0, 1);
  const auto* t2 = registry.find_task(0, 2);
  const auto* t3 = registry.find_task(0, 3);
  REQUIRE(t0 != nullptr);
  REQUIRE(t1 != nullptr);
  REQUIRE(t2 != nullptr);
  REQUIRE(t3 != nullptr);

  CHECK(t0->pathname == "/A");
  CHECK(t0->udl_uuid == "uuid-a");
  CHECK(t0->upstream.empty());
  REQUIRE(t0->downstream.size() == 2);
  CHECK(std::find(t0->downstream.begin(), t0->downstream.end(), 1) != t0->downstream.end());
  CHECK(std::find(t0->downstream.begin(), t0->downstream.end(), 2) != t0->downstream.end());

  REQUIRE(t3->upstream.size() == 2);
  CHECK(std::find(t3->upstream.begin(), t3->upstream.end(), 1) != t3->upstream.end());
  CHECK(std::find(t3->upstream.begin(), t3->upstream.end(), 2) != t3->upstream.end());
  CHECK(t3->downstream.empty());

  const auto t1_id = registry.find_task_id_by_path("/B");
  REQUIRE(t1_id.has_value());
  CHECK(*t1_id == 1);

  const auto missing_id = registry.find_task_id_by_path("/missing");
  CHECK_FALSE(missing_id.has_value());

  std::error_code ec;
  fs::remove(path, ec);
}

TEST_CASE("DagRegistry rejects unmatched udl uuid", "[dag_registry]") {
  namespace fs = std::filesystem;
  const auto path = write_temp_json(make_valid_jobs_config(), "vortex_dag_registry_jobs_no_match.json");

  REQUIRE_THROWS_WITH(
      DagRegistry::from_dfg_file(path.string(), "uuid-missing"),
      Catch::Contains("No graph matched the requested UDL uuid."));

  std::error_code ec;
  fs::remove(path, ec);
}

TEST_CASE("DagRegistry rejects duplicate task ids", "[dag_registry]") {
  namespace fs = std::filesystem;
  auto cfg = make_valid_jobs_config();
  cfg["tasks"].push_back(
      {{"task_id", 1}, {"pathname", "/B2"}, {"udl_uuid", "uuid-b2"}});

  const auto path = write_temp_json(cfg, "vortex_dag_registry_jobs_dup_task.json");

  REQUIRE_THROWS_WITH(
      DagRegistry::from_dfg_file(path.string(), ""),
      Catch::Contains("Duplicate task_id"));

  std::error_code ec;
  fs::remove(path, ec);
}

TEST_CASE("DagRegistry rejects unknown upstream task", "[dag_registry]") {
  namespace fs = std::filesystem;
  auto cfg = make_valid_jobs_config();
  cfg["graphs"][0]["upstream_by_task"][1]["from"] = nlohmann::json::array({99});

  const auto path = write_temp_json(cfg, "vortex_dag_registry_jobs_bad_upstream.json");

  REQUIRE_THROWS_WITH(
      DagRegistry::from_dfg_file(path.string(), ""),
      Catch::Contains("references unknown upstream task"));

  std::error_code ec;
  fs::remove(path, ec);
}

TEST_CASE("DagRegistry lookup returns null for absent ids", "[dag_registry]") {
  namespace fs = std::filesystem;
  const auto path = write_temp_json(make_valid_jobs_config(), "vortex_dag_registry_jobs_lookup.json");

  const auto registry = DagRegistry::from_dfg_file(path.string(), "");
  CHECK(registry.find_task(999, 0) == nullptr);
  CHECK(registry.find_task(0, 999) == nullptr);
  CHECK(registry.find_upstream(999, 0) == nullptr);
  CHECK(registry.find_upstream(0, 999) == nullptr);

  std::error_code ec;
  fs::remove(path, ec);
}
