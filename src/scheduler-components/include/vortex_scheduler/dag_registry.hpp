#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include <vortex_scheduler/core.hpp>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

struct DagNodeSpec {
  uint16_t task_id = 0;
  std::string pathname;
  std::string udl_uuid;
  std::vector<uint16_t> upstream;
  std::vector<uint16_t> downstream;
};

struct DagGraphSpec {
  uint16_t graph_id = 0;
  std::unordered_map<uint16_t, DagNodeSpec> tasks;
};

class DagRegistry {
 public:
  DagRegistry() = default;

  /// @brief load the registry from a json file path
  static DagRegistry from_dfg_file(const std::string& path, const std::string& udl_uuid);

  const DagNodeSpec* find_task(uint16_t graph_id, uint16_t task_id) const;
  const DagGraphSpec* find_graph(uint16_t graph_id) const;

  std::size_t graph_count() const;

  const std::vector<uint16_t>* find_upstream(uint16_t graph_id,
                                              uint16_t task_id) const;

  std::optional<uint16_t> find_task_id_by_path(const std::string& pathname) const;

  void print_registry_debug(std::ostream& out = std::cout) const;

 private:
  std::unordered_map<uint16_t, DagGraphSpec> graphs_;

  static nlohmann::json load_json_file(const std::string& path);

  static uint16_t parse_u16(const nlohmann::json& json,
                            const char* field,
                            const char* scope);

  void finalize_graphs();
};

VORTEX_SCHEDULER_NAMESPACE_END
