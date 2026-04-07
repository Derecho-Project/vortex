#include <vortex_scheduler/dag_registry.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

DagRegistry DagRegistry::from_dfg_file(const std::string& path,
                                       const std::string& udl_uuid) {
  const auto config = load_json_file(path);

  if(!config.is_object()) {
    throw std::runtime_error("jobs config root must be a JSON object.");
  }
  if(!config.contains("tasks") || !config.at("tasks").is_array()) {
    throw std::runtime_error("jobs config must contain array field 'tasks'.");
  }
  if(!config.contains("graphs") || !config.at("graphs").is_array()) {
    throw std::runtime_error("jobs config must contain array field 'graphs'.");
  }

  std::unordered_map<uint16_t, DagNodeSpec> task_catalog;
  for(const auto& task_json : config.at("tasks")) {
    if(!task_json.is_object()) {
      throw std::runtime_error("Each task entry must be a JSON object.");
    }

    DagNodeSpec task;
    task.task_id = parse_u16(task_json, "task_id", "task");
    if(!task_json.contains("pathname") || !task_json.at("pathname").is_string()) {
      throw std::runtime_error(
          "Task " + std::to_string(task.task_id) + " must contain string field 'pathname'.");
    }
    task.pathname = task_json.at("pathname").get<std::string>();

    if(!task_json.contains("udl_uuid") || !task_json.at("udl_uuid").is_string()) {
      throw std::runtime_error(
          "Task " + std::to_string(task.task_id) + " must contain string field 'udl_uuid'.");
    }
    task.udl_uuid = task_json.at("udl_uuid").get<std::string>();

    auto [_, inserted] = task_catalog.emplace(task.task_id, std::move(task));
    if(!inserted) {
      throw std::runtime_error("Duplicate task_id in top-level tasks: " + std::to_string(task_json.at("task_id").get<int64_t>()));
    }
  }

  DagRegistry registry;

  for(const auto& graph_json : config.at("graphs")) {
    if(!graph_json.is_object()) {
      throw std::runtime_error("Each graph entry must be a JSON object.");
    }

    const uint16_t graph_id = parse_u16(graph_json, "graph_id", "graph");
    if(!graph_json.contains("task_list") || !graph_json.at("task_list").is_array()) {
      throw std::runtime_error("Graph " + std::to_string(graph_id) + " must contain array field 'task_list'.");
    }

    DagGraphSpec graph;
    graph.graph_id = graph_id;

    std::unordered_set<uint16_t> task_ids_in_graph;
    for(const auto& task_id_json : graph_json.at("task_list")) {
      if(!task_id_json.is_number_integer()) {
        throw std::runtime_error("Graph " + std::to_string(graph_id) + " task_list entries must be integers.");
      }

      const auto signed_raw = task_id_json.get<int64_t>();
      if(signed_raw < 0) {
        throw std::runtime_error("Graph " + std::to_string(graph_id) + " task_list entries must be non-negative.");
      }
      const auto raw = static_cast<uint64_t>(signed_raw);
      if(raw > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("Graph " + std::to_string(graph_id) + " task_list entry exceeds uint16_t range.");
      }
      const auto task_id = static_cast<uint16_t>(raw);

      if(!task_ids_in_graph.insert(task_id).second) {
        throw std::runtime_error(
            "Graph " + std::to_string(graph_id) + " has duplicate task_id in task_list: " + std::to_string(task_id));
      }

      const auto catalog_it = task_catalog.find(task_id);
      if(catalog_it == task_catalog.end()) {
        throw std::runtime_error(
            "Graph " + std::to_string(graph_id) + " references unknown task_id in task_list: " + std::to_string(task_id));
      }

      graph.tasks.emplace(task_id, catalog_it->second);
    }

    for(auto& [_, task] : graph.tasks) {
      task.upstream.clear();
      task.downstream.clear();
    }

    if(graph_json.contains("upstream_by_task")) {
      if(!graph_json.at("upstream_by_task").is_array()) {
        throw std::runtime_error(
            "Graph " + std::to_string(graph_id) + " field 'upstream_by_task' must be an array.");
      }

      for(const auto& upstream_entry : graph_json.at("upstream_by_task")) {
        if(!upstream_entry.is_object()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " upstream_by_task entries must be objects.");
        }
        const auto task_id = parse_u16(upstream_entry, "task_id", "upstream_by_task entry");
        if(!upstream_entry.contains("from") || !upstream_entry.at("from").is_array()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " upstream_by_task entry for task " +
              std::to_string(task_id) + " must contain array field 'from'.");
        }

        auto task_it = graph.tasks.find(task_id);
        if(task_it == graph.tasks.end()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " upstream_by_task references task not in task_list: " +
              std::to_string(task_id));
        }

        std::unordered_set<uint16_t> seen_upstream;
        for(const auto& upstream_id_json : upstream_entry.at("from")) {
          if(!upstream_id_json.is_number_integer()) {
            throw std::runtime_error(
                "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
                " from entries must be integers.");
          }

          const auto signed_raw = upstream_id_json.get<int64_t>();
          if(signed_raw < 0) {
            throw std::runtime_error(
                "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
                " from entries must be non-negative.");
          }

          const auto raw = static_cast<uint64_t>(signed_raw);
          if(raw > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error(
                "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
                " from entry exceeds uint16_t range.");
          }

          const auto upstream_id = static_cast<uint16_t>(raw);
          if(seen_upstream.insert(upstream_id).second) {
            task_it->second.upstream.push_back(upstream_id);
          }
        }
      }
    } else if(graph_json.contains("edges")) {
      if(!graph_json.at("edges").is_object()) {
        throw std::runtime_error("Graph " + std::to_string(graph_id) + " field 'edges' must be an object.");
      }

      for(auto it = graph_json.at("edges").begin(); it != graph_json.at("edges").end(); ++it) {
        const auto task_id_str = it.key();
        const auto task_id_long = std::stol(task_id_str);
        if(task_id_long < 0 || task_id_long > std::numeric_limits<uint16_t>::max()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " edges contains invalid task id key: " + task_id_str);
        }
        const auto task_id = static_cast<uint16_t>(task_id_long);

        auto task_it = graph.tasks.find(task_id);
        if(task_it == graph.tasks.end()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " edges references task not in task_list: " +
              std::to_string(task_id));
        }

        if(!it.value().is_array()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " edges entry for task " + std::to_string(task_id) +
              " must be an array.");
        }

        std::unordered_set<uint16_t> seen_upstream;
        for(const auto& upstream_id_json : it.value()) {
          if(!upstream_id_json.is_number_integer()) {
            throw std::runtime_error(
                "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
                " edges entries must be integers.");
          }
          const auto signed_raw = upstream_id_json.get<int64_t>();
          if(signed_raw < 0) {
            throw std::runtime_error(
                "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
                " edges entries must be non-negative.");
          }
          const auto raw = static_cast<uint64_t>(signed_raw);
          if(raw > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error(
                "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
                " edges entry exceeds uint16_t range.");
          }
          const auto upstream_id = static_cast<uint16_t>(raw);
          if(seen_upstream.insert(upstream_id).second) {
            task_it->second.upstream.push_back(upstream_id);
          }
        }
      }
    }

    for(auto& [task_id, task] : graph.tasks) {
      for(const auto upstream_task_id : task.upstream) {
        auto upstream_it = graph.tasks.find(upstream_task_id);
        if(upstream_it == graph.tasks.end()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
              " references unknown upstream task " + std::to_string(upstream_task_id));
        }

        auto& downstream = upstream_it->second.downstream;
        if(std::find(downstream.begin(), downstream.end(), task_id) == downstream.end()) {
          downstream.push_back(task_id);
        }
      }
    }

    if(!udl_uuid.empty()) {
      bool has_udl = false;
      for(const auto& [_, task] : graph.tasks) {
        if(task.udl_uuid == udl_uuid) {
          has_udl = true;
          break;
        }
      }
      if(!has_udl) {
        continue;
      }
    }

    auto [_, graph_inserted] = registry.graphs_.emplace(graph_id, std::move(graph));
    if(!graph_inserted) {
      throw std::runtime_error("Duplicate graph_id in jobs config: " + std::to_string(graph_id));
    }
  }

  if(!udl_uuid.empty() && registry.graphs_.empty()) {
    throw std::runtime_error("No graph matched the requested UDL uuid.");
  }

  registry.finalize_graphs();
  return registry;
}

const DagNodeSpec* DagRegistry::find_task(uint16_t graph_id, uint16_t task_id) const {
  const auto graph_it = graphs_.find(graph_id);
  if(graph_it == graphs_.end()) {
    return nullptr;
  }

  const auto task_it = graph_it->second.tasks.find(task_id);
  if(task_it == graph_it->second.tasks.end()) {
    return nullptr;
  }

  return &task_it->second;
}

const DagGraphSpec* DagRegistry::find_graph(uint16_t graph_id) const {
  const auto it = graphs_.find(graph_id);
  if(it == graphs_.end()) {
    return nullptr;
  }

  return &it->second;
}

std::size_t DagRegistry::graph_count() const {
  return graphs_.size();
}

const std::vector<uint16_t>* DagRegistry::find_upstream(uint16_t graph_id,
                                                         uint16_t task_id) const {
  const auto* task = find_task(graph_id, task_id);
  if(!task) {
    return nullptr;
  }

  return &task->upstream;
}

std::optional<uint16_t> DagRegistry::find_task_id_by_path(
    const std::string& pathname) const {
  bool found = false;
  uint16_t found_task_id = 0;

  for(const auto& [_, graph] : graphs_) {
    for(const auto& [task_id, task] : graph.tasks) {
      if(task.pathname != pathname) {
        continue;
      }

      if(found && found_task_id != task_id) {
        throw std::runtime_error(
            "Ambiguous pathname appears with different task_ids across graphs: " + pathname);
      }

      found = true;
      found_task_id = task_id;
    }
  }

  if(!found) {
    return std::nullopt;
  }
  return found_task_id;
}

void DagRegistry::print_registry_debug(std::ostream& out) const {
  out << "[EXAMPLE ocdpo]: DAG registry dump begin" << std::endl;

  std::vector<uint16_t> graph_ids;
  graph_ids.reserve(graphs_.size());
  for(const auto& [graph_id, _] : graphs_) {
    graph_ids.push_back(graph_id);
  }
  std::sort(graph_ids.begin(), graph_ids.end());

  for(const auto graph_id : graph_ids) {
    const auto* graph = find_graph(graph_id);
    if(!graph) {
      continue;
    }

    out << "  graph_id=" << graph_id << " task_count=" << graph->tasks.size() << std::endl;

    std::vector<uint16_t> task_ids;
    task_ids.reserve(graph->tasks.size());
    for(const auto& [task_id, _] : graph->tasks) {
      task_ids.push_back(task_id);
    }
    std::sort(task_ids.begin(), task_ids.end());

    for(const auto task_id : task_ids) {
      const auto* task = find_task(graph_id, task_id);
      if(!task) {
        continue;
      }

      out << "    task_id=" << task->task_id;
      out << " pathname=" << task->pathname;
      out << " udl_uuid=" << task->udl_uuid;

      out << " upstream=[";
      for(std::size_t i = 0; i < task->upstream.size(); ++i) {
        out << task->upstream[i];
        if(i + 1 < task->upstream.size()) {
          out << ",";
        }
      }
      out << "]";

      out << " downstream=[";
      for(std::size_t i = 0; i < task->downstream.size(); ++i) {
        out << task->downstream[i];
        if(i + 1 < task->downstream.size()) {
          out << ",";
        }
      }
      out << "]" << std::endl;
    }
  }

  out << "[EXAMPLE ocdpo]: DAG registry dump end" << std::endl;
}

nlohmann::json DagRegistry::load_json_file(const std::string& path) {
  std::ifstream in(path);
  if(!in.is_open()) {
    throw std::runtime_error("unable to open json file: " + path);
  }

  nlohmann::json parsed;
  in >> parsed;
  return parsed;
}

uint16_t DagRegistry::parse_u16(const nlohmann::json& json,
                                const char* field,
                                const char* scope) {
  if(!json.contains(field) || !json.at(field).is_number_integer()) {
    throw std::runtime_error(std::string(scope) + " must contain integer field '" + field + "'.");
  }

  const auto signed_raw = json.at(field).get<int64_t>();
  if(signed_raw < 0) {
    throw std::runtime_error(std::string(scope) + " field '" + field + "' must be non-negative.");
  }

  const auto raw = static_cast<uint64_t>(signed_raw);
  if(raw > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error(std::string(scope) + " field '" + field + "' exceeds uint16_t range.");
  }

  return static_cast<uint16_t>(raw);
}

void DagRegistry::finalize_graphs() {
  for(auto& [graph_id, graph] : graphs_) {
    for(const auto& [task_id, task] : graph.tasks) {
      for(const auto downstream_task_id : task.downstream) {
        if(graph.tasks.find(downstream_task_id) == graph.tasks.end()) {
          throw std::runtime_error(
              "Graph " + std::to_string(graph_id) + " task " + std::to_string(task_id) +
              " references unknown downstream task " + std::to_string(downstream_task_id));
        }
      }
    }
  }
}

VORTEX_SCHEDULER_NAMESPACE_END
