/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "openvino_model_base.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#ifdef MELO_DEBUG
// dump exectuation graph
#include "openvino/core/graph_util.hpp"
#endif

namespace melo {
// Constuctor
AbstractOpenvinoModel::AbstractOpenvinoModel(std::unique_ptr<ov::Core>& core_ptr,
                                             const std::filesystem::path& model_path,
                                             const std::string& device,
                                             const std::optional<ov::AnyMap> config) {
    assert(std::filesystem::exists(model_path) && "model_path does not exit!");
    _device = device;
    // Reduce CPU infer memory
    if (device.find("CPU") != std::string::npos) {
        core_ptr->set_property("CPU", {{"CPU_RUNTIME_CACHE_CAPACITY", "0"}});
        std::cout << "Set CPU_RUNTIME_CACHE_CAPACITY 0\n";
    }
    ov::AnyMap ov_config = config.has_value() ? config.value() : AbstractOpenvinoModel::set_ov_config(device);
    // Compiled OV model
    auto startTime = Time::now();
    _compiled_model = std::make_unique<ov::CompiledModel>(core_ptr->compile_model(model_path.string(), device, ov_config));
    auto compileTime = get_duration_ms_till_now(startTime);
    _infer_request = std::make_unique<ov::InferRequest>(_compiled_model->create_infer_request());
    std::cout << std::format("compile model {} on {} using {}ms.\n", model_path.string(), device, compileTime);
    get_ov_info(core_ptr, device);
#ifdef MELO_DEBUG
    // dump exectuation graph
    auto runtime_model = _compiled_model->get_runtime_model();
    ov::serialize(runtime_model, "exec_graph.xml");
#endif  // MELO_DEBUG
}

void AbstractOpenvinoModel::print_input_names() const {
    const std::vector<ov::Output<const ov::Node>>& inputs = _compiled_model->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto& item = inputs[i];
        auto iop_precision = ov::element::undefined;
        auto type_to_set = ov::element::undefined;
        std::string name;
        // Some tensors might have no names, get_any_name will throw exception in that case.
        // -iop option will not work for those tensors.
        name = item.get_any_name();
        std::cout << i << " " << name << std::endl;
        // iop_precision = getPrecision2(user_precisions_map.at(item.get_any_name()));
    }
}

}  // namespace melo
