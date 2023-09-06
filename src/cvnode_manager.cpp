// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_manager/cvnode_manager.hpp>
#include <kenning_computer_vision_msgs/msg/runtime_msg_type.hpp>
#include <mutex>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>

namespace cvnode_manager
{

using RuntimeMsgType = kenning_computer_vision_msgs::msg::RuntimeMsgType;

CVNodeManager::CVNodeManager(const rclcpp::NodeOptions &options) : Node("cvnode_manager", options)
{
    manage_service = create_service<ManageCVNode>(
        "cvnode_register",
        std::bind(&CVNodeManager::manage_node_callback, this, std::placeholders::_1, std::placeholders::_2));

    dataprovider_service = create_service<RuntimeProtocolSrv>(
        "dataprovider",
        std::bind(&CVNodeManager::dataprovider_callback, this, std::placeholders::_1, std::placeholders::_2));

    rcl_interfaces::msg::ParameterDescriptor descriptor;

    // Parameter bool to publish visualizations
    descriptor.description = "Publishes input and output data to dedicated topics if set to true";
    descriptor.additional_constraints = "Must be a boolean value";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    declare_parameter("publish_visualizations", false, descriptor);

    // Parameter int to set the timeout for the real-world scenarios
    descriptor.description = "Timeout for the real-world scenarios in milliseconds";
    descriptor.additional_constraints = "Must be an integer value greater than 0";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
    declare_parameter("inference_timeout_ms", 1000, descriptor);

    // Parameter string to choose testing strategy
    descriptor.description = "Testing scenario to run";
    descriptor.additional_constraints = "Must be one of: real_world_last, real_world_first, synthetic";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
    declare_parameter("scenario", "synthetic", descriptor);

    // Parameter bool to preserve last output
    descriptor.description = "Preserves last frame output if timeout is reached during inference";
    descriptor.additional_constraints = "Must be a boolean value";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    declare_parameter("preserve_output", true, descriptor);

    // Publishers for input and output data
    gui_input_publisher = create_publisher<sensor_msgs::msg::Image>("input_frame", 1);
    gui_output_publisher = create_publisher<SegmentationMsg>("output_segmentations", 1);
}

void CVNodeManager::dataprovider_callback(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    std_srvs::srv::Trigger::Request::SharedPtr trigger_request = std::make_shared<std_srvs::srv::Trigger::Request>();
    std::string data;
    switch (request->message_type)
    {
    case RuntimeMsgType::OK:
        if (dataprovider_initialized)
        {
            abort(header, "[OK] Received unexpected 'OK' message.");
            break;
        }
        initialize_dataprovider(header);
        break;
    case RuntimeMsgType::ERROR:
        abort(header, "[ERROR] Received error message from the dataprovider.");
        break;
    case RuntimeMsgType::IOSPEC:
        // IGNORE
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
        break;
    case RuntimeMsgType::MODEL:
        cv_node.prepare->async_send_request(
            trigger_request,
            [this, header](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future)
            {
                auto response = future.get();
                if (!response->success)
                {
                    abort(header, "[MODEL] Error while loading the model.");
                    return;
                }
                RCLCPP_DEBUG(get_logger(), "CVNode prepared successfully.");
                RuntimeProtocolSrv::Response runtime_response = RuntimeProtocolSrv::Response();
                runtime_response.message_type = RuntimeMsgType::OK;
                dataprovider_service->send_response(*header, runtime_response);
                return;
            });
        break;
    case RuntimeMsgType::DATA:
        extract_data_from_request(header, request);
        break;
    case RuntimeMsgType::PROCESS:
        if (inference_scenario_func == nullptr)
        {
            abort(header, "[PROCESS] Inference scenario function is not initialized");
            break;
        }
        if (!get_parameter("preserve_output").as_bool())
        {
            output_data = nlohmann::json();
            output_data.emplace("output", nlohmann::json::array());
        }
        std::thread(inference_scenario_func, header).detach();
        break;
    case RuntimeMsgType::OUTPUT:
        data = output_data.dump();
        response.data = std::vector<uint8_t>(data.begin(), data.end());
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
        break;
    case RuntimeMsgType::STATS:
        data = measurements.dump();
        response.data = std::vector<uint8_t>(data.begin(), data.end());
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
        cv_node.cleanup->async_send_request(
            trigger_request,
            [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future)
            {
                auto response = future.get();
                if (!response->success)
                {
                    RCLCPP_ERROR(get_logger(), "Error while cleaning up the node.");
                }
            });
        break;
    default:
        data = "[UNKNOWN] Received unknown message type: " + std::to_string(request->message_type);
        abort(header, data);
        break;
    }
}

void CVNodeManager::initialize_dataprovider(const std::shared_ptr<rmw_request_id_t> header)
{
    dataprovider_initialized = true;
    if (!configure_scenario())
    {
        abort(header, "[OK] Error while setting scenario.");
        return;
    }

    output_data = nlohmann::json();
    output_data.emplace("output", nlohmann::json::array());

    measurements = nlohmann::json();
    measurements.emplace("target_inference_step", nlohmann::json::array());
    measurements.emplace("target_inference_step_timestamp", nlohmann::json::array());

    if (cv_node.process == nullptr)
    {
        std::thread(
            [this, header]()
            {
                std::mutex dataprovider_mutex;
                std::unique_lock<std::mutex> lk(dataprovider_mutex);
                RCLCPP_DEBUG(get_logger(), "[DATAPROVIDER] Waiting for inference to start");
                cvnode_wait_cv.wait(lk);
                RCLCPP_DEBUG(get_logger(), "[DATAPROVIDER] Starting inference");
                RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
                response.message_type = RuntimeMsgType::OK;
                dataprovider_service->send_response(*header, response);
            })
            .detach();
    }
    else
    {
        RCLCPP_DEBUG(get_logger(), "[DATAPROVIDER] Starting inference");
        RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
    }
}

// NOTE: This method should be reworked to use more precisely described data format
void CVNodeManager::extract_data_from_request(
    const std::shared_ptr<rmw_request_id_t> header,
    const RuntimeProtocolSrv::Request::SharedPtr request)
{
    bool publish = get_parameter("publish_visualizations").as_bool();
    try
    {
        nlohmann::json data_j = nlohmann::json::parse(std::string(request->data.begin(), request->data.end()));
        std::vector<nlohmann::json> data = data_j.at("data");
        if (data.size() == 0)
        {
            abort(header, "[DATA] Error while extracting data.");
            return;
        }
        cvnode_request->input.clear();
        for (auto &image : data)
        {
            sensor_msgs::msg::Image img;
            img.height = image.at("height");
            img.width = image.at("width");
            img.encoding = "bgr8";
            img.data = image.at("data").get<std::vector<uint8_t>>();
            img.step = img.width * 3;
            cvnode_request->input.push_back(img);
            if (publish)
            {
                gui_input_publisher->publish(img);
            }
        }
    }
    catch (nlohmann::json::exception &e)
    {
        cvnode_request->input.clear();
        abort(header, "[DATA] Error while extracting data.");
    }

    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

bool CVNodeManager::configure_scenario()
{
    std::string scenario = get_parameter("scenario").as_string();
    if (scenario == "synthetic")
    {
        RCLCPP_DEBUG(get_logger(), "[SCENARIO] Set to 'Synthetic'");
        inference_scenario_func =
            std::bind(&CVNodeManager::execute_synthetic_inference_scenario, this, std::placeholders::_1);
    }
    else if (scenario == "real_world_last")
    {
        RCLCPP_DEBUG(get_logger(), "[SCENARIO] Set to 'Real World Last'");
        inference_scenario_func =
            std::bind(&CVNodeManager::execute_real_world_last_inference_scenario, this, std::placeholders::_1);
    }
    else if (scenario == "real_world_first")
    {
        RCLCPP_DEBUG(get_logger(), "[SCENARIO] Set to 'Real World First'");
        inference_scenario_func =
            std::bind(&CVNodeManager::execute_real_world_first_inference_scenario, this, std::placeholders::_1);
    }
    else
    {
        RCLCPP_ERROR(get_logger(), "Unknown scenario: %s", scenario.c_str());
        return false;
    }
    return true;
}

void CVNodeManager::execute_synthetic_inference_scenario(const std::shared_ptr<rmw_request_id_t> header)
{
    using namespace std::chrono;

    if (cvnode_request->input.empty())
    {
        abort(header, "[PROCESS] No data received.");
        return;
    }

    start = steady_clock::now();
    cvnode_future = cv_node.process->async_send_request(cvnode_request).future.share();
    cvnode_future.wait();
    SegmentCVNodeSrv::Response::SharedPtr inference_response = cvnode_future.get();
    end = steady_clock::now();
    cvnode_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();
    if (!inference_response->success)
    {
        abort(header, "[SYNTHETIC] Error while processing data.");
        return;
    }

    output_data = segmentations_to_json(inference_response);

    // Save measurements
    float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    measurements.at("target_inference_step").push_back(duration);
    measurements.at("target_inference_step_timestamp")
        .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

    // Reset future
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

void CVNodeManager::execute_real_world_last_inference_scenario(const std::shared_ptr<rmw_request_id_t> header)
{
    using namespace std::chrono;

    if (cvnode_request->input.empty())
    {
        abort(header, "[PROCESS] No data received.");
        return;
    }

    /// Send inference request
    start = steady_clock::now();
    cvnode_future = cv_node.process->async_send_request(cvnode_request).future.share();
    if (cvnode_future.wait_for(milliseconds(get_parameter("inference_timeout_ms").as_int())) ==
        std::future_status::ready)
    {
        SegmentCVNodeSrv::Response::SharedPtr inference_response = cvnode_future.get();
        if (!inference_response->success)
        {
            abort(header, "[REAL WORLD LAST] Error while processing data.");
            return;
        }
        end = steady_clock::now();
        output_data = segmentations_to_json(inference_response);
    }
    else
    {
        end = steady_clock::now();
    }

    // Measure inference time
    float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    measurements.at("target_inference_step").push_back(duration);
    measurements.at("target_inference_step_timestamp")
        .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

    // Reset future to make sure that it is not reused
    cvnode_future = std::future<SegmentCVNodeSrv::Response::SharedPtr>();

    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

void CVNodeManager::execute_real_world_first_inference_scenario(const std::shared_ptr<rmw_request_id_t> header)
{
    using namespace std::chrono;

    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    if (!cvnode_future.valid())
    {
        if (cvnode_request->input.empty())
        {
            abort(header, "[PROCESS] No data received.");
            return;
        }
        start = steady_clock::now();
        cvnode_future = cv_node.process->async_send_request(cvnode_request).future.share();
    }
    if (cvnode_future.wait_for(milliseconds(get_parameter("inference_timeout_ms").as_int())) ==
        std::future_status::ready)
    {
        SegmentCVNodeSrv::Response::SharedPtr inference_response = cvnode_future.get();
        if (!inference_response->success)
        {
            abort(header, "[REAL WORLD FIRST] Error while processing data.");
            return;
        }
        end = steady_clock::now();
        cvnode_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();

        float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
        measurements.at("target_inference_step").push_back(duration);
        measurements.at("target_inference_step_timestamp")
            .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

        output_data = segmentations_to_json(inference_response);
    }
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

nlohmann::json CVNodeManager::segmentations_to_json(const SegmentCVNodeSrv::Response::SharedPtr response)
{
    bool publish = get_parameter("publish_visualizations").as_bool();
    nlohmann::json output_data_json = nlohmann::json();
    output_data_json.emplace("output", nlohmann::json::array());
    if (!response->output.empty())
    {
        for (auto &segmentation : response->output)
        {
            if (publish)
            {
                gui_output_publisher->publish(segmentation);
            }

            nlohmann::json json = nlohmann::json::array();
            for (size_t i = 0; i < segmentation.classes.size(); i++)
            {
                nlohmann::json object;
                nlohmann::json mask;
                nlohmann::json bbox;
                object.emplace("class", segmentation.classes[i]);
                object.emplace("score", segmentation.scores[i]);
                // NOTE: Attaching mask takes a lot of time (may take seconds, depending on amount and resolution)
                mask.emplace("dimensions", segmentation.masks[i].dimension);
                mask.emplace("data", segmentation.masks[i].data);
                object.emplace("mask", mask);
                bbox.emplace("xmin", segmentation.boxes[i].xmin);
                bbox.emplace("ymin", segmentation.boxes[i].ymin);
                bbox.emplace("xmax", segmentation.boxes[i].xmax);
                bbox.emplace("ymax", segmentation.boxes[i].ymax);
                object.emplace("box", bbox);
                json.push_back(object);
            }
            output_data_json.at("output").push_back(json);
        }
    }
    return output_data_json;
}

void CVNodeManager::manage_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    ManageCVNode::Response::SharedPtr response)
{
    if (request->type == request->REGISTER)
    {
        register_node_callback(request, response);
    }
    else if (request->type == request->UNREGISTER)
    {
        unregister_node_callback(request, response);
    }
    else
    {
        RCLCPP_ERROR(get_logger(), "Unknown type of the service");
        response->status = false;
        response->message = "Unknown type of the service";
    }
    return;
}

void CVNodeManager::register_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;
    response->status = false;
    RCLCPP_DEBUG(get_logger(), "[REGISTER] Registering the '%s' node", node_name.c_str());

    if (cv_node.process != nullptr)
    {
        response->message = "There is already a node registered";
        RCLCPP_WARN(
            get_logger(),
            "[REGISTER] Could not register the node '%s' because there is already a node registered",
            node_name.c_str());
        return;
    }

    // Create clients to communicate with the node
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cvnode_prepare;
    if (!initialize_service_client<std_srvs::srv::Trigger>(request->prepare_srv_name, cvnode_prepare))
    {
        response->message = "Could not initialize the prepare service client";
        RCLCPP_ERROR(get_logger(), "[REGISTER] Could not initialize the prepare service client");
        return;
    }
    rclcpp::Client<SegmentCVNodeSrv>::SharedPtr cvnode_process;
    if (!initialize_service_client<SegmentCVNodeSrv>(request->process_srv_name, cvnode_process))
    {
        response->message = "Could not initialize the process service client";
        RCLCPP_ERROR(get_logger(), "[REGISTER] Could not initialize the process service client");
        return;
    }
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cvnode_cleanup;
    if (!initialize_service_client<std_srvs::srv::Trigger>(request->cleanup_srv_name, cvnode_cleanup))
    {
        response->message = "Could not initialize the cleanup service client";
        RCLCPP_ERROR(get_logger(), "[REGISTER] Could not initialize the cleanup service client");
        return;
    }

    response->status = true;
    response->message = "The node is registered";
    cv_node = CVNode(node_name, cvnode_prepare, cvnode_process, cvnode_cleanup);

    if (dataprovider_initialized)
    {
        cvnode_wait_cv.notify_one();
    }
    RCLCPP_DEBUG(get_logger(), "[REGISTER] The node '%s' is registered", node_name.c_str());
}

void CVNodeManager::unregister_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    [[maybe_unused]] ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;

    RCLCPP_DEBUG(get_logger(), "[UNREGISTER] Unregistering the node '%s'", node_name.c_str());
    if (cv_node.name != node_name)
    {
        RCLCPP_WARN(get_logger(), "[UNREGISTER] The node '%s' is not registered", node_name.c_str());
        return;
    }

    cv_node = CVNode();
    RCLCPP_DEBUG(get_logger(), "[UNREGISTER] The node '%s' is unregistered", node_name.c_str());
    return;
}

void CVNodeManager::abort(const std::shared_ptr<rmw_request_id_t> header, const std::string &error_msg)
{
    RCLCPP_ERROR(get_logger(), "%s", error_msg.c_str());
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    response.message_type = RuntimeMsgType::ERROR;
    dataprovider_service->send_response(*header, response);
}

} // namespace cvnode_manager

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
