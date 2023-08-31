// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_manager/cvnode_manager.hpp>
#include <kenning_computer_vision_msgs/msg/runtime_msg_type.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>

namespace cvnode_manager
{
using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using RuntimeProtocolSrv = kenning_computer_vision_msgs::srv::RuntimeProtocolSrv;
using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;

using RuntimeMsgType = kenning_computer_vision_msgs::msg::RuntimeMsgType;
using SegmentationMsg = kenning_computer_vision_msgs::msg::SegmentationMsg;

CVNodeManager::CVNodeManager(const rclcpp::NodeOptions &options) : Node("cvnode_manager", options)
{
    manage_service = create_service<ManageCVNode>(
        "node_manager/register",
        std::bind(&CVNodeManager::manage_node_callback, this, std::placeholders::_1, std::placeholders::_2));

    dataprovider_service = create_service<RuntimeProtocolSrv>(
        "node_manager/dataprovider",
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
    descriptor.description = "Preserve last output";
    descriptor.additional_constraints = "Must be a boolean value";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    declare_parameter("preserve_output", true, descriptor);

    // Publishers for input and output data
    input_publisher = create_publisher<sensor_msgs::msg::Image>("node_manager/input_frame", 1);
    output_publisher = create_publisher<SegmentationMsg>("node_manager/output_segmentations", 1);
}

void CVNodeManager::dataprovider_callback(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
    std::string data;
    switch (request->message_type)
    {
    case RuntimeMsgType::OK:
        if (!dataprovider_initialized)
        {
            initialize_dataprovider(header);
        }
        else
        {
            abort(header, "[OK] Received unexpected 'OK' message.");
        }
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
        inference_request->message_type = RuntimeMsgType::MODEL;
        async_broadcast_request(header, inference_request);
        break;
    case RuntimeMsgType::DATA:
        forward_data_request(header, request);
        break;
    case RuntimeMsgType::STATS:
        data = measurements.dump();
        response.data = std::vector<uint8_t>(data.begin(), data.end());
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
        break;
    case RuntimeMsgType::OUTPUT:
        forward_output_request(header);
        break;
    case RuntimeMsgType::PROCESS:
        if (inference_scenario_func != nullptr)
        {
            std::thread(inference_scenario_func, header).detach();
        }
        else
        {
            abort(header, "[PROCESS] Inference scenario function is not initialized");
        }
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
    if (!set_scenario())
    {
        abort(header, "[OK] Error while setting scenario.");
        return;
    }

    output_data = nlohmann::json();
    output_data.emplace("output", nlohmann::json::array());

    measurements = nlohmann::json();
    measurements.emplace("target_inference_step", nlohmann::json::array());
    measurements.emplace("target_inference_step_timestamp", nlohmann::json::array());

    if (std::get<1>(cv_node) == nullptr)
    {
        std::thread(
            [this, header]()
            {
                std::unique_lock<std::mutex> lk(dataprovider_mutex);
                RCLCPP_DEBUG(get_logger(), "[DATAPROVIDER] Waiting for inference to start");
                dataprovider_cv.wait(lk);
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

void CVNodeManager::forward_data_request(
    const std::shared_ptr<rmw_request_id_t> header,
    const RuntimeProtocolSrv::Request::SharedPtr request)
{
    SegmentCVNodeSrv::Request::SharedPtr inference_request;
    bool publish = get_parameter("publish_visualizations").as_bool();

    if (publish || !cv_node_future.valid())
    {
        inference_request = extract_images(request->data);
    }

    if (publish)
    {
        for (auto &image : inference_request->input)
        {
            input_publisher->publish(image);
        }
    }

    if (!cv_node_future.valid())
    {
        if (inference_request->message_type == RuntimeMsgType::ERROR)
        {
            abort(header, "[DATA] Error while extracting data.");
        }
        async_broadcast_request(header, inference_request);
    }
    else
    {
        RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
    }
}

void CVNodeManager::forward_output_request(const std::shared_ptr<rmw_request_id_t> header)
{
    if (!cv_node_future.valid())
    {
        SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
        inference_request->message_type = RuntimeMsgType::OUTPUT;
        async_broadcast_request(
            header,
            inference_request,
            std::bind(&CVNodeManager::segmentations_to_output_data, this, std::placeholders::_1));
    }
    else
    {
        RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
        if (!get_parameter("preserve_output").as_bool())
        {
            output_data = nlohmann::json();
            output_data.emplace("output", nlohmann::json::array());
        }
        std::string output_string = output_data.dump();

        response.data = std::vector<uint8_t>(output_string.begin(), output_string.end());
        response.message_type = RuntimeMsgType::OK;
        dataprovider_service->send_response(*header, response);
    }
}

bool CVNodeManager::set_scenario()
{
    std::string scenario = get_parameter("scenario").as_string();
    if (scenario == "synthetic")
    {
        RCLCPP_DEBUG(get_logger(), "[SCENARIO] Set to 'Synthetic'");
        inference_scenario_func = std::bind(&CVNodeManager::synthetic_inference_scenario, this, std::placeholders::_1);
    }
    else if (scenario == "real_world_last")
    {
        RCLCPP_DEBUG(get_logger(), "[SCENARIO] Set to 'Real World Last'");
        inference_scenario_func =
            std::bind(&CVNodeManager::real_world_last_inference_scenario, this, std::placeholders::_1);
    }
    else if (scenario == "real_world_first")
    {
        RCLCPP_DEBUG(get_logger(), "[SCENARIO] Set to 'Real World First'");
        inference_scenario_func =
            std::bind(&CVNodeManager::real_world_first_inference_scenario, this, std::placeholders::_1);
    }
    else
    {
        RCLCPP_ERROR(get_logger(), "Unknown scenario: %s", scenario.c_str());
        return false;
    }
    return true;
}

void CVNodeManager::synthetic_inference_scenario(const std::shared_ptr<rmw_request_id_t> header)
{
    using namespace std::chrono;

    SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
    inference_request->message_type = RuntimeMsgType::PROCESS;

    /// Send inference request
    start = steady_clock::now();
    cv_node_future = std::get<1>(cv_node)->async_send_request(inference_request).future.share();
    cv_node_future.wait();
    SegmentCVNodeSrv::Response::SharedPtr inference_response = cv_node_future.get();

    // Measure inference time
    end = steady_clock::now();
    float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    measurements.at("target_inference_step").push_back(duration);
    measurements.at("target_inference_step_timestamp")
        .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

    // Reset future
    cv_node_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();
    if (inference_response->message_type != RuntimeMsgType::OK)
    {
        abort(header, "[SYNTHETIC] Error while processing data.");
        return;
    }
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

void CVNodeManager::real_world_last_inference_scenario(const std::shared_ptr<rmw_request_id_t> header)
{
    using namespace std::chrono;

    SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
    inference_request->message_type = RuntimeMsgType::PROCESS;

    /// Send inference request
    start = steady_clock::now();
    cv_node_future = std::get<1>(cv_node)->async_send_request(inference_request).future.share();
    if (cv_node_future.wait_for(milliseconds(get_parameter("inference_timeout_ms").as_int())) ==
        std::future_status::ready)
    {
        SegmentCVNodeSrv::Response::SharedPtr inference_response = cv_node_future.get();
        if (inference_response->message_type != RuntimeMsgType::OK)
        {
            abort(header, "[REAL WORLD LAST] Error while processing data.");
            return;
        }
    }

    // Measure inference time
    end = steady_clock::now();
    float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    measurements.at("target_inference_step").push_back(duration);
    measurements.at("target_inference_step_timestamp")
        .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

    // Reset future to make sure that it is not reused
    cv_node_future = std::future<SegmentCVNodeSrv::Response::SharedPtr>();

    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

void CVNodeManager::real_world_first_inference_scenario(const std::shared_ptr<rmw_request_id_t> header)
{
    using namespace std::chrono;

    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    if (!cv_node_future.valid())
    {
        SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
        inference_request->message_type = RuntimeMsgType::PROCESS;
        start = steady_clock::now();
        cv_node_future = std::get<1>(cv_node)->async_send_request(inference_request).future.share();
    }
    if (cv_node_future.wait_for(milliseconds(get_parameter("inference_timeout_ms").as_int())) ==
        std::future_status::ready)
    {
        SegmentCVNodeSrv::Response::SharedPtr inference_response = cv_node_future.get();
        end = steady_clock::now();
        float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
        measurements.at("target_inference_step").push_back(duration);
        measurements.at("target_inference_step_timestamp")
            .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);
        if (inference_response->message_type != RuntimeMsgType::OK)
        {
            abort(header, "[REAL WORLD FIRST] Error while processing data.");
            return;
        }
        cv_node_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();
    }
    response.message_type = RuntimeMsgType::OK;
    dataprovider_service->send_response(*header, response);
}

RuntimeProtocolSrv::Response
CVNodeManager::segmentations_to_output_data(const SegmentCVNodeSrv::Response::SharedPtr response)
{
    bool publish = get_parameter("publish_visualizations").as_bool();
    RuntimeProtocolSrv::Response runtime_response = RuntimeProtocolSrv::Response();
    if (!response->output.empty() || get_parameter("scenario").as_string() == "synthetic")
    {
        output_data = nlohmann::json();
        output_data.emplace("output", nlohmann::json::array());

        for (auto &segmentation : response->output)
        {
            if (publish)
            {
                output_publisher->publish(segmentation);
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

            try
            {
                output_data.at("output").push_back(json);
            }
            catch (const nlohmann::json::exception &e)
            {
                RCLCPP_ERROR(get_logger(), "Error while parsing output json: %s", e.what());
                runtime_response.message_type = RuntimeMsgType::ERROR;
                return runtime_response;
            }
        }
    }
    else if (!get_parameter("preserve_output").as_bool())
    {
        output_data = nlohmann::json();
        output_data.emplace("output", nlohmann::json::array());
    }
    std::string json_string = output_data.dump();

    runtime_response.message_type = RuntimeMsgType::OK;
    runtime_response.data = std::vector<uint8_t>(json_string.begin(), json_string.end());
    return runtime_response;
}

void CVNodeManager::async_broadcast_request(
    const std::shared_ptr<rmw_request_id_t> header,
    const SegmentCVNodeSrv::Request::SharedPtr request,
    std::function<RuntimeProtocolSrv::Response(const SegmentCVNodeSrv::Response::SharedPtr)> callback)
{
    if (std::get<1>(cv_node) == nullptr)
    {
        RCLCPP_ERROR(get_logger(), "No CVNode registered");
        RuntimeProtocolSrv::Response response;
        response.message_type = RuntimeMsgType::ERROR;
        dataprovider_service->send_response(*header, response);
    }

    std::get<1>(cv_node)->async_send_request(
        request,
        [this, header, callback](rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            auto response = future.get();
            if (response->message_type != RuntimeMsgType::OK)
            {
                abort(header, "[BROADCAST] Failed to receive confirmation from the CVNode");
                return;
            }

            RuntimeProtocolSrv::Response dataprovider_response;
            if (callback != nullptr)
            {
                RCLCPP_DEBUG(get_logger(), "[BROADCAST] Executing callback.");
                dataprovider_response = callback(response);
                if (dataprovider_response.message_type != RuntimeMsgType::OK)
                {
                    abort(header, "[BROADCAST] Failed to process response from the CVNode");
                    return;
                }
            }
            dataprovider_response.message_type = RuntimeMsgType::OK;
            dataprovider_service->send_response(*header, dataprovider_response);
        });
}

// NOTE: This method should be reworked to use more precisely described data format
SegmentCVNodeSrv::Request::SharedPtr CVNodeManager::extract_images(std::vector<uint8_t> &input_data_b)
{
    SegmentCVNodeSrv::Request::SharedPtr request = std::make_shared<SegmentCVNodeSrv::Request>();

    try
    {
        nlohmann::json data_j = nlohmann::json::parse(std::string(input_data_b.begin(), input_data_b.end()));
        std::vector<nlohmann::json> data = data_j.at("data");
        if (data.size() == 0)
        {
            RCLCPP_ERROR(get_logger(), "[EXTRACT IMAGES] Data is empty");
            request->message_type = RuntimeMsgType::ERROR;
            return request;
        }
        for (auto &image : data)
        {
            sensor_msgs::msg::Image img;
            img.height = image.at("height");
            img.width = image.at("width");
            img.encoding = "bgr8";
            img.data = image.at("data").get<std::vector<uint8_t>>();
            img.step = img.width * 3;
            request->input.push_back(img);
        }
    }
    catch (nlohmann::json::exception &e)
    {
        RCLCPP_ERROR(get_logger(), "[EXTRACT IMAGES] Error while parsing data json: %s", e.what());
        request->message_type = RuntimeMsgType::ERROR;
        return request;
    }
    request->message_type = RuntimeMsgType::DATA;
    return request;
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
    RCLCPP_DEBUG(get_logger(), "Registering the '%s' node", node_name.c_str());

    if (std::get<1>(cv_node) != nullptr)
    {
        response->message = "There is already a node registered";
        RCLCPP_WARN(
            get_logger(),
            "Could not register the node '%s' because there is already a node registered",
            node_name.c_str());
        return;
    }

    // Create a client to communicate with the node
    rclcpp::Client<SegmentCVNodeSrv>::SharedPtr cv_node_service;
    if (!initialize_service_client<SegmentCVNodeSrv>(request->srv_name, cv_node_service))
    {
        response->message = "Could not initialize the communication service client";
        RCLCPP_ERROR(get_logger(), "Could not initialize the communication service client");
        return;
    }

    response->status = true;
    response->message = "The node is registered";
    cv_node = std::make_tuple(node_name, cv_node_service);

    if (dataprovider_initialized)
    {
        dataprovider_cv.notify_one();
    }
    RCLCPP_DEBUG(get_logger(), "The node '%s' is registered", node_name.c_str());
}

void CVNodeManager::unregister_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    [[maybe_unused]] ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;

    RCLCPP_DEBUG(get_logger(), "Unregistering the node '%s'", node_name.c_str());
    if (std::get<0>(cv_node) != node_name)
    {
        RCLCPP_WARN(get_logger(), "The node '%s' is not registered", node_name.c_str());
        return;
    }

    cv_node = std::make_tuple("", nullptr);
    RCLCPP_DEBUG(get_logger(), "The node '%s' is unregistered", node_name.c_str());
    return;
}

void CVNodeManager::abort(const std::shared_ptr<rmw_request_id_t> header, const std::string &error_msg)
{
    RCLCPP_ERROR(get_logger(), "%s", error_msg.c_str());
    SegmentCVNodeSrv::Request::SharedPtr request = std::make_shared<SegmentCVNodeSrv::Request>();
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    request->message_type = RuntimeMsgType::ERROR;
    response.message_type = RuntimeMsgType::ERROR;
    std::get<1>(cv_node)->async_send_request(request);
    dataprovider_service->send_response(*header, response);
}

} // namespace cvnode_manager

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
