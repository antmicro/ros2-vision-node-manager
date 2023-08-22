// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_manager/cvnode_manager.hpp>
#include <kenning_computer_vision_msgs/runtime_msg_type.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>

namespace cvnode_manager
{
using namespace kenning_computer_vision_msgs::runtime_message_type;

using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;
using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using RuntimeProtocolSrv = kenning_computer_vision_msgs::srv::RuntimeProtocolSrv;
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
    descriptor.additional_constraints = "Must be an integer value";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
    declare_parameter("inference_timeout_ms", 1000, descriptor);

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
    RCLCPP_DEBUG(get_logger(), "Received message (%d) from the dataprovider", request->request.message_type);
    switch (request->request.message_type)
    {
    case OK:
        if (!dataprovider_initialized)
        {
            RCLCPP_DEBUG(get_logger(), "Received DataProvider initialization request");
            dataprovider_initialized = true;
            if (std::get<1>(cv_node) == nullptr)
            {
                std::thread(
                    [this, header]()
                    {
                        std::unique_lock<std::mutex> lk(dataprovider_mutex);
                        RCLCPP_DEBUG(get_logger(), "Waiting for inference to start");
                        dataprovider_cv.wait(lk);
                        RCLCPP_DEBUG(get_logger(), "Starting inference");
                        RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
                        response.response.message_type = OK;
                        dataprovider_service->send_response(*header, response);
                    })
                    .detach();
            }
            else
            {
                RCLCPP_DEBUG(get_logger(), "Starting inference");
                response.response.message_type = OK;
                dataprovider_service->send_response(*header, response);
            }
        }
        else
        {
            RCLCPP_DEBUG(get_logger(), "Received unexpected 'OK' message. Responding with error.");
            response.response.message_type = ERROR;
        }
        dataprovider_service->send_response(*header, response);
        break;
    case ERROR:
        RCLCPP_ERROR(get_logger(), "Received error message from the dataprovider");
        abort();
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    case IOSPEC:
        // IGNORE
        response.response.message_type = OK;
        dataprovider_service->send_response(*header, response);
        break;
    case MODEL:
        inference_request->message_type = MODEL;
        async_broadcast_request(header, inference_request);
        break;
    case DATA:
        inference_request = extract_images(request->request.data);
        if (inference_request->message_type == ERROR)
        {
            RCLCPP_DEBUG(get_logger(), "Error while extracting data. Aborting.");
            abort();
            response.response.message_type = ERROR;
            dataprovider_service->send_response(*header, response);
        }
        if (get_parameter("publish_visualizations").as_bool())
        {
            for (auto &image : inference_request->input)
            {
                input_publisher->publish(image);
            }
        }
        if (!cv_node_future.valid())
        {
            async_broadcast_request(header, inference_request);
        }
        else
        {
            response.response.message_type = OK;
            dataprovider_service->send_response(*header, response);
        }
        break;
    case STATS:
        // NYI: Implement stats collecting
        RCLCPP_ERROR(get_logger(), "Not yet implemented. Aborting.");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    case OUTPUT:
        output_json->clear();
        output_json->emplace("output", nlohmann::json::array());
        if (!cv_node_future.valid())
        {
            inference_request->message_type = OUTPUT;
            async_broadcast_request(
                header,
                inference_request,
                [this](SegmentCVNodeSrv::Response::SharedPtr response) -> RuntimeProtocolSrv::Response
                {
                    if (get_parameter("publish_visualizations").as_bool())
                    {
                        for (auto &segmentation : response->output)
                        {
                            output_publisher->publish(segmentation);
                        }
                    }
                    RuntimeProtocolSrv::Response runtime_response = RuntimeProtocolSrv::Response();
                    runtime_response.response.message_type = OK;
                    for (auto &segmentation : response->output)
                    {
                        try
                        {
                            output_json->at("output").push_back(segmentation_to_json(segmentation));
                        }
                        catch (nlohmann::json::exception &e)
                        {
                            RCLCPP_ERROR(get_logger(), "Error while parsing output json: %s", e.what());
                            runtime_response.response.message_type = ERROR;
                            return runtime_response;
                        }
                    }
                    std::string output_string = output_json->dump();
                    runtime_response.response.data = std::vector<uint8_t>(output_string.begin(), output_string.end());
                    return runtime_response;
                });
        }
        else
        {
            std::string output_string = output_json->dump();
            response.response.data = std::vector<uint8_t>(output_string.begin(), output_string.end());
            response.response.message_type = OK;
            dataprovider_service->send_response(*header, response);
        }
        break;
    default:
        if (inference_scenario_func != nullptr)
        {
            inference_scenario_func(header, request);
        }
        else
        {
            RCLCPP_ERROR(get_logger(), "Inference scenario function is not initialized");
            abort();
            response.response.message_type = ERROR;
            dataprovider_service->send_response(*header, response);
        }
        break;
    }
}

void CVNodeManager::synthetic_inference_scenario(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
    switch (request->request.message_type)
    {
    case PROCESS:
        inference_request->message_type = PROCESS;
        async_broadcast_request(header, inference_request);
        break;
    default:
        RCLCPP_ERROR(get_logger(), "Unsupported message type. Aborting.");
        abort();
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    }
}

void CVNodeManager::real_world_last_inference_scenario(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    switch (request->request.message_type)
    {
    case PROCESS:
        std::thread(
            [this, header]()
            {
                RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
                SegmentCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<SegmentCVNodeSrv::Request>();
                inference_request->message_type = PROCESS;
                cv_node_future = std::get<1>(cv_node)->async_send_request(inference_request);
                if (cv_node_future.wait_for(std::chrono::milliseconds(
                        get_parameter("inference_timeout_ms").as_int())) == std::future_status::ready)
                {
                    SegmentCVNodeSrv::Response::SharedPtr inference_response = cv_node_future.get();
                    if (inference_response->message_type != OK)
                    {
                        RCLCPP_ERROR(get_logger(), "Error while processing data. Aborting.");
                        abort();
                        response.response.message_type = ERROR;
                        dataprovider_service->send_response(*header, response);
                        return;
                    }
                }
                // Reset future to make sure that it is not reused
                cv_node_future = std::future<SegmentCVNodeSrv::Response::SharedPtr>();
                response.response.message_type = OK;
                dataprovider_service->send_response(*header, response);
            })
            .detach();
        break;
    default:
        RCLCPP_ERROR(get_logger(), "Unsupported message type. Aborting.");
        abort();
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    }
}

void CVNodeManager::real_world_first_inference_scenario(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    switch (request->request.message_type)
    {
    case PROCESS:
        std::thread(
            [this, header]()
            {
                RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
                if (!cv_node_future.valid())
                {
                    SegmentCVNodeSrv::Request::SharedPtr inference_request =
                        std::make_shared<SegmentCVNodeSrv::Request>();
                    inference_request->message_type = PROCESS;
                    cv_node_future = std::get<1>(cv_node)->async_send_request(inference_request);
                }
                else
                {
                }
                if (cv_node_future.wait_for(std::chrono::milliseconds(
                        get_parameter("inference_timeout_ms").as_int())) == std::future_status::ready)
                {
                    SegmentCVNodeSrv::Response::SharedPtr inference_response = cv_node_future.get();
                    if (inference_response->message_type != OK)
                    {
                        RCLCPP_ERROR(get_logger(), "Error while processing data. Aborting.");
                        abort();
                        response.response.message_type = ERROR;
                        dataprovider_service->send_response(*header, response);
                        return;
                    }
                    cv_node_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();
                }
                response.response.message_type = OK;
                dataprovider_service->send_response(*header, response);
            })
            .detach();
        break;
    default:
        RCLCPP_ERROR(get_logger(), "Unsupported message type. Aborting.");
        abort();
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    }
}

nlohmann::json CVNodeManager::segmentation_to_json(const SegmentationMsg &segmentation)
{
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
    return json;
}

void CVNodeManager::async_broadcast_request(
    const std::shared_ptr<rmw_request_id_t> header,
    const SegmentCVNodeSrv::Request::SharedPtr request)
{
    if (std::get<1>(cv_node) == nullptr)
    {
        RCLCPP_ERROR(get_logger(), "No CVNode registered");
        RuntimeProtocolSrv::Response response;
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
    }

    std::get<1>(cv_node)->async_send_request(
        request,
        [this, header](rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            auto response = future.get();
            RuntimeProtocolSrv::Response dataprovider_response;
            switch (response->message_type)
            {
            case OK:
                RCLCPP_DEBUG(get_logger(), "Received confirmation");
                dataprovider_response.response.message_type = OK;
                break;
            case ERROR:
                RCLCPP_ERROR(get_logger(), "Failed to receive confirmation from the CVNode");
                dataprovider_response.response.message_type = ERROR;
                abort();
                break;
            default:
                RCLCPP_ERROR(get_logger(), "Unknown response from the CV node");
                dataprovider_response.response.message_type = ERROR;
                abort();
                break;
            }
            dataprovider_service->send_response(*header, dataprovider_response);
        });
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
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
    }

    std::get<1>(cv_node)->async_send_request(
        request,
        [this, header, callback](rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            auto response = future.get();
            RuntimeProtocolSrv::Response dataprovider_response;
            if (response->message_type != OK)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Failed to receive confirmation from the CVNode (%d)",
                    response->message_type);
                dataprovider_response.response.message_type = ERROR;
                abort();
                dataprovider_service->send_response(*header, dataprovider_response);
                return;
            }

            RCLCPP_DEBUG(get_logger(), "Received response from the CVNode. Executing callback");
            dataprovider_response = callback(response);
            if (dataprovider_response.response.message_type != OK)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Failed to process response from the CVNode (%d)",
                    dataprovider_response.response.message_type);
                abort();
            }
            dataprovider_service->send_response(*header, dataprovider_response);
        });
}

SegmentCVNodeSrv::Request::SharedPtr CVNodeManager::extract_images(std::vector<uint8_t> &input_data_b)
{
    SegmentCVNodeSrv::Request::SharedPtr request = std::make_shared<SegmentCVNodeSrv::Request>();

    try
    {
        nlohmann::json data_j = nlohmann::json::parse(std::string(input_data_b.begin(), input_data_b.end()));
        std::vector<nlohmann::json> data = data_j.at("data");
        if (data.size() == 0)
        {
            RCLCPP_ERROR(get_logger(), "Data is empty");
            request->message_type = ERROR;
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
        RCLCPP_ERROR(get_logger(), "Error while parsing data json: %s", e.what());
        request->message_type = ERROR;
        return request;
    }
    request->message_type = DATA;
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

void CVNodeManager::abort()
{
    SegmentCVNodeSrv::Request::SharedPtr request = std::make_shared<SegmentCVNodeSrv::Request>();
    request->message_type = ERROR;
    std::get<1>(cv_node)->async_send_request(
        request,
        [this](rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            auto response = future.get();
            if (response->message_type == ERROR)
            {
                RCLCPP_DEBUG(get_logger(), "Received confirmation");
            }
            else
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Error while aborting the node, unknown response (%d)",
                    response->message_type);
            }
        });
}

} // namespace cvnode_manager

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
