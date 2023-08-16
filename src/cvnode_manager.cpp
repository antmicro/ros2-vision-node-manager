// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_manager/cvnode_manager.hpp>
#include <kenning_computer_vision_msgs/runtime_msg_type.hpp>
#include <nlohmann/json.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace cvnode_manager
{
using namespace kenning_computer_vision_msgs::runtime_message_type;

using InferenceCVNodeSrv = kenning_computer_vision_msgs::srv::InferenceCVNodeSrv;
using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using RuntimeProtocolSrv = kenning_computer_vision_msgs::srv::RuntimeProtocolSrv;

CVNodeManager::CVNodeManager(const rclcpp::NodeOptions &options) : Node("cvnode_manager", options)
{
    manage_service = create_service<ManageCVNode>(
        "node_manager/register",
        std::bind(&CVNodeManager::manage_node_callback, this, std::placeholders::_1, std::placeholders::_2));

    dataprovider_service = create_service<RuntimeProtocolSrv>(
        "node_manager/dataprovider",
        std::bind(&CVNodeManager::dataprovider_callback, this, std::placeholders::_1, std::placeholders::_2));
}

void CVNodeManager::dataprovider_callback(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{

    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    InferenceCVNodeSrv::Request::SharedPtr inference_request;
    switch (request->request.message_type)
    {
    case OK:
        if (!dataprovider_initialized)
        {
            // TODO: Prepare resources and lock up untill inference is started
            RCLCPP_DEBUG(get_logger(), "Received DataProvider initialization request");
            dataprovider_initialized = true;
            response.response.message_type = OK;
        }
        else
        {
            RCLCPP_DEBUG(get_logger(), "Received unexpected 'OK' message. Responding with error.");
            response.response.message_type = ERROR;
        }
        dataprovider_service->send_response(*header, response);
        break;
    case ERROR:
        // TODO: Abort further processing
        RCLCPP_ERROR(get_logger(), "Received error message from the dataprovider");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    case IOSPEC:
        // IGNORE
        response.response.message_type = OK;
        dataprovider_service->send_response(*header, response);
        break;
    case MODEL:
        inference_request = std::make_shared<InferenceCVNodeSrv::Request>();
        inference_request->message_type = MODEL;
        async_broadcast_request(header, inference_request);
        break;
    case DATA:
        inference_request = extract_images(request->request.data);
        if (inference_request->message_type == ERROR)
        {
            // TODO: Abort further processing
            RCLCPP_DEBUG(get_logger(), "Error while extracting data. Aborting.");
            RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
            response.response.message_type = ERROR;
            dataprovider_service->send_response(*header, response);
        }
        async_broadcast_request(header, inference_request);
        break;
    default:
        if (inference_scenario_func != nullptr)
        {
            (this->*inference_scenario_func)(header, request);
        }
        else
        {
            // TODO: Abort further processing
            RCLCPP_ERROR(get_logger(), "Inference scenario function is not initialized");
            response.response.message_type = ERROR;
            dataprovider_service->send_response(*header, response);
        }
        break;
    }
}

void CVNodeManager::synthetic_scenario(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    InferenceCVNodeSrv::Request::SharedPtr inference_request = std::make_shared<InferenceCVNodeSrv::Request>();
    switch (request->request.message_type)
    {
    case PROCESS:
        inference_request->message_type = PROCESS;
        async_broadcast_request(header, inference_request);
        break;
    case OUTPUT:
        // TODO: Implement
        RCLCPP_ERROR(get_logger(), "Not yet implemented. Aborting.");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    case STATS:
        // TODO: Implement
        RCLCPP_ERROR(get_logger(), "Not yet implemented. Aborting.");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    default:
        // TODO: Abort further processing
        RCLCPP_ERROR(get_logger(), "Unsupported message type. Aborting.");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        break;
    }
}

void CVNodeManager::async_broadcast_request(
    const std::shared_ptr<rmw_request_id_t> header,
    const InferenceCVNodeSrv::Request::SharedPtr request)
{
    if (cv_nodes.size() < 1)
    {
        // TODO: Reset node here
        RuntimeProtocolSrv::Response response;
        RCLCPP_ERROR(get_logger(), "No CV nodes registered");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
    }

    answer_counter = 0;
    for (auto &cv_node : cv_nodes)
    {
        cv_node.second->async_send_request(
            request,
            [this, header](rclcpp::Client<InferenceCVNodeSrv>::SharedFuture future)
            {
                if (answer_counter < 0)
                {
                    return;
                }
                answer_counter++;
                auto response = future.get();
                RuntimeProtocolSrv::Response dataprovider_response;
                switch (response->message_type)
                {
                case OK:
                    if (static_cast<int>(cv_nodes.size()) == answer_counter)
                    {
                        RCLCPP_DEBUG(get_logger(), "Received confirmation from all CV nodes");
                        dataprovider_response.response.message_type = OK;
                        dataprovider_service->send_response(*header, dataprovider_response);
                        answer_counter = 0;
                    }
                    break;
                case ERROR:
                    RCLCPP_ERROR(get_logger(), "Failed to receive confirmation from the CV node");
                    dataprovider_response.response.message_type = ERROR;
                    dataprovider_service->send_response(*header, dataprovider_response);
                    answer_counter = -1;
                    break;
                default:
                    RCLCPP_ERROR(get_logger(), "Unknown response from the CV node");
                    dataprovider_response.response.message_type = ERROR;
                    dataprovider_service->send_response(*header, dataprovider_response);
                    answer_counter = -1;
                    break;
                }
            });
    }
}

InferenceCVNodeSrv::Request::SharedPtr CVNodeManager::extract_images(std::vector<uint8_t> &input_data_b)
{
    InferenceCVNodeSrv::Request::SharedPtr request = std::make_shared<InferenceCVNodeSrv::Request>();

    try
    {
        nlohmann::json data_j = nlohmann::json::parse(std::string(input_data_b.begin(), input_data_b.end()));
        std::vector<nlohmann::json> data = data_j.at("data");
        if (data.size() == 0)
        {
            // TODO: Abort further processing
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
        // TODO: Abort further processing
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
    RCLCPP_DEBUG(get_logger(), "Registering the node '%s'", node_name.c_str());

    response->status = false;

    // TODO: Abort if inference testing already started

    // Check if the node is already registered
    if (cv_nodes.find(node_name) != cv_nodes.end())
    {
        response->message = "The node is already registered";
        RCLCPP_ERROR(get_logger(), "The node '%s' is already registered", node_name.c_str());
        return;
    }

    // Create a client to communicate with the node
    rclcpp::Client<InferenceCVNodeSrv>::SharedPtr cv_node_service;
    if (!initialize_service_client<InferenceCVNodeSrv>(request->srv_name, cv_node_service))
    {
        response->message = "Could not initialize the communication service client";
        RCLCPP_ERROR(get_logger(), "Could not initialize the communication service client");
        return;
    }

    response->status = true;
    response->message = "The node is registered";
    cv_nodes[node_name] = cv_node_service;

    RCLCPP_DEBUG(get_logger(), "The node '%s' is registered", node_name.c_str());
}

void CVNodeManager::unregister_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    [[maybe_unused]] ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;

    RCLCPP_DEBUG(get_logger(), "Unregistering the node '%s'", node_name.c_str());

    // Check if the node is already registered
    if (cv_nodes.find(node_name) == cv_nodes.end())
    {
        RCLCPP_ERROR(get_logger(), "The node '%s' is not registered", node_name.c_str());
        return;
    }

    cv_nodes.erase(node_name);

    RCLCPP_DEBUG(get_logger(), "The node '%s' is unregistered", node_name.c_str());
    return;
}

} // namespace cvnode_manager

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
