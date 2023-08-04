#include <cvnode_manager/cvnode_manager.hpp>

namespace cvnode_manager
{

using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using RuntimeProtocolSrv = kenning_computer_vision_msgs::srv::RuntimeProtocolSrv;

CVNodeManager::CVNodeManager(
    const std::string node_name,
    const std::string &manage_service_name,
    const rclcpp::NodeOptions &options)
    : Node(node_name, options)
{
    manage_service = create_service<ManageCVNode>(
        manage_service_name,
        std::bind(&CVNodeManager::manage_node_callback, this, std::placeholders::_1, std::placeholders::_2));
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
    RCLCPP_INFO(get_logger(), "Registering the node '%s'", node_name.c_str());

    response->status = false;

    // Check if the node is already registered
    if (cv_nodes.find(node_name) != cv_nodes.end())
    {
        response->message = "The node is already registered";
        RCLCPP_ERROR(get_logger(), "The node '%s' is already registered", node_name.c_str());
        return;
    }

    // Create a client to communicate with the node
    rclcpp::Client<RuntimeProtocolSrv>::SharedPtr cv_node_service;
    if (!initialize_service_client<RuntimeProtocolSrv>(request->srv_name, cv_node_service))
    {
        response->message = "Could not initialize the communication service client";
        RCLCPP_ERROR(get_logger(), "Could not initialize the communication service client");
        return;
    }

    response->status = true;
    response->message = "The node is registered";
    cv_nodes[node_name] = cv_node_service;

    RCLCPP_INFO(get_logger(), "The node '%s' is registered", node_name.c_str());
}

void CVNodeManager::unregister_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    [[maybe_unused]] ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;

    RCLCPP_INFO(get_logger(), "Unregistering the node '%s'", node_name.c_str());

    // Check if the node is already registered
    if (cv_nodes.find(node_name) == cv_nodes.end())
    {
        RCLCPP_ERROR(get_logger(), "The node '%s' is not registered", node_name.c_str());
        return;
    }

    cv_nodes.erase(node_name);

    RCLCPP_INFO(get_logger(), "The node '%s' is unregistered", node_name.c_str());
    return;
}

} // namespace cvnode_manager
