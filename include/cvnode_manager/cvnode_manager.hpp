#pragma once

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <unordered_map>

#include <cvnode_msgs/srv/manage_cv_node.hpp>
#include <cvnode_msgs/srv/runtime_protocol_srv.hpp>

namespace cvnode_manager
{

/**
 * Node to manage testing process of the computer vision system.
 */
class CVNodeManager : public rclcpp::Node
{
private:
    /**
     * Callback for the service to manage the BaseCVNode-like nodes.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void manage_node_callback(const cvnode_msgs::srv::ManageCVNode::Request::SharedPtr request,
                              cvnode_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to register the BaseCVNode-like nodes.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void register_node_callback(const cvnode_msgs::srv::ManageCVNode::Request::SharedPtr request,
                                cvnode_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to unregister the BaseCVNode-like nodes.
     *
     * @param request Request of the service for unregistering the node.
     * @param response Response of the service for unregistering the node.
     */
    void unregister_node_callback(const cvnode_msgs::srv::ManageCVNode::Request::SharedPtr request,
                                  [[maybe_unused]] cvnode_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Creates a client to a service.
     *
     * @param service_name Name of the service.
     * @param client Client of the service to be created and initialized.
     *
     * @tparam TClient Type of the service client.
     *
     * @return True if the client was created and initialized successfully.
     */
    template <class TClient>
    bool initialize_service_client(const std::string &service_name, typename rclcpp::Client<TClient>::SharedPtr &client)
    {
        if (service_name.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "%s", "The process service name is empty");
            return false;
        }

        client = this->create_client<TClient>(service_name);

        if (!client->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_ERROR(this->get_logger(), "The service '%s' is not available", service_name.c_str());
            return false;
        }
        return true;
    }

    /// Service to register the CVNode.
    rclcpp::Service<cvnode_msgs::srv::ManageCVNode>::SharedPtr manage_service;

    /// Map of registered nodes.
    std::unordered_map<std::string, rclcpp::Client<cvnode_msgs::srv::RuntimeProtocolSrv>::SharedPtr> cv_nodes;

public:
    /**
     * Constructor.
     *
     * @param node_name Name of the node.
     * @param manage_service_name Name of the service to manage BaseCVNode-like nodes.
     * @param options Node options.
     */
    CVNodeManager(const std::string node_name, const std::string &manage_service_name,
                  const rclcpp::NodeOptions &options);
};

} // namespace cvnode_manager
