#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <unordered_map>

#include <cvnode_msgs/srv/register_cv_node.hpp>
#include <cvnode_msgs/srv/unregister_cv_node.hpp>

namespace cvnode_manager
{

/**
 * Structure to hold CVNode's services.
 *
 * @tparam T Type of the service to process an image.
 */
template <class T> struct CVNodeService
{
    /// Client to prepare the Node for segmentation.
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr prepare_service;

    /// Client to process an image.
    typename rclcpp::Client<T>::SharedPtr process_service;

    /// Client to free Node's resources.
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cleanup_service;
};

/**
 * Node to manage testing process of the computer vision system.
 *
 * @tparam T Type of the service to process an image.
 */
template <class T> class CVNodeManager : public rclcpp::Node
{
private:
    /**
     * Callback for the service to register the node.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void register_node_callback(const cvnode_msgs::srv::RegisterCVNode::Request::SharedPtr request,
                                cvnode_msgs::srv::RegisterCVNode::Response::SharedPtr response)
    {
        std::string node_name = request->node_name;

        RCLCPP_INFO(this->get_logger(), "Registering the node '%s'", node_name.c_str());

        // Check if the node is already registered
        if (cv_nodes.find(node_name) != cv_nodes.end())
        {
            response->status = false;
            response->message = "The node is already registered";
            RCLCPP_ERROR(this->get_logger(), "The node '%s' is already registered", node_name.c_str());
            return;
        }

        CVNodeService<T> cv_node_service{};

        response->status = false;

        if (!initialize_service_client<std_srvs::srv::Trigger>(request->prepare_service_name,
                                                               cv_node_service.prepare_service))
        {
            response->message = "Could not initialize the prepare service";
            return;
        }

        if (!initialize_service_client<T>(request->process_service_name, cv_node_service.process_service))
        {
            response->message = "Could not initialize the process service";
            return;
        }

        if (!initialize_service_client<std_srvs::srv::Trigger>(request->cleanup_service_name,
                                                               cv_node_service.cleanup_service))
        {
            response->message = "Could not initialize the close service";
            return;
        }

        response->status = true;
        response->message = "The node is registered";

        cv_nodes[node_name] = cv_node_service;

        RCLCPP_INFO(this->get_logger(), "The node '%s' is registered", node_name.c_str());
        return;
    }

    /**
     * Callback for the service to unregister the node.
     *
     * @param request Request of the service for unregistering the node.
     * @param response Response of the service for unregistering the node.
     */
    void unregister_node_callback(const cvnode_msgs::srv::UnregisterCVNode::Request::SharedPtr request,
                                  [[maybe_unused]] cvnode_msgs::srv::UnregisterCVNode::Response::SharedPtr response)
    {

        std::string node_name = request->node_name;

        RCLCPP_INFO(this->get_logger(), "Unregistering the node '%s'", node_name.c_str());

        // Check if the node is already registered
        if (cv_nodes.find(node_name) == cv_nodes.end())
        {
            RCLCPP_ERROR(this->get_logger(), "The node '%s' is not registered", node_name.c_str());
            return;
        }

        cv_nodes.erase(node_name);

        RCLCPP_INFO(this->get_logger(), "The node '%s' is unregistered", node_name.c_str());
        return;
    }

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

        client = this->create_client<T>(service_name);

        if (!client->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_ERROR(this->get_logger(), "The service '%s' is not available", service_name.c_str());
            return false;
        }
        return true;
    }

    /// Service to register the CVNode.
    rclcpp::Service<cvnode_msgs::srv::RegisterCVNode>::SharedPtr register_service;

    /// Service to unregister the CVNode.
    rclcpp::Service<cvnode_msgs::srv::UnregisterCVNode>::SharedPtr unregister_service;

    std::unordered_map<std::string, CVNodeService<T>> cv_nodes; ///< Map of registered nodes.

public:
    /**
     * Constructor.
     *
     * @param node_name Name of the node.
     * @param register_service_name Name of the service to register nodes.
     * @param unregister_service_name Name of the service to unregister nodes.
     * @param options Node options.
     */
    CVNodeManager(const std::string node_name, const std::string &register_service_name,
                  const std::string &unregister_service_name, const rclcpp::NodeOptions &options)
        : Node(node_name, options)
    {
        register_service = this->create_service<cvnode_msgs::srv::RegisterCVNode>(
            register_service_name,
            std::bind(&CVNodeManager::register_node_callback, this, std::placeholders::_1, std::placeholders::_2));

        unregister_service = this->create_service<cvnode_msgs::srv::UnregisterCVNode>(
            unregister_service_name,
            std::bind(&CVNodeManager::unregister_node_callback, this, std::placeholders::_1, std::placeholders::_2));
    }
};

} // namespace cvnode_manager
