#pragma once

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <unordered_map>

#include <kenning_computer_vision_msgs/srv/inference_cv_node_srv.hpp>
#include <kenning_computer_vision_msgs/srv/manage_cv_node.hpp>
#include <kenning_computer_vision_msgs/srv/runtime_protocol_srv.hpp>

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
    void manage_node_callback(
        const kenning_computer_vision_msgs::srv::ManageCVNode::Request::SharedPtr request,
        kenning_computer_vision_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to register the BaseCVNode-like nodes.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void register_node_callback(
        const kenning_computer_vision_msgs::srv::ManageCVNode::Request::SharedPtr request,
        kenning_computer_vision_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to unregister the BaseCVNode-like nodes.
     *
     * @param request Request of the service for unregistering the node.
     * @param response Response of the service for unregistering the node.
     */
    void unregister_node_callback(
        const kenning_computer_vision_msgs::srv::ManageCVNode::Request::SharedPtr request,
        [[maybe_unused]] kenning_computer_vision_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for communication with dataprovider.
     *
     * @param header Header of the service.
     * @param request Request of the service.
     */
    void dataprovider_callback(
        const std::shared_ptr<rmw_request_id_t> header,
        const std::shared_ptr<kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request> request);

    /**
     * Parses IO specification and saves it for further images processing.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void extract_input_spec(
        const std::shared_ptr<rmw_request_id_t> header,
        const std::shared_ptr<kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request> request);

    /**
     * Prepares all the registered CVNode-like nodes.
     *
     * @param header Header of the service request.
     */
    void prepare_nodes(const std::shared_ptr<rmw_request_id_t> header);

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

    /// Service to register the CVNode
    rclcpp::Service<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedPtr manage_service;

    /// Client to communicate with Kenning
    rclcpp::Service<kenning_computer_vision_msgs::srv::RuntimeProtocolSrv>::SharedPtr dataprovider_service;

    /// Map of registered nodes
    std::unordered_map<std::string, rclcpp::Client<kenning_computer_vision_msgs::srv::InferenceCVNodeSrv>::SharedPtr>
        cv_nodes;

    /// Testing scenario function
    std::function<void(
        const std::shared_ptr<rmw_request_id_t>,
        const kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr)>
        inference_scenario_func = nullptr;

    bool dataprovider_initialized = false; ///< Indicates whether the dataprovider is initialized
    std::vector<int> input_shape;          ///< Input specification of dataprovider
    int answer_counter = 0;                ///< Counter of received answers from the CVNodes

public:
    /**
     * Constructor.
     *
     * @param options Node options.
     */
    CVNodeManager(const rclcpp::NodeOptions &options);
};

} // namespace cvnode_manager
