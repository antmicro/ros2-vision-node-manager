// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>

#include <kenning_computer_vision_msgs/msg/segmentation_msg.hpp>
#include <kenning_computer_vision_msgs/srv/manage_cv_node.hpp>
#include <kenning_computer_vision_msgs/srv/runtime_protocol_srv.hpp>
#include <kenning_computer_vision_msgs/srv/segment_cv_node_srv.hpp>

namespace cvnode_manager
{

using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using RuntimeProtocolSrv = kenning_computer_vision_msgs::srv::RuntimeProtocolSrv;
using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;
using SegmentationMsg = kenning_computer_vision_msgs::msg::SegmentationMsg;

/**
 * Structure holding information about registered CVNode-like node.
 */
struct CVNode
{
    std::string name;                                          ///< Name of the CVNode-like node.
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr prepare; ///< Client to prepare the CVNode-like node.
    rclcpp::Client<SegmentCVNodeSrv>::SharedPtr process;       ///< Client to run inference on the CVNode-like node.
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cleanup; ///< Client to cleanup the CVNode-like node.

    /**
     * Constructor.
     *
     * @param name Name of the CVNode-like node.
     * @param prepare Client to prepare the CVNode-like node.
     * @param process Client to communicate with the CVNode-like node.
     * @param cleanup Client to cleanup the CVNode-like node.
     */
    CVNode(
        const std::string &name,
        rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr prepare,
        rclcpp::Client<SegmentCVNodeSrv>::SharedPtr process,
        rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cleanup)
        : name(name), prepare(prepare), process(process), cleanup(cleanup)
    {
    }

    /**
     * Default constructor.
     */
    CVNode() = default;
};

/**
 * Node to manage testing process of the computer vision system.
 */
class CVNodeManager : public rclcpp::Node
{
private:
    /**
     * Callback for the service to manage the CVNode-like node.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void
    manage_node_callback(const ManageCVNode::Request::SharedPtr request, ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to register the CVNode-like node.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void
    register_node_callback(const ManageCVNode::Request::SharedPtr request, ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to unregister the CVNode-like node.
     *
     * @param request Request of the service for unregistering the node.
     * @param response Response of the service for unregistering the node.
     */
    void unregister_node_callback(
        const ManageCVNode::Request::SharedPtr request,
        [[maybe_unused]] ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for communication with DataProvider.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void dataprovider_callback(
        const std::shared_ptr<rmw_request_id_t> header,
        const RuntimeProtocolSrv::Request::SharedPtr request);

    /**
     * Holds DataProvider initialization up until inference should be started.
     *
     * @param header Header of the service request.
     */
    void initialize_dataprovider(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Extracts input data from DataProvider request.
     * Prepares inference request for CVNode-like node.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void extract_data_from_request(
        const std::shared_ptr<rmw_request_id_t> header,
        const RuntimeProtocolSrv::Request::SharedPtr request);

    /**
     * Configures testing scenario strategy based on 'scenario' ROS2 parameter.
     *
     * @return True if initialization was successful, false otherwise.
     */
    bool configure_scenario();

    /**
     * Extracts input data from bytes-encoded json.
     *
     * @param input_data_b Bytes-encoded input data.
     *
     * @return Request with data to distribute. If error occurred, message type is set to ERROR.
     */
    SegmentCVNodeSrv::Request::SharedPtr extract_images(std::vector<uint8_t> &input_data_b);

    /**
     * Converts output segmentations from the CVNode-like node to bytes-encoded json state.
     *
     * @param response CVNode-like node response with segmentations attached.
     *
     * @return Bytes-encoded output data in JSON format.
     */
    nlohmann::json segmentations_to_json(const SegmentCVNodeSrv::Response::SharedPtr response);

    /**
     * Synthetic testing scenario.
     * Forwards input data to the CVNode-like node and waits for response.
     *
     * @param header Header of the service request.
     */
    void execute_synthetic_inference_scenario(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Real-world testing scenario for the last request.
     * Tries to always process the newest request by aborting the older ones.
     *
     * @param header Header of the service request.
     */
    void execute_real_world_last_inference_scenario(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Real-world testing scenario for the first request.
     * Tries to always finish the oldest request by ignoring the newer ones.
     *
     * @param header Header of the service request.
     */
    void execute_real_world_first_inference_scenario(const std::shared_ptr<rmw_request_id_t> header);

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

    /**
     * Aborts further processing.
     * Reports error to the registered CVNode-like node and DataProvider.
     *
     * @param header Header of the service request.
     * @param error_msg Error message to be logged.
     */
    void abort(const std::shared_ptr<rmw_request_id_t> header, const std::string &error_msg);

    rclcpp::Service<ManageCVNode>::SharedPtr manage_service; ///< Service to register the CVNode

    /// Client to communicate with DataProvider
    rclcpp::Service<RuntimeProtocolSrv>::SharedPtr dataprovider_service;

    /// Publisher to publish input data to GUI
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gui_input_publisher;

    /// Publisher to publish output data to GUI
    rclcpp::Publisher<SegmentationMsg>::SharedPtr gui_output_publisher;

    /// Condition variable for DataProvider to wait for CVNode to be initialized
    std::condition_variable cvnode_wait_cv;

    bool dataprovider_initialized = false; ///< Flag indicating whether the DataProvider is initialized
    CVNode cv_node = CVNode();             ///< Registered CVNode-like node used for inference

    /// Request to communicate with the CVNode-like node
    SegmentCVNodeSrv::Request::SharedPtr cvnode_request = std::make_shared<SegmentCVNodeSrv::Request>();

    // Shared future from last request to CVNode-like node
    rclcpp::Client<SegmentCVNodeSrv>::SharedFuture cvnode_future = rclcpp::Client<SegmentCVNodeSrv>::SharedFuture();

    /// Function responsible for executing proper inference scenario strategy
    std::function<void(const std::shared_ptr<rmw_request_id_t>)> inference_scenario_func = nullptr;

    nlohmann::json output_data;                  ///< Output data from inference in JSON format
    nlohmann::json measurements;                 ///< Measurements from inference in JSON format
    std::chrono::steady_clock::time_point start; ///< Timestamp indicating start of the inference
    std::chrono::steady_clock::time_point end;   ///< Timestamp indicating ending of the inference
public:
    /**
     * Constructor.
     *
     * @param options Node options.
     */
    CVNodeManager(const rclcpp::NodeOptions &options);
};

} // namespace cvnode_manager
