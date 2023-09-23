// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>

#include <kenning_computer_vision_msgs/action/segmentation_action.hpp>
#include <kenning_computer_vision_msgs/msg/segmentation_msg.hpp>
#include <kenning_computer_vision_msgs/srv/manage_cv_node.hpp>
#include <kenning_computer_vision_msgs/srv/segment_cv_node_srv.hpp>

namespace cvnode_manager
{

using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;
using SegmentationMsg = kenning_computer_vision_msgs::msg::SegmentationMsg;
using SegmentationAction = kenning_computer_vision_msgs::action::SegmentationAction;

using Trigger = std_srvs::srv::Trigger;

/**
 * Structure holding information about registered CVNode-like node.
 */
struct CVNode
{
    std::string name;                                    ///< Name of the CVNode-like node.
    rclcpp::Client<Trigger>::SharedPtr prepare;          ///< Client to prepare the CVNode-like node.
    rclcpp::Client<SegmentCVNodeSrv>::SharedPtr process; ///< Client to run inference on the CVNode-like node.
    rclcpp::Client<Trigger>::SharedPtr cleanup;          ///< Client to cleanup the CVNode-like node.

    /**
     * Constructor.
     *
     * @param name Name of the CVNode-like node.
     * @param prepare Client to prepare the CVNode-like node.
     * @param process Client to run inference on the CVNode-like node.
     * @param cleanup Client to cleanup the CVNode-like node.
     */
    CVNode(
        const std::string &name,
        rclcpp::Client<Trigger>::SharedPtr prepare,
        rclcpp::Client<SegmentCVNodeSrv>::SharedPtr process,
        rclcpp::Client<Trigger>::SharedPtr cleanup)
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
     * Decides whether to accept or reject process request.
     *
     * @param uuid UUID of the process request.
     * @param goal Pointer to the goal of process request.
     *
     * @return Response of the testing process.
     */
    rclcpp_action::GoalResponse handle_test_process_request(
        const rclcpp_action::GoalUUID &uuid,
        std::shared_ptr<const SegmentationAction::Goal> goal);

    /**
     * Handles cancelation of process request.
     *
     * @param goal_handle Pointer to the goal handle of process request.
     *
     * @return Cancel response of the testing process.
     */
    rclcpp_action::CancelResponse
    handle_test_process_cancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle);

    /**
     * Executes processing request.
     *
     * @param goal_handle Pointer to the goal handle of process request.
     */
    void handle_test_process_start_processing(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle);

    /**
     * Responds to the process request.
     * Publishes data to visualization topic if it is enabled.
     */
    void publish_data();

    /**
     * Checks if input data is available.
     *
     * @return True if input data is available, false otherwise.
     */
    bool is_input_data_available();

    /**
     * Prepares CVNode-like node for the testing process.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void prepare_node(const std::shared_ptr<rmw_request_id_t> header, const Trigger::Request::SharedPtr request);

    /**
     * Uploads time measurements of inference testing in a JSON-encoded string.
     * Performs cleanup of the CVNode-like node.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void upload_measurements(const std::shared_ptr<rmw_request_id_t> header, const Trigger::Request::SharedPtr request);

    /**
     * Synthetic testing scenario.
     * Forwards input data to the CVNode-like node and waits for response.
     */
    void execute_synthetic_inference_scenario();

    /**
     * Real-world testing scenario for the first request.
     * Tries to always finish the oldest request by ignoring the newer ones.
     */
    void execute_real_world_first_inference_scenario();

    /**
     * Real-world testing scenario for the last request.
     * Tries to always process the newest request by aborting the older ones.
     */
    void execute_real_world_last_inference_scenario();

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

    rclcpp::Service<ManageCVNode>::SharedPtr manage_service;  ///< Service to register the CVNode
    rclcpp::Service<Trigger>::SharedPtr prepare_service;      ///< Service to prepare the CVNode
    rclcpp::Service<Trigger>::SharedPtr measurements_service; ///< Service to get inference measurements

    /// Server to communicate with DataProvider
    rclcpp_action::Server<SegmentationAction>::SharedPtr dataprovider_server;
    /// Publisher to publish input data to GUI
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gui_input_publisher;
    /// Publisher to publish output data to GUI
    rclcpp::Publisher<SegmentationMsg>::SharedPtr gui_output_publisher;

    /// Condition variable for DataProvider to wait for CVNode to be initialized
    std::condition_variable cvnode_wait_cv;

    bool dataprovider_initialized = false; ///< Flag indicating whether the DataProvider is initialized
    CVNode cv_node = CVNode();             ///< Registered CVNode-like node used for inference

    ///< Flag indicating whether the CVNode-like node is processing a request
    bool processing_request = false;
    /// Inference request for registered CVNode-like node
    SegmentCVNodeSrv::Request::SharedPtr cvnode_request = std::make_shared<SegmentCVNodeSrv::Request>();
    /// Inference response from registered CVNode-like node
    SegmentCVNodeSrv::Response::SharedPtr cvnode_response = std::make_shared<SegmentCVNodeSrv::Response>();
    /// Request handle of the inference request from DataProvider
    std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> request_handle;

    rclcpp::TimerBase::SharedPtr real_world_timer; ///< Timer for periodic execution of real-world inference scenarios
    nlohmann::json measurements;                   ///< Measurements from inference in JSON format
    std::chrono::steady_clock::time_point start;   ///< Timestamp indicating start of the inference
    std::chrono::steady_clock::time_point end;     ///< Timestamp indicating ending of the inference
public:
    /**
     * Constructor.
     *
     * @param options Node options.
     */
    CVNodeManager(const rclcpp::NodeOptions &options);
};

} // namespace cvnode_manager
