#include <cvnode_manager/cvnode_manager.hpp>
#include <kenning_computer_vision_msgs/runtime_msg_type.hpp>
#include <nlohmann/json.hpp>

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
    switch (request->request.message_type)
    {
    case OK:
        if (!dataprovider_initialized)
        {
            input_shape.clear();

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
    case MODEL:
        prepare_nodes(header);
        break;
    case IOSPEC:
        extract_input_spec(header, request);
        break;
    default:
        if (inference_scenario_func != nullptr)
        {
            inference_scenario_func(header, request);
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

void CVNodeManager::extract_input_spec(
    const std::shared_ptr<rmw_request_id_t> header,
    const std::shared_ptr<RuntimeProtocolSrv::Request> request)
{
    RuntimeProtocolSrv::Response response = RuntimeProtocolSrv::Response();
    try
    {
        nlohmann::json iospec_j =
            nlohmann::json::parse(std::string(request->request.data.begin(), request->request.data.end()));
        std::vector<nlohmann::json> input_spec = iospec_j.at("input");
        if (input_spec.size() != 1)
        {
            // TODO: Abort further processing
            RCLCPP_ERROR(get_logger(), "Input spec length is not 1");
            response.response.message_type = ERROR;
            dataprovider_service->send_response(*header, response);
            return;
        }
        input_shape = input_spec.at(0).at("shape").get<std::vector<int>>();
    }
    catch (nlohmann::json::exception &e)
    {
        // TODO: Abort further processing
        RCLCPP_ERROR(get_logger(), "Could not extract input spec: %s", e.what());
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
        return;
    }

    std::string input_shape_str = "[";
    for (auto &dim : input_shape)
    {
        input_shape_str += std::to_string(dim) + ", ";
    }
    input_shape_str += "]";
    RCLCPP_DEBUG(get_logger(), "Extracted input shape: %s", input_shape_str.c_str());

    response.response.message_type = OK;
    dataprovider_service->send_response(*header, response);
    return;
}

void CVNodeManager::prepare_nodes(const std::shared_ptr<rmw_request_id_t> header)
{
    RuntimeProtocolSrv::Response response;
    if (cv_nodes.size() < 1)
    {
        // TODO: Reset node here
        RCLCPP_ERROR(get_logger(), "No CV nodes registered");
        response.response.message_type = ERROR;
        dataprovider_service->send_response(*header, response);
    }

    std::shared_ptr<InferenceCVNodeSrv::Request> cv_node_request = std::make_shared<InferenceCVNodeSrv::Request>();
    cv_node_request->message_type = MODEL;
    answer_counter = 0;
    for (auto &cv_node : cv_nodes)
    {
        cv_node.second->async_send_request(
            cv_node_request,
            [this, header, &cv_node](rclcpp::Client<InferenceCVNodeSrv>::SharedFuture future)
            {
                RCLCPP_DEBUG(get_logger(), "Received response from the %s node", cv_node.first.c_str());
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
                        RCLCPP_DEBUG(get_logger(), "All CV nodes are prepared (%d)", answer_counter);
                        dataprovider_response.response.message_type = OK;
                        dataprovider_service->send_response(*header, dataprovider_response);
                        answer_counter = 0;
                    }
                    break;
                case ERROR:
                    RCLCPP_ERROR(get_logger(), "Could not send model to the CV node");
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
    RCLCPP_DEBUG(get_logger(), "Sent preparation request to the CV nodes");
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
