#include <cvnode_manager/cvnode_manager.hpp>
#include <kenning_computer_vision_msgs/runtime_msg_type.hpp>
#include <nlohmann/json.hpp>

namespace cvnode_manager
{

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
    const RuntimeProtocolSrv::Request::SharedPtr request,
    RuntimeProtocolSrv::Response::SharedPtr response)
{
    using namespace kenning_computer_vision_msgs::runtime_message_type;
    switch (request->request.message_type)
    {
    case OK:
        if (!dataprovider_initialized)
        {
            input_shape.clear();

            // TODO: Prepare resources and lock up untill inference is started
            RCLCPP_INFO(get_logger(), "Received DataProvider initialization request");
            response->response.message_type = OK;
            dataprovider_initialized = true;
        }
        break;
    case ERROR:
        // TODO: Abort further procecssing
        RCLCPP_ERROR(get_logger(), "Received error message from the dataprovider");
        break;
    case MODEL:
        RCLCPP_INFO(get_logger(), "Received model from the dataprovider");
        response->response.message_type = prepare_nodes();
        break;
    case IOSPEC:
        RCLCPP_INFO(get_logger(), "Received input spec from the dataprovider");
        response->response.message_type = extract_input_spec(request->request.data);
        break;
    default:
        if (inference_scenario_func != nullptr)
        {
            response->response.message_type = inference_scenario_func(request, response);
        }
        else
        {
            RCLCPP_ERROR(get_logger(), "Inference scenario function is not initialized");
            response->response.message_type = ERROR;
        }
        break;
    }
    std::string text = "my custom text";
    response->response.data = std::vector<uint8_t>(text.begin(), text.end());
}

uint8_t CVNodeManager::extract_input_spec(const std::vector<uint8_t> &iospec_b)
{
    try
    {
        nlohmann::json iospec_j = nlohmann::json::parse(std::string(iospec_b.begin(), iospec_b.end()));
        std::vector<nlohmann::json> input_spec = iospec_j.at("input");
        if (input_spec.size() != 1)
        {
            RCLCPP_ERROR(get_logger(), "Input spec length is not 1");
            return kenning_computer_vision_msgs::runtime_message_type::ERROR;
        }
        input_shape = input_spec.at(0).at("shape").get<std::vector<int>>();
    }
    catch (nlohmann::json::exception &e)
    {
        RCLCPP_ERROR(get_logger(), "Could not extract input spec: %s", e.what());
        return kenning_computer_vision_msgs::runtime_message_type::ERROR;
    }

    std::string input_shape_str = "[";
    for (auto &dim : input_shape)
    {
        input_shape_str += std::to_string(dim) + ", ";
    }
    input_shape_str += "]";
    RCLCPP_INFO(get_logger(), "Extracted input shape: %s", input_shape_str.c_str());
    return kenning_computer_vision_msgs::runtime_message_type::OK;
}

uint8_t CVNodeManager::prepare_nodes()
{
    using namespace kenning_computer_vision_msgs::runtime_message_type;
    using InferenceCVNodeSrv = kenning_computer_vision_msgs::srv::InferenceCVNodeSrv;

    if (cv_nodes.size() < 1)
    {
        RCLCPP_ERROR(get_logger(), "No CV nodes registered");
        return ERROR;
    }

    std::shared_ptr<InferenceCVNodeSrv::Request> cv_node_request = std::make_shared<InferenceCVNodeSrv::Request>();
    cv_node_request->message_type = MODEL;
    for (auto &cv_node : cv_nodes)
    {
        auto cv_node_response = cv_node.second->async_send_request(cv_node_request);
        if (cv_node_response.wait_for(std::chrono::seconds(5)) == std::future_status::ready)
        {
            if (cv_node_response.get()->message_type == OK)
            {
                continue;
            }
        }
        RCLCPP_ERROR(get_logger(), "Node '%s' is not ready. Aborting", cv_node.first.c_str());
        return ERROR;
    }
    RCLCPP_INFO(get_logger(), "All CV nodes are prepared");
    return OK;
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

    using InferenceCVNodeSrv = kenning_computer_vision_msgs::srv::InferenceCVNodeSrv;
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

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
