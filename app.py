from pyats.topology import loader
from langchain_community.llms import Ollama
from langchain_core.tools import tool, render_text_description
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import streamlit as st

def show_ip_interface_brief():
    try:
        print("Loading testbed...")
        # Load testbed configuration from YAML file.
        testbed = loader.load('testbed.yaml')
        # Select the device from the testbed based on its name.
        device = testbed.devices['Cat8000V']
        print("Connecting to device...")
        # Establish an SSH connection to the device.
        device.connect()
        print("Executing 'show ip interface brief'...")
        parsed_output = device.parse("show ip interface brief")
        print("Disconnecting from device")
        device.disconnect()
        return parsed_output
    except Exception as e:
        return {"error": str(e)}

def show_ip_route():
    try:
        print("Loading testbed...")
        testbed = loader.load('testbed.yaml')
        device = testbed.devices['Cat8000V']
        print("Connecting to device...")
        device.connect()
        print("Executing 'show ip route'...")
        parsed_output = device.parse("show ip route")
        print("Disconnecting from device")
        device.disconnect()
        return parsed_output
    except Exception as e:
        return {"error": str(e)}


@tool() # LangChain's tool decorator registers this function as a tool to be used in the agent.
def show_interface_tool(dummy_input: str = "default") -> dict:
    """Execute the 'show ip interface brief' command on the router using pyATS and return the parsed JSON. the input is ignored"""
    return show_ip_interface_brief()

@tool()
def show_ip_route_tool(dummy_input: str = "default") -> dict:
    """Execute the 'show ip route' command on the router using pyATS and return the parsed JSON. the input is ignored"""
    return show_ip_route()


llm = Ollama(model="llama3.1", base_url="http://192.168.1.10:11434")

# Define a list of tools that the agent can use. For now, it includes only the show_interface_tool.
tools = [show_interface_tool, show_ip_route_tool]

# Render a human-readable description of the tools available, used for display purposes.
tool_description = render_text_description(tools)

# Create a template for how the agent should interact, guiding it to use the tools or provide a final answer.
template = '''
Assistant is a large language model trained by OpenAI.

Assistant is designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on various topics. As a language model, Assistant can generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide coherent and relevant responses.

Assistant is constantly learning and improving. It can process and understand large amounts of text and use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant can generate its text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on various topics.

NETWORK INSTRUCTIONS:

Assistant is a network assistant with the capability to run tools to gather information, configure the network, and provide accurate answers. You MUST use the provided tools for checking interface statuses, retrieving the running configuration, configuring settings, or finding which commands are supported.

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (it will be ignored by the tool)
Observation: the result of the action
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Action: Display [your response here]

IMPORTANT GUIDELINES:

Tool Selection: Only use the tool that is necessary to answer the question. for example:

If the question is about IP addresses of interface status, start by using the show ip interface brief tool.
Use show run only if the question explicitly requires detailed configuration that show ip interface brief cannot provide.
Check After Each Tool Use: After using a tool. check if you already have enough information to answer the question. if yes, provide the final answer immediately and do not use another tool.

For example, if asked about the IP address of Loopback0, after retrieving the data from 'show ip interface brief', respond as follows:

Thought: Do I need to use a tool? No
Final Answer: The IP address for Loopback0 is 10.0.0.1.

Avoid Repetition: if you have already provided a final answer, do not repeat it or perform additional steps. the configuration should end there.

Correct Formatting is Essentials: Make sure every response follows the format strictly ro avoid errors. Use 'Final Answer' to deliver the final output.

Handling Errors or Invalid Tool Usage: If an invalid action is taken or if there is an error. correct the thought process and provide the accurate answer directly without repeating tools.

TOOLS:

Assistant has access to the following tools:

{tools}

Begin!

Previous conversation history:

{chat_history}

New Input: {input}

{agent_scratchpad}
'''

# Define the input variables for the PromptTemplate that guide the agent's interaction.
input_variables = ["input", "tools", "tool_names", "agent_scratchpad", "chat_history"]

# Create the prompt template with partial variables for the tools and tool names that the agent can use.
prompt_template = PromptTemplate(
    template=template,
    input_variables=input_variables,
    partial_variables={"tools":tool_description, "tool_names": ", ".join(t.name for t in tools)}
)

# Create the agent that will be responsible for executing the tools and generating responses.
agent = create_react_agent(llm, tools, prompt_template)

# Create the agent executor which manages the agent and handles errors, running in verbose mode with JSON output format.
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, format="json")


# Streamlit section for creating a simple web interface.
st.title("Network Admin Virtual Engineer")
st.write("What do you want to know about the Network")

user_input = st.text_input("Enter your question:")

# Store conversation history in session state, initializing if not present.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# Initialize conversation storage if not present in session state.
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Handle the button click to send the user's question.
if st.button("Send"):
    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": ""
        })

        final_response = response.get("output", "No answer provided.")

        st.write(f"**Question:** {user_input}")
        st.write(f"**Answer:** {final_response}")
        # Append the agent's response to the conversation history.
        st.session_state.conversation.append({"role": "agent", "content": final_response})
        # Update the chat history with the latest conversation.
        st.session_state.chat_history = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation])

if st.session_state.conversation:
    st.write("## Conversation History")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")