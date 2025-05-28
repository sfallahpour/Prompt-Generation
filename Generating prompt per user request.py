from autogen import AssistantAgent, UserProxyAgent
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI configuration
azure_config = {
    "model": os.getenv("AZURE_model_DEPLOYMENT_NAME"),
    "api_type": "azure",
    "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
}

def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """
    Retry a function with exponential backoff.
    """
    delay = initial_delay
    for retry in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and retry < max_retries - 1:
                print(f"\nRate limit hit. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            raise e

def get_last_assistant_message(messages):
    """Helper function to get the last assistant message from a conversation."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]
    return None

# Create prompt generator agent
# Note: This agent only generates prompt templates. It does NOT execute, validate, or run any queries
# (such as SOQL queries). You will need to add your own examples and validation later.
prompt_generator = AssistantAgent(
    name="PromptGenerator",
    llm_config={
        "temperature": 0.7,
        "config_list": [azure_config]
    },
    system_message="""You are an expert prompt engineer. Your role is to generate clear, effective prompts based on user queries.
    
    IMPORTANT: Generate prompts ONLY. Do not engage in conversation or pleasantries.
    
    Follow these guidelines:
    1. Break down complex requirements into clear instructions
    2. Include specific examples where helpful
    3. Define scope and constraints clearly
    4. Use consistent formatting and structure
    5. Consider edge cases and failure modes
    
    Format your response STRICTLY as:
    ### Generated Prompt:
    [Your prompt here]
    """
)

# Create prompt critic agent
# Note: This agent only reviews and critiques prompt templates. It does NOT execute, validate, or run any queries
# (such as SOQL queries). You will need to add your own examples and validation later.
prompt_critic = AssistantAgent(
    name="PromptCritic", 
    llm_config={
        "temperature": 0.3,
        "config_list": [azure_config]
    },
    system_message="""You are an expert prompt critic. Your role is to analyze prompts and provide constructive feedback to improve them.
    
    IMPORTANT: Focus ONLY on critiquing and improving the prompt. Do not engage in conversation or pleasantries.
    
    Evaluate prompts based on:
    1. Clarity and specificity
    2. Completeness of instructions
    3. Potential ambiguities or gaps
    4. Appropriate constraints and guardrails
    5. Overall effectiveness for intended use case
    
    Format your response STRICTLY as:
    ### Critique:
    [Your detailed critique here]
    
    ### Suggested Improvements:
    [List of specific improvements if needed]
    
    ### Final Approved Prompt:
    [The final, improved version of the prompt]
    """
)

# Create user proxy agent
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config={
        "temperature": 0.1,
        "config_list": [azure_config]
    }
)

def generate_and_refine_prompt(user_query: str, max_iterations: int = 4) -> str:
    """
    Generate and refine a prompt through sequential agent collaboration.
    
    Note: This function only generates prompt templates. It does NOT execute, validate, or run any queries
    (such as SOQL queries). You will need to add your own examples and validation later.
    
    Args:
        user_query: The user's request for what kind of prompt to generate
        max_iterations: Maximum number of iterations between generator and critic (default: 3)
    """
    try:
        # Step 1: Generate initial prompt
        print("\nGenerating initial prompt...")
        
        def generate_initial_prompt():
            # Reset the chat for a clean start
            user_proxy.reset()
            user_proxy.initiate_chat(
                prompt_generator,
                message=f"Generate a prompt for the following query: {user_query}"
            )
            messages = user_proxy.chat_messages[prompt_generator]
            return get_last_assistant_message(messages)
        
        generated_prompt = retry_with_backoff(generate_initial_prompt)
        if not generated_prompt:
            raise Exception("Failed to get initial prompt from generator")
            
        print("\nInitial prompt generated. Getting critique...")
        
        # Step 2: Get critique and improvements
        current_prompt = generated_prompt
        iteration = 0
        all_critiques = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}/{max_iterations}: Getting critique...")
            
            critique_message = f"""Review and improve this prompt:

{current_prompt}"""
            
            def get_critique():
                try:
                    # Reset the chat for a clean conversation with the critic
                    user_proxy.reset()
                    user_proxy.initiate_chat(
                        prompt_critic,
                        message=critique_message
                    )
                    messages = user_proxy.chat_messages[prompt_critic]
                    return get_last_assistant_message(messages)
                except Exception as e:
                    if "Maximum number of consecutive auto-replies reached" in str(e):
                        # If we hit the auto-reply limit, try to get the last message
                        print("\nMaximum auto-replies reached. Attempting to extract final prompt...")
                        try:
                            messages = user_proxy.chat_messages[prompt_critic]
                            if messages:
                                return get_last_assistant_message(messages)
                        except:
                            pass
                    raise e
            
            critic_response = retry_with_backoff(get_critique)
            if not critic_response:
                print(f"\nWarning: No response from critic in iteration {iteration}")
                break
                
            all_critiques.append(critic_response)
            
            # Extract the final approved prompt from the critic's response
            if "### Final Approved Prompt:" in critic_response:
                current_prompt = critic_response.split("### Final Approved Prompt:")[1].strip()
                print(f"Iteration {iteration}: Prompt approved by critic.")
                break
            else:
                print(f"Iteration {iteration}: Prompt needs improvement. Continuing to next iteration...")
        
        # Format the complete output
        complete_output = f"""
{'='*80}
INITIAL GENERATED PROMPT:
{'='*80}
{generated_prompt}

{'='*80}
PROMPT CRITIC'S FEEDBACK:
{'='*80}
{chr(10).join(all_critiques)}

{'='*80}
FINAL APPROVED PROMPT:
{'='*80}
{current_prompt}
{'='*80}
"""
        return complete_output
        
    except Exception as e:
        error_message = f"""
Error occurred while generating prompt: {str(e)}
Please try again in 60 seconds if you hit the rate limit.
"""
        return error_message

# Example usage:
if __name__ == "__main__":
    # Get user query
    user_query = input("Please enter your prompt generation query: ")
    
    print("\nStarting prompt generation process...")
    print("This may take a few moments...\n")
    
    refined_prompt = generate_and_refine_prompt(user_query)
    print(refined_prompt)
