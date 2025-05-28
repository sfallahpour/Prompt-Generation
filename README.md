To make prompt design easier and more effective, I wrote a small script that uses two LLMs in a collaborative-competitive loop—one generates the prompt, and the other critiques it. Think of it as a mini-GAN but for prompts. 

💡 How it works:

One LLM generates a prompt based on your query.

The second LLM reviews and critiques it.

They go back and forth (up to max_iterations, which you can set—default is 4) until they agree on a strong final version.

You can try queries like:
"Create a prompt for a SQL agent that reports to a supervisor agent, incorporating OpenAI best practices from the OpenAI prompting guide."

You'll see how the prompt evolves through feedback—fascinating to watch in real time.

Built on the Autogen framework (make sure it’s installed).

Also you need to create your .env file including your API connections and credentials

