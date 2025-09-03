import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

# Define prompt template
prompt = PromptTemplate.from_template("""
You are a helpful AI-powered investment analyst. The user may upload financial charts and describe them in text.

Analyze the chart based on the user's description and provide insights such as trends, anomalies, or investment implications.

Use markdown formatting and tables where appropriate.

User description: {question}
""")

# Initialize Groq model
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Create LangChain pipeline
chain = prompt | llm

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    # Extract image file paths (optional, for future use or logging)
    image_paths = [file.path for file in message.elements if "image" in file.mime]

    # Use user's message content as chart description
    msg = cl.Message(content="")
    response = await cl.make_async(chain.invoke)({"question": message.content})
    await msg.stream_token(response["text"])
    await msg.send()