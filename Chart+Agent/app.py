import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

# === Prompt for text-based investment queries ===
text_prompt = PromptTemplate.from_template("""
You are a helpful AI-powered investment analyst. The user may ask about stock prices, company fundamentals, analyst recommendations, or historical trends.

Format your response using markdown and use tables where appropriate.

Question: {question}
""")

# === Prompt for chart analysis ===
chart_prompt = PromptTemplate.from_template("""
You are a financial chart analyst. The user has uploaded a chart and described it in text.

Analyze the chart based on the user's description. Identify trends, anomalies, or investment implications.

Use markdown formatting and tables where appropriate.

Chart description: {question}
""")

# === Groq LLaMA model ===
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# === LangChain pipelines ===
text_chain = text_prompt | llm
chart_chain = chart_prompt | llm

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("text_chain", text_chain)
    cl.user_session.set("chart_chain", chart_chain)

@cl.on_message
async def on_message(message: cl.Message):
    text_chain = cl.user_session.get("text_chain")
    chart_chain = cl.user_session.get("chart_chain")

    # Check if user uploaded an image
    has_image = any("image" in file.mime for file in message.elements)

    msg = cl.Message(content="")

    if has_image:
        # Use chart analysis pipeline
        response = await cl.make_async(chart_chain.invoke)({"question": message.content})
    else:
        # Use investment Q&A pipeline
        response = await cl.make_async(text_chain.invoke)({"question": message.content})

    await msg.stream_token(response["text"])
    await msg.send()