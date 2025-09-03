import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an investment analyst that researches stock prices, company infos, stock fundamentals, analyst recommendations, and historical prices.

Format your response using markdown and use tables to display data where possible.

Question: {question}
"""
)


llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

chain = LLMChain(llm=llm, prompt=prompt)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    response = await cl.make_async(chain.run)(question=message.content)
    await msg.stream_token(response)
    await msg.send()