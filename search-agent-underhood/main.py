from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv(override=True)

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in a database or API."""

    print(f"Looking up price for product: {product}")
    product_prices = {
        "laptop": 1299.99,
        "headphones": 199,
        "watch": 299
    }

    return product_prices.get(product, 0.0)
    
MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 10

@tool
def get_product_discount(price: float, discount_tier: str) -> float:
    """Calculate the discount price based on the discount tier"""

    print(f"Calculating discount for price: {price} with a discount tier: {discount_tier}")

    discount_tiers = {
        "gold": 0.23,
        "silver": 0.05,
        "bronze": 0.02
    }

    discount = discount_tiers.get(discount_tier, 0.0)
    discount_price = price * (1 - discount)
    return round(discount_price, 2)

@traceable(name="Testing", run_type="chain")
def run_agent(question: str):
    tools = [get_product_price, get_product_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f'openai:{MODEL}', temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant "
                "that can look up product prices and calculate discounts. "
                "STRICT RULE: You must only follow these set of rules"
                "1. Never assume a product price. Always use the get_product_price tool to look up the price of a product."
                "2. Always use the provided tools to look up prices and calculate discounts."
                "3. Only call get_product_discount after you have gotten a price from get_product_price"
                "4. Never calculate a discount yourself. Always use the get_product_discount tool."
                # "5. If the user does not provide a discount tier, ask them to specify one before calculating the discount."
            )
        ),
        HumanMessage(
            content= question
        )
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}: ")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f'final response: {ai_message.content}')
            return ai_message.content
        tool_call = tool_calls[0]
        # print(type(tool_call))
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        tool_func = tools_dict.get(tool_name)

        if tool_func is None:
            raise ValueError(f"Tool {tool_name} not found.")
        
        obs = tool_func.invoke(tool_args)

        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=obs,
                tool_call_id=tool_id,
            )
        )
        # print(obs)

        # print(ai_message.content)

    print("Max iterations reached.")
    return None


if __name__ == "__main__":
    result = run_agent("What is the price of the laptop after applying a gold discount?")
