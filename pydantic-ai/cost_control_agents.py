import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo # ignore if you are using JupyterNotebook or Python file
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Controlling AI Inference Costs in Agents
    <br>
    It's important to control costs as a developer when developing Agents. FinOps will definitely add `AI Cost Management` as it's new vertical.
    Here, we will be looking into examples using Pydantic-AI and how we can utilize it efficiently to control Token and ToolsUsage or calls made to the AI Model using inference.
    We don't want to waste unnecessary tokens or call the tools infinitely.

    <i>I believe code is more than 1000 words of text to explain. You have a right to disagree😉</i>

    Here we will be using <b>OpenRouter</b> for <b>LLMInference</b>, OpenRouter is generous enough to provide gpt-oss and other models for free to use.

    <b> Let's go through the code below </b>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <br>
    First thing first let's Import necessary modules.
    """
    )
    return


@app.cell
def _():
    # import Pydantic Agent
    from pydantic_ai import Agent
    # import OpenRouter modules to call the api (OpenRouter is the Provider)
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    # Import Usage Limits functions and modules
    from pydantic_ai import UsageLimitExceeded, UsageLimits

    return (
        Agent,
        OpenAIChatModel,
        OpenRouterProvider,
        UsageLimitExceeded,
        UsageLimits,
    )


@app.cell
def _(Agent, OpenAIChatModel, OpenRouterProvider):
    # create the model
    model = OpenAIChatModel("openai/gpt-oss-20b",
                           provider=OpenRouterProvider(api_key='Your-API-Key'))
    # create the Agent
    agent = Agent(model)
    return agent, model


@app.cell
def _():
    # If you are using Jupyter Notebooks or Colab or Marimo make sur to add this
    # Else if you are running as an normal file skip this
    import nest_asyncio

    nest_asyncio.apply()
    return


@app.cell
def _(UsageLimits, agent):
    # Here we ask simple question to the agent also limit the number of words to 10, let's see the result
    result_sync = agent.run_sync(
        'Who Invented Computer? Answer Just with the Name',
        usage_limits=UsageLimits(response_tokens_limit=200)
    )
    # Print the Output
    print(result_sync.output)
    # Also print the Usage
    print(result_sync.usage())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Clearly above we can see that for even while asking like name of the Inventer it takes arount 138 tokens. 

    How do we handle this error ?

    Let's Try adding some exceptions here
    """
    )
    return


@app.cell
def _(UsageLimitExceeded, UsageLimits, agent):
    try:
        result_sync_2 = agent.run_sync(
            'What is a Sun?',
            usage_limits=UsageLimits(response_tokens_limit=200)
        )
        print(result_sync_2.output)
        print(result_sync_2.usage())
    except UsageLimitExceeded as e:
        print(e)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Clearly we can see that when the tokens exceeded with more than 200, we received the error. Hence we were able to control the usage limits of Agent by limiting to send only 200 tokens""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Let's see by limiting the number of requests , which can be helpful in preventing infinite loops or excessive tool calling
    <br>
    We will be reusing the `model` defined above.
    """
    )
    return


@app.cell
def _(Agent, UsageLimitExceeded, UsageLimits, model):
    from typing_extensions import TypedDict

    # Only import `ModelRetry` here , all other imports are above defined
    from pydantic_ai import ModelRetry

    # This class docstring tells the Pydantic-AI Agent 
    # "leave the output as-is, do not try to convert it"
    class NeverOutputType(TypedDict):
        """
        Never ever coerce data to this type.
        """

        never_use_this: str
    
    # create Infinite retry agent
    infinite_agent = Agent(model,
                          retries=3,
                           output_type=NeverOutputType)

    # Always retry the tool, maximum retries 5
    @infinite_agent.tool_plain(retries=5)  
    def infinite_retry_tool() -> int:
        raise ModelRetry('Please try again.')


    try:
        result_sync_3 = infinite_agent.run_sync(
            'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3)  
        )
        print(result_sync_3.output)
        print(result_sync_3.usage())
    except UsageLimitExceeded as e:
        print(e)
        #> The next request would exceed the request_limit of 3 -- output example
    return


@app.cell
def _(mo):
    mo.md(r"""##Let's see capping tool_calls example""")
    return


@app.cell
def _(Agent, UsageLimitExceeded, UsageLimits, model):

    class Counter:
        def __init__(self):
            self.value = 0

        def increment(self) -> str:
            self.value += 1
            return f'Counter value is {self.value}'

    counter = Counter()

    cap_agent = Agent(model)
    @cap_agent.tool_plain
    def do_work() -> str:
        return counter.increment()

    try:
        # Allow at most one executed tool call in this run
        cap_agent.run_sync(
            'Please call the tool twice',
            usage_limits=UsageLimits(tool_calls_limit=1)
        )
    except UsageLimitExceeded as e:
        print(e)
        #> The next tool call would exceed the tool_calls_limit of 1 (tool_calls=1) (Output of Capping)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
