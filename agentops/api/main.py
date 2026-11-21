from fastapi import FastAPI

from agentops.api.routes import analytics, documents, labels, runs, tools

app = FastAPI(
    title="AgentOps Intelligence API",
    description="AI Agent Monitoring and Analytics",
    version="2.0.0"
)


@app.get("/")
def home():
    return {"message": "AgentOps API is running!"}


app.include_router(analytics.router)
app.include_router(runs.router)
app.include_router(tools.router)
app.include_router(documents.router)
app.include_router(labels.router)