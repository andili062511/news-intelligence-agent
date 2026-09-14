from fastapi import FastAPI


app = FastAPI(
    title="News Intelligence Agent",
    description="Multi-source news intelligence agent based on RAG and tool orchestration.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "News Intelligence Agent is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
