# Langsmith will trace whole langraph flow autmatically as we included langsmith in .env
# Here if we dont take decorator(@trace on nodes) still it will trace nodes but using it.
# It also traces functions so we also get to know about the functions latency and everything
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
import os

os.environ["LANGCHAIN_PROJECT"] = "LangGraph"
load_dotenv()
model = ChatOllama(model = "llama3.2:3b", temperature = 0.1)

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description = "Detailed feedback for the essay")
    score: int = Field(description = "Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay2 = """ 
Sports are popular activities that people do for fun and fitness. They include games like football, cricket, and basketball. Sports help people stay healthy and active.
Benefits of Sports
Playing sports makes the body strong. It improves heart health and builds muscles. Sports also teach teamwork and discipline. When people play together, they learn to cooperate. This is good for children and adults.
Types of Sports
There are many types of sports. Team sports need many players like soccer. Individual sports are for one person like tennis. Outdoor sports happen in open areas. Indoor sports are in buildings like badminton. Water sports include swimming.
Problems with Sports
Sports can cause injuries. Players sometimes get hurt during games. Also, some sports cost a lot of money for equipment. Not everyone can afford them. Professional sports have pressure to win.
Conclusion
Sports are good for health and social skills. People should play them regularly. But they must be careful to avoid injuries. Sports make life better for everyone
"""
# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]  # merges parallel lists
    avg_score: float

# ---------- Traced node functions ----------
@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: UPSCState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: UPSCState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: UPSCState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}

# ---------- Build graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay",  # becomes root run name
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay2),
                "model": "llama3.2",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))