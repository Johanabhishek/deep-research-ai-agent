from agents.researcher_agent import run_research_agent
from agents.answer_drafter_agent import run_drafter_agent

if __name__ == "__main__":
    print("Starting Deep Research AI Agent...")
    # Example entry points
    run_research_agent("Latest AI regulations in the EU")
    run_drafter_agent("Latest AI regulations in the EU")
