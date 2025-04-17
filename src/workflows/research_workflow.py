import sys
import os
from typing import Dict, List

library_path = os.path.abspath('/Users/mitchens/Local/6I-CYBORG-AGENTS/AGENT_PKM/src')
sys.path.append(library_path)

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import Graph, END
from src.services.notion_service import query_notion
from src.services.airtable_service import query_airtable
from src.state.state_manager import StateManager
from src.agents.pkm_agent import PKMAgent

class ResearchWorkflow:
    def __init__(self):
        self.state_manager = StateManager()
        self.pkm_agent = PKMAgent()
        self.pkm_agent.set_workflow(self)
        self.state_manager.set_state("pkm_agent", self.pkm_agent)
        self.workflow = self._define_workflow()

        # Configuration for Notion databases and Airtable bases/tables
        self.notion_databases = [
            os.getenv("NOTION_DATABASE_ID"),
            # Add more Notion database IDs here
        ]
        self.airtable_bases = {
            os.getenv("AIRTABLE_BASE_ID"): [
                os.getenv("AIRTABLE_TABLE_ID"),
                # Add more table IDs for this base here
            ],
            # Add more base IDs and their corresponding table IDs here
        }

    def _define_workflow(self):
        workflow = Graph()

        workflow.add_node("research_query", self.research_query)
        workflow.add_node("analyze_results", self.analyze_results)
        workflow.add_node("generate_response", self.generate_response)

        workflow.set_entry_point("research_query")

        workflow.add_edge("research_query", "analyze_results")
        workflow.add_edge("analyze_results", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def research_query(self, state):
        query = state["query"]
        notion_results = query_notion(query, databases=self.notion_databases)
        airtable_results = query_airtable(query, bases=self.airtable_bases)
        personal_data_results = self.pkm_agent.query_personal_data(query)

        combined_results = f"{notion_results}\n\n{airtable_results}\n\n{personal_data_results}"
        return {"query": query, "initial_results": combined_results}

    def analyze_results(self, state):
        query = state["query"]
        results = state["initial_results"]

        prompt = PromptTemplate.from_template(
        "Analyze and synthesize the following research results for the query: {query}\n\n"
        "Results:\n{results}\n\n"
        "Provide a structured report with the following sections:\n"
        "1. References: List all sources of information (Notion, Airtable, personal data) with clear attribution.\n"
        "2. Synthesis: Combine and summarize the information from all sources to address the user query.\n"
        "3. Supplemental Information: Add any relevant information from your general knowledge that complements the query results.\n"
        "Ensure each piece of information is clearly attributed to its source."
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        analysis = llm.invoke(prompt.format(query=query, results=results))

        return {"query": query, "analysis": analysis.content}

    def generate_response(self, state):
        analysis = state["analysis"]

        prompt = PromptTemplate.from_template(
        "Based on the following analysis, generate a final response for the user:\n\n"
        "{analysis}\n\n"
        "Structure your response as follows:\n"
        "1. TLDR: A brief summary of the key points.\n"
        "2. References: List all sources of information used (Notion, Airtable, your general knowledge).\n"
        "3. Detailed Content: Provide a comprehensive answer to the user's query, clearly attributing information to its source.\n"
        "4. Additional Insights: Include any supplemental information from your general knowledge that enhances the response."
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt.format(analysis=analysis))

        return {"response": response.content}

    def run(self, inputs):
        return self.workflow.invoke(inputs)
