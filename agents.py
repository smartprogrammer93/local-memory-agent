"""
Agent definitions and orchestrator for the memory agent system.

Extracts agent configuration from the monolithic agent.py into
reusable components: prompt constants, a builder function, and
a keyword-based routing orchestrator.
"""

from openai import AsyncOpenAI

from llm import QwenAgent
from tools import (
    store_memory,
    read_all_memories,
    read_unconsolidated_memories,
    store_consolidation,
    read_consolidation_history,
)

# ─── System Prompts ───────────────────────────────────────────

INGEST_PROMPT = (
    "You are a Memory Ingest Agent. For any input you receive:\n"
    "1. Create a concise 1-2 sentence summary\n"
    "2. Extract key entities (people, companies, products, concepts)\n"
    "3. Assign 2-4 topic tags\n"
    "4. Rate importance from 0.0 to 1.0\n"
    "5. Call store_memory with all extracted information\n\n"
    "Use the input text as raw_text in store_memory.\n"
    "Always call store_memory. Be concise and accurate.\n"
    "After storing, confirm what was stored in one sentence."
)

CONSOLIDATE_PROMPT = (
    "You are a Memory Consolidation Agent. You:\n"
    "1. Call read_unconsolidated_memories to see what needs processing\n"
    "2. If fewer than 2 memories, say nothing to consolidate\n"
    "3. Find connections and patterns across the memories\n"
    "4. Create a synthesized summary and one key insight\n"
    "5. Call store_consolidation with source_ids, summary, insight, and connections\n\n"
    "Connections: list of dicts with 'from_id', 'to_id', 'relationship' keys.\n"
    "Think deeply about cross-cutting patterns."
)

QUERY_PROMPT = (
    "You are a Memory Query Agent. When asked a question:\n"
    "1. Call read_all_memories to access the memory store\n"
    "2. Call read_consolidation_history for higher-level insights\n"
    "3. Synthesize an answer based ONLY on stored memories\n"
    "4. Reference memory IDs: [Memory 1], [Memory 2], etc.\n"
    "5. If no relevant memories exist, say so honestly\n\n"
    "Be thorough but concise. Always cite sources."
)

# ─── Agent Builder ────────────────────────────────────────────


def build_agents(client: AsyncOpenAI, model: str) -> dict[str, QwenAgent]:
    """Create the three memory agents with their tools and prompts.

    Args:
        client: An AsyncOpenAI client configured for the Qwen API.
        model: The model identifier to use for completions.

    Returns:
        dict mapping agent names to configured QwenAgent instances.
    """
    return {
        "ingest": QwenAgent(
            name="ingest_agent",
            system_prompt=INGEST_PROMPT,
            tools=[store_memory],
            client=client,
            model=model,
        ),
        "consolidate": QwenAgent(
            name="consolidate_agent",
            system_prompt=CONSOLIDATE_PROMPT,
            tools=[read_unconsolidated_memories, store_consolidation],
            client=client,
            model=model,
        ),
        "query": QwenAgent(
            name="query_agent",
            system_prompt=QUERY_PROMPT,
            tools=[read_all_memories, read_consolidation_history],
            client=client,
            model=model,
        ),
    }


# ─── Orchestrator ─────────────────────────────────────────────

# Keywords that route to each agent, checked in order.
_CONSOLIDATE_KEYWORDS = ("consolidate", "merge", "patterns")
_INGEST_KEYWORDS = ("remember", "ingest", "store", "file", "analyze")
_QUERY_KEYWORDS = ("what", "answer", "tell me", "based on", "query")


class MemoryOrchestrator:
    """Routes incoming messages to the appropriate memory agent."""

    def __init__(
        self,
        agents: dict[str, QwenAgent],
        client: AsyncOpenAI,
        model: str,
    ):
        self.agents = agents
        self.client = client
        self.model = model

    async def route(self, message: str, content: str | None = None) -> str:
        """Route a message to the correct agent based on keywords.

        Args:
            message: The user message / command to route.
            content: Optional content payload (passed to ingest agent).

        Returns:
            The agent's response string.
        """
        lower = message.lower()

        if any(kw in lower for kw in _CONSOLIDATE_KEYWORDS):
            return await self.agents["consolidate"].run(
                "Consolidate unconsolidated memories. Find connections and patterns."
            )

        if any(kw in lower for kw in _INGEST_KEYWORDS):
            text = content if content else message
            return await self.agents["ingest"].run(
                f"Remember this information:\n\n{text}"
            )

        if any(kw in lower for kw in _QUERY_KEYWORDS):
            return await self.agents["query"].run(
                f"Based on my memories, answer: {message}"
            )

        # Fallback → query agent
        return await self.agents["query"].run(
            f"Based on my memories, answer: {message}"
        )
