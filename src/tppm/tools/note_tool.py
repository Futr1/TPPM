"""TPM-backed note tools preserving Mini-Agent's original tool API."""

from pathlib import Path
from typing import Any

from tppm.core.memory import TPMMemoryManager

from .base import Tool, ToolResult


class SessionNoteTool(Tool):
    """Tool for recording important memories into TPM."""

    def __init__(self, memory_file: str = "./workspace/.agent_memory.json", memory_manager: TPMMemoryManager | None = None):
        """Initialize note tool backed by TPM.

        Args:
            memory_file: TPM persistence file path
            memory_manager: Optional shared TPM manager
        """
        self.memory_file = Path(memory_file)
        self.memory_manager = memory_manager or TPMMemoryManager(memory_file=self.memory_file)

    @property
    def name(self) -> str:
        return "record_note"

    @property
    def description(self) -> str:
        return (
            "Record important information into Temporal Profile Memory (TPM) for future reference. "
            "Use this to save explicit user preferences, project facts, decisions, or stable profile information."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to record as a note. Be concise but specific.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category/tag for this note (e.g., 'user_preference', 'project_info', 'decision')",
                },
            },
            "required": ["content"],
        }

    async def execute(self, content: str, category: str = "general") -> ToolResult:
        """Record explicit information into TPM."""
        try:
            accepted = self.memory_manager.record_manual(content=content, category=category)
            summary = ", ".join(f"{item.attribute}={item.value}" for item in accepted) if accepted else content

            return ToolResult(
                success=True,
                content=f"Recorded TPM note: {summary} (category: {category})",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to record TPM note: {str(e)}",
            )


class RecallNoteTool(Tool):
    """Tool for recalling stored TPM memories."""

    def __init__(self, memory_file: str = "./workspace/.agent_memory.json", memory_manager: TPMMemoryManager | None = None):
        """Initialize recall tool backed by TPM.

        Args:
            memory_file: TPM persistence file path
            memory_manager: Optional shared TPM manager
        """
        self.memory_file = Path(memory_file)
        self.memory_manager = memory_manager or TPMMemoryManager(memory_file=self.memory_file)

    @property
    def name(self) -> str:
        return "recall_notes"

    @property
    def description(self) -> str:
        return (
            "Recall information stored in Temporal Profile Memory (TPM). "
            "Use this to inspect prior user preferences, project facts, context, and stable profile knowledge."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional: filter notes by category",
                },
            },
        }

    async def execute(self, category: str = None) -> ToolResult:
        """Recall TPM memories."""
        try:
            result = self.memory_manager.format_recall(category=category)
            return ToolResult(success=True, content=result)

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to recall TPM notes: {str(e)}",
            )
