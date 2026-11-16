from pathlib import Path

from extractor.agent import ToolAgent


def test_agent_calculates_common_math_requests():
    answer = ToolAgent().run("Add 12 and 8, then multiply by 2")
    assert "Result: 40" in answer


def test_agent_reads_local_file_preview(tmp_path: Path):
    sample = tmp_path / "notes.txt"
    sample.write_text("Alpha\nBeta\nGamma\n", encoding="utf-8")

    answer = ToolAgent().run(f"Read file {sample}")
    assert "Alpha" in answer and "Beta" in answer
