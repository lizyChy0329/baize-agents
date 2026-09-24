"""系统提示词组装的测试。"""
from __future__ import annotations

from pathlib import Path

from baize_agents.prompt import DEFAULT_IDENTITY, build_system_prompt


def test_default_prompt_contains_identity():
    prompt = build_system_prompt()
    assert DEFAULT_IDENTITY.strip() in prompt


def test_default_prompt_tells_model_to_use_tools():
    """这是加系统提示词的主要目的：减少「该调工具却不调」。"""
    prompt = build_system_prompt()
    assert "主动调用" in prompt


def test_environment_block_includes_cwd():
    prompt = build_system_prompt()
    assert str(Path.cwd()) in prompt
    assert "工作目录" in prompt


def test_environment_block_lists_registered_tools():
    prompt = build_system_prompt()
    assert "file_read" in prompt
    assert "可用工具" in prompt


def test_custom_prompt_replaces_identity():
    prompt = build_system_prompt("你是一个只说实话的助手。")
    assert "你是一个只说实话的助手。" in prompt
    assert DEFAULT_IDENTITY.strip() not in prompt


def test_custom_prompt_still_gets_environment():
    """自定义只替换身份部分，环境信息（模型无法自知的事实）仍会附加。"""
    prompt = build_system_prompt("你是一个只说实话的助手。")
    assert str(Path.cwd()) in prompt
    assert "file_read" in prompt


def test_blank_custom_falls_back_to_default():
    assert DEFAULT_IDENTITY.strip() in build_system_prompt("   ")
    assert DEFAULT_IDENTITY.strip() in build_system_prompt("")
