import pytest
from click.testing import CliRunner
import json
import llm
from llm.cli import cli
import llm_alias_options
from llm.plugins import pm
from pydantic import Field
from typing import Optional
import importlib

class MockModel(llm.Model):
    model_id = "mock"
    attachment_types = {"image/png", "audio/wav"}
    supports_schema = True
    supports_tools = True

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate.", default=None
        )
        temperature: Optional[str] = Field(
            description="Temperature", default=None
        )

    def __init__(self):
        self.history = []
        self._queue = []
        self.resolved_model_name = None

    def enqueue(self, messages):
        assert isinstance(messages, list)
        self._queue.append(messages)

    def execute(self, prompt, stream, response, conversation):
        self.history.append((prompt, stream, response, conversation))
        yield "Response from mock"
        response.set_usage(input=1, output=1)

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture(autouse=True)
def register_mock_model(mock_model):
    class MockModelsPlugin:
        __name__ = "MockModelsPlugin"
        @llm.hookimpl
        def register_models(self, register):
            register(mock_model)
    
    pm.register(MockModelsPlugin(), name="test-mock-models")
    try:
        yield
    finally:
        pm.unregister(name="test-mock-models")

@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir

@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))

@pytest.fixture(autouse=True)
def remove_core_features(monkeypatch):
    import llm
    import llm.cli
    # Remove core features if they exist to test the plugin's implementation
    if hasattr(llm, "resolve_alias_options"):
        monkeypatch.delattr(llm, "resolve_alias_options", raising=False)
    if hasattr(llm, "set_alias_with_options"):
        monkeypatch.delattr(llm, "set_alias_with_options", raising=False)
    if hasattr(llm, "get_aliases_with_options"):
        monkeypatch.delattr(llm, "get_aliases_with_options", raising=False)
    
    # Also strip the options from the CLI commands if they are present in core
    from llm.cli import cli
    aliases_cmd = cli.commands.get("aliases")
    if aliases_cmd:
        set_cmd = aliases_cmd.commands.get("set")
        if set_cmd:
            set_cmd.params = [p for p in set_cmd.params if p.name not in ("option", "options")]
        list_cmd = aliases_cmd.commands.get("list")
        if list_cmd:
            list_cmd.params = [p for p in list_cmd.params if p.name != "options"]
    
    # Re-apply plugin logic
    importlib.reload(llm_alias_options)

@pytest.fixture(autouse=True)
def register_plugin(remove_core_features):
    # Register the plugin hooks with llm's plugin manager
    if not pm.get_plugin("llm-alias-options-test"):
        pm.register(llm_alias_options, name="llm-alias-options-test")
    # Call the hook manually to ensure commands are patched even if cli was already loaded
    llm_alias_options.register_commands(cli)
    yield

def test_cli_aliases_set_with_options(user_path):
    """Test setting aliases with options via CLI"""
    runner = CliRunner()
    
    # Test setting alias with options
    result = runner.invoke(cli, [
        "aliases", "set", "test-alias", "mock", 
        "-o", "temperature", "0.5",
        "-o", "max_tokens", "100"
    ])
    assert result.exit_code == 0
    
    # Check that aliases.json contains the correct structure
    aliases_file = user_path / "aliases.json"
    assert aliases_file.exists()
    aliases_data = json.loads(aliases_file.read_text("utf-8"))
    
    expected = {
        "test-alias": {
            "model": "mock",
            "options": {
                "temperature": "0.5",
                "max_tokens": "100"
            }
        }
    }
    assert aliases_data == expected

def test_cli_aliases_set_with_query_and_options(user_path):
    """Test setting aliases with -q query and options"""
    runner = CliRunner()
    
    # Test setting alias with query and options
    result = runner.invoke(cli, [
        "aliases", "set", "test-query-alias", 
        "-q", "mock",
        "-o", "temperature", "0.7"
    ])
    assert result.exit_code == 0
    
    # Check that aliases.json contains the correct structure
    aliases_file = user_path / "aliases.json"
    assert aliases_file.exists()
    aliases_data = json.loads(aliases_file.read_text("utf-8"))
    
    # It should have resolved 'mock' to 'mock' (since it's a registered model in conftest)
    expected = {
        "test-query-alias": {
            "model": "mock",
            "options": {
                "temperature": "0.7"
            }
        }
    }
    assert aliases_data["test-query-alias"] == expected["test-query-alias"]

def test_set_alias_with_options_function(user_path):
    """Test the set_alias_with_options function directly"""
    # Test the function directly
    llm.set_alias_with_options("direct-test", "mock", {
        "temperature": 0.3,
        "max_tokens": 50
    })
    
    # Check that aliases.json was created correctly
    aliases_file = user_path / "aliases.json"
    assert aliases_file.exists()
    aliases_data = json.loads(aliases_file.read_text("utf-8"))
    
    expected = {
        "direct-test": {
            "model": "mock",
            "options": {
                "temperature": 0.3,
                "max_tokens": 50
            }
        }
    }
    assert aliases_data["direct-test"] == expected["direct-test"]

def test_resolve_alias_options_function(user_path):
    """Test the resolve_alias_options function"""
    # First set an alias with options
    llm.set_alias_with_options("resolve-test", "mock", {
        "temperature": 0.8
    })
    
    # Test resolving the alias
    result = llm.resolve_alias_options("resolve-test")
    expected = {
        "model": "mock",
        "options": {
            "temperature": 0.8
        }
    }
    assert result == expected
    
    # Test resolving a non-alias (should return None)
    result = llm.resolve_alias_options("not-an-alias")
    assert result is None

def test_cli_aliases_list_with_options_flag(user_path):
    """Test the --options flag for aliases list command"""
    # Set up aliases - one with options, one without
    llm.set_alias("simple-alias", "mock")
    llm.set_alias_with_options("opts-alias", "mock", {
        "temperature": "0.5",
        "max_tokens": "100"
    })
    
    runner = CliRunner()
    
    # Test --options flag shows only aliases with options
    result = runner.invoke(cli, ["aliases", "--options"])
    assert result.exit_code == 0
    assert "opts-alias: mock" in result.output
    assert "temperature: 0.5" in result.output
    assert "max_tokens: 100" in result.output
    # Simple alias should NOT appear
    assert "simple-alias" not in result.output
    
    # Test --options with --json
    result = runner.invoke(cli, ["aliases", "--options", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "opts-alias" in data
    assert data["opts-alias"]["model"] == "mock"
    assert data["opts-alias"]["options"]["temperature"] == "0.5"
    assert "simple-alias" not in data

def test_cli_aliases_list_options_empty(user_path):
    """Test --options flag when no aliases have options"""
    # Only set a simple alias without options
    llm.set_alias("simple-only", "mock")
    
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "--options"])
    assert result.exit_code == 0
    assert "No aliases with options found" in result.output

def test_prompt_applies_alias_options(user_path, mock_model):
    """Test that prompt command applies options from the alias"""
    runner = CliRunner()
    
    # Set alias with options
    llm.set_alias_with_options("creative", "mock", {"temperature": "0.9"})
    
    # Run prompt
    result = runner.invoke(cli, ["-m", "creative", "hello"])
    assert result.exit_code == 0
    
    # Verify that the mock model received the option
    # mock_model.history contains (prompt, stream, response, conversation)
    # prompt is an llm.Prompt object which has an 'options' attribute
    last_prompt = mock_model.history[0][0]
    # In llm, options are usually an Options object or a dict depending on how it's resolved
    assert last_prompt.options.dict()["temperature"] == "0.9"

def test_prompt_override_alias_options(user_path, mock_model):
    """Test that CLI options override alias options"""
    runner = CliRunner()
    
    # Set alias with options
    llm.set_alias_with_options("creative", "mock", {"temperature": "0.9"})
    
    # Run prompt with override
    result = runner.invoke(cli, ["-m", "creative", "-o", "temperature", "0.5", "hello"])
    assert result.exit_code == 0
    
    last_prompt = mock_model.history[0][0]
    assert last_prompt.options.dict()["temperature"] == "0.5"
