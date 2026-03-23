import llm
from llm import user_dir, UnknownModelError, get_model, get_embedding_model, get_models_with_aliases
import json
import click
import functools
import sys

# Implementation of helper functions from PR #1324
def set_alias_with_options(alias, model_id_or_alias, options):
    path = user_dir() / "aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}\n")
    try:
        current = json.loads(path.read_text())
    except json.decoder.JSONDecodeError:
        current = {}
    
    try:
        model = get_model(model_id_or_alias)
        model_id = model.model_id
    except Exception:
        try:
            model = get_embedding_model(model_id_or_alias)
            model_id = model.model_id
        except Exception:
            model_id = model_id_or_alias
            
    current[alias] = {
        "model": model_id,
        "options": options
    }
    path.write_text(json.dumps(current, indent=4) + "\n")

def resolve_alias_options(alias_or_model_id):
    path = user_dir() / "aliases.json"
    if not path.exists():
        return None
    try:
        current = json.loads(path.read_text())
    except json.decoder.JSONDecodeError:
        return None
    
    if alias_or_model_id not in current:
        return None
    
    alias_value = current[alias_or_model_id]
    if isinstance(alias_value, dict) and "model" in alias_value:
        return alias_value
    return None

def get_aliases_with_options():
    path = user_dir() / "aliases.json"
    if not path.exists():
        return {}
    try:
        current = json.loads(path.read_text())
    except json.decoder.JSONDecodeError:
        return {}
    
    result = {}
    for alias, value in current.items():
        if isinstance(value, dict) and "model" in value and "options" in value:
            if value["options"]:
                result[alias] = value
    return result

# Monkeypatch llm module if functions are missing
if not hasattr(llm, "resolve_alias_options"):
    llm.resolve_alias_options = resolve_alias_options
if not hasattr(llm, "set_alias_with_options"):
    llm.set_alias_with_options = set_alias_with_options
if not hasattr(llm, "get_aliases_with_options"):
    llm.get_aliases_with_options = get_aliases_with_options

# Monkeypatch get_models_with_aliases to handle dict format in aliases.json
def patched_get_models_with_aliases():
    from llm import ModelWithAliases, load_plugins, pm
    model_aliases = []
    aliases_path = user_dir() / "aliases.json"
    extra_model_aliases = {}
    if aliases_path.exists():
        try:
            configured_aliases = json.loads(aliases_path.read_text())
            for alias, model_id_or_config in configured_aliases.items():
                if isinstance(model_id_or_config, dict) and "model" in model_id_or_config:
                    model_id = model_id_or_config["model"]
                else:
                    model_id = model_id_or_config
                extra_model_aliases.setdefault(model_id, []).append(alias)
        except Exception:
            pass

    def register(model, async_model=None, aliases=None):
        alias_list = list(aliases or [])
        if model.model_id in extra_model_aliases:
            alias_list.extend(extra_model_aliases[model.model_id])
        model_aliases.append(ModelWithAliases(model, async_model, alias_list))

    load_plugins()
    pm.hook.register_models(register=register)
    return model_aliases

llm.get_models_with_aliases = patched_get_models_with_aliases

def patched_get_embedding_models_with_aliases():
    from llm import EmbeddingModelWithAliases, load_plugins, pm
    model_aliases = []
    aliases_path = user_dir() / "aliases.json"
    extra_model_aliases = {}
    if aliases_path.exists():
        try:
            configured_aliases = json.loads(aliases_path.read_text())
            for alias, model_id_or_config in configured_aliases.items():
                if isinstance(model_id_or_config, dict) and "model" in model_id_or_config:
                    model_id = model_id_or_config["model"]
                else:
                    model_id = model_id_or_config
                extra_model_aliases.setdefault(model_id, []).append(alias)
        except Exception:
            pass

    def register(model, aliases=None):
        alias_list = list(aliases or [])
        if model.model_id in extra_model_aliases:
            alias_list.extend(extra_model_aliases[model.model_id])
        model_aliases.append(EmbeddingModelWithAliases(model, alias_list))

    load_plugins()
    pm.hook.register_embedding_models(register=register)
    return model_aliases

llm.get_embedding_models_with_aliases = patched_get_embedding_models_with_aliases

@llm.hookimpl
def register_commands(cli):
    # Wrap 'aliases set'
    aliases_cmd = cli.commands.get("aliases")
    if aliases_cmd:
        set_cmd = aliases_cmd.commands.get("set")
        if set_cmd:
            # Check if it already has the option to avoid duplicate params
            if not any(p.name == "option" for p in set_cmd.params):
                option_param = click.Option(
                    ["-o", "--option"],
                    type=(str, str),
                    multiple=True,
                    help="Options to include with the alias",
                )
                set_cmd.params.append(option_param)
            
            original_callback = set_cmd.callback
            @functools.wraps(original_callback)
            def new_aliases_set(alias, model_id, query, option=None, **kwargs):
                # In some versions, 'option' might already be in kwargs if core has it
                if option is None:
                    option = kwargs.pop("option", None)
                
                if option:
                    if query and not model_id:
                        # Search for the first model matching all query strings
                        found = None
                        for model_with_aliases in llm.get_models_with_aliases():
                            if all(model_with_aliases.matches(q) for q in query):
                                found = model_with_aliases
                                break
                        if not found:
                             raise click.ClickException("No model found matching query: " + ", ".join(query))
                        model_id = found.model.model_id
                    
                    llm.set_alias_with_options(alias, model_id, dict(option))
                    click.echo(f"Alias '{alias}' set to model '{model_id}' with options", err=True)
                else:
                    return original_callback(alias=alias, model_id=model_id, query=query, **kwargs)
            
            set_cmd.callback = new_aliases_set

        list_cmd = aliases_cmd.commands.get("list")
        if list_cmd:
            if not any(p.name == "options" for p in list_cmd.params):
                options_param = click.Option(
                    ["--options"], is_flag=True, help="Show only aliases with options"
                )
                list_cmd.append_params(options_param) if hasattr(list_cmd, 'append_params') else list_cmd.params.append(options_param)
            
            original_list_callback = list_cmd.callback
            @functools.wraps(original_list_callback)
            def new_list_callback(json_, options=False, **kwargs):
                if options is False:
                    options = kwargs.pop("options", False)
                if options:
                    aliases_with_opts = llm.get_aliases_with_options()
                    if json_:
                        click.echo(json.dumps(aliases_with_opts, indent=4))
                        return
                    if not aliases_with_opts:
                        click.echo("No aliases with options found")
                        return
                    for alias, config in aliases_with_opts.items():
                        click.echo(f"{alias}: {config['model']}")
                        click.echo("  Options:")
                        for opt_name, opt_value in config["options"].items():
                            click.echo(f"    {opt_name}: {opt_value}")
                    return
                return original_list_callback(json_=json_, **kwargs)
            list_cmd.callback = new_list_callback

    # Wrap 'prompt' and 'chat'
    def make_wrapper(original_callback):
        @functools.wraps(original_callback)
        def wrapped_callback(*args, **kwargs):
            # model_id or model depends on the command
            model_id = kwargs.get("model_id") or kwargs.get("model")
            options = kwargs.get("options") or kwargs.get("option")
            
            if not model_id and not kwargs.get("queries"):
                from llm import get_default_model
                model_id = get_default_model()
            
            if model_id:
                alias_options = llm.resolve_alias_options(model_id)
                if alias_options:
                    # Update the correct key
                    if "model_id" in kwargs:
                        kwargs["model_id"] = alias_options["model"]
                    elif "model" in kwargs:
                        kwargs["model"] = alias_options["model"]
                        
                    merged_options = alias_options.get("options", {}).copy()
                    if options:
                        merged_options.update(dict(options))
                    
                    if "options" in kwargs:
                        kwargs["options"] = list(merged_options.items())
                    elif "option" in kwargs:
                        kwargs["option"] = list(merged_options.items())
                    else:
                        kwargs["options"] = list(merged_options.items())
            
            return original_callback(*args, **kwargs)
        return wrapped_callback

    for cmd_name in ("prompt", "chat"):
        cmd = cli.commands.get(cmd_name)
        if cmd:
            cmd.callback = make_wrapper(cmd.callback)
