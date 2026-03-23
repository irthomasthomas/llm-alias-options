# llm-alias-options

LLM plugin to save prompt options alongside an alias.

Built for `llm` version 0.28.

This plugin allows you to save default prompt options (like temperature, max_tokens, etc.) with an alias, so they are automatically applied when you use that alias.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-alias-options
```

## Usage

You can save default prompt options with an alias using the `-o` or `--option` flag with `llm aliases set`:

```bash
llm aliases set creative gpt-4o -o temperature 0.9
```

Now, whenever you use the `creative` alias, the `temperature=0.9` option will be applied automatically:

```bash
llm -m creative "Write a story"
```

The options you specify on the command line will override the saved alias options:

```bash
llm -m creative -o temperature 0.5 "Write a story"
```

### Listing aliases with options

To see which aliases have options configured, use the `--options` flag:

```bash
llm aliases list --options
```

Or just:

```bash
llm aliases --options
```

Output:
```
creative: gpt-4o
  Options:
    temperature: 0.9
```

You can also get this information in JSON format:

```bash
llm aliases --options --json
```

## How it works

This plugin monkeypatches the core `llm` alias handling to support a dictionary format in `aliases.json`. It also wraps the `prompt` and `chat` commands to intercept the model ID and inject the saved options before the command executes.
