# OpenRouter to Langfuse Model Pricing Sync

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python script that automatically syncs model pricing data from OpenRouter to Langfuse's pricing page. This tool fetches the latest pricing information from OpenRouter's API and creates corresponding models in Langfuse with accurate pricing data.

You might also be interested in my other script: [LiteLLM Proxy OpenRouter Price Updater](https://github.com/thiswillbeyourgithub/litellm_proxy_openrouter_price_updated).

This script was created with assistance from [aider.chat](https://github.com/Aider-AI/aider/).

## Features

- 🔄 **Automatic Sync**: Fetches pricing data from OpenRouter and creates models in Langfuse
- 🧹 **Clean Management**: Automatically deletes previously created models before adding new ones
- 🏃 **Continuous Operation**: Safe to run repeatedly - handles updates gracefully
- 🔍 **Dry Run Mode**: Test operations without making actual changes
- 🗑️ **Reset Mode**: Option to only delete script-managed models without creating new ones
- ⚡ **Rate Limiting**: Built-in rate limiting to respect API limits
- 📊 **Progress Tracking**: Visual progress bars and detailed logging
- 🔧 **Flexible Configuration**: Environment variables or CLI arguments

## Compatibility

- ✅ Tested with **Langfuse v3**
- 🐍 Python 3.7+

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install click requests tqdm loguru
```

## Configuration

The script requires Langfuse credentials to sync models. You can provide these via environment variables or CLI arguments.

### Environment Variables

```bash
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"
export LANGFUSE_HOST="https://your-langfuse-instance.com"
```

### CLI Arguments

Alternatively, pass credentials directly:
```bash
python script.py --langfuse-public-key="your_key" --langfuse-secret-key="your_secret" --langfuse-host="https://your-instance.com"
```

## Usage

### Basic Sync
Sync all OpenRouter models to Langfuse:
```bash
python script.py
```

### Dry Run
Test the sync without making actual changes:
```bash
python script.py --dry
```

### Reset Mode
Delete only the script-managed models (useful for cleanup):
```bash
python script.py --reset
```

### Custom Start Date
Set a specific start date for model pricing (dd-mm-yyyy format):
```bash
python script.py --start-date="01-12-2024"
```

### Rate Limiting
Add delays between API calls to respect rate limits:
```bash
python script.py --rate-limit-delay=0.5
```

## How It Works

1. **Fetch Data**: Retrieves current model pricing from OpenRouter API
2. **Clean Slate**: Identifies and deletes previously created models (those starting with `openrouter_script_`)
3. **Create Models**: Creates new Langfuse models with current pricing data
4. **Smart Naming**: Handles duplicate model names and special cases (e.g., "thinking" models)
5. **Tokenizer Detection**: Automatically assigns appropriate tokenizers based on model provider

## Model Management

The script creates models in Langfuse with:
- **Naming Pattern**: `openrouter_script_{canonical_slug}`
- **Match Pattern**: Regex to match the OpenRouter model ID
- **Pricing**: Input/output token prices from OpenRouter
- **Tokenizer**: Automatically selected (Claude for Anthropic models, OpenAI for others)

### Model Identification

Script-managed models are identified by:
- Model name starts with `openrouter_script_`
- `isLangfuseManaged` is set to `False`

This ensures the script only manages its own models and won't interfere with manually created ones.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--langfuse-public-key` | Langfuse public key | From `LANGFUSE_PUBLIC_KEY` env var |
| `--langfuse-secret-key` | Langfuse secret key | From `LANGFUSE_SECRET_KEY` env var |
| `--langfuse-host` | Langfuse host URL | From `LANGFUSE_HOST` env var |
| `--dry` | Dry run mode (no actual changes) | `False` |
| `--reset` | Only delete script-managed models | `False` |
| `--start-date` | Start date for models (dd-mm-yyyy) | Today's date |
| `--rate-limit-delay` | Delay between API calls (seconds) | `0` |

## Logging

The script creates detailed logs in `script.log` with:
- Automatic rotation (10 MB files)
- 7-day retention
- INFO level and above

## Continuous Operation

This script is designed for continuous operation:

- **Safe Re-runs**: Each execution cleans up its previous models before creating new ones
- **No Conflicts**: Only manages models it created (identified by naming pattern)
- **Incremental Updates**: Handles pricing changes and new models automatically

You can safely run this script on a schedule (e.g., daily) to keep your Langfuse pricing data up-to-date with OpenRouter.

## Error Handling

- **API Failures**: Graceful handling of network issues and API errors
- **Rate Limiting**: Built-in delays to respect API limits
- **Data Validation**: Skips models with missing or invalid pricing data
- **Detailed Logging**: Comprehensive error reporting and debugging information

## Contributing

Issues and pull requests are welcome! Please feel free to:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit improvements
- 📖 Improve documentation

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Created with assistance from [aider.chat](https://github.com/Aider-AI/aider/)
- Uses [OpenRouter API](https://openrouter.ai/docs) for model pricing data
- Integrates with [Langfuse](https://langfuse.com/) for LLM observability and pricing management
