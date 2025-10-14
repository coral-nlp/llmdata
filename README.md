# LLMData

A framework for large-scale LLM data preprocessing using [Ray Data](https://docs.ray.io/en/latest/data/data.html).  Think `datatrove` or `dolma` (in fact, it borrows a lot from those), but with native Ray integration.

## Quick Start

### Configuration

Create a YAML pipeline configuration:

```yaml
name: "language_filtering"
description: "An example pipeline that filters input data by language, keeping only English text."
input:
  format: "jsonl"
  path: "data/input.jsonl"

processors:
  - category: "tagger"
    type: "language"
    params: {}
  - category: "filter"
    type: "language"
    params:
      allowed_languages: ["en"]

output:
  format: "parquet"
  path: "data/output.parquet"
```

### Usage

```python
from llmdata import DataPipeline, PipelineConfig

config = PipelineConfig.from_yaml("config.yaml")
pipeline = DataPipeline(config)
pipeline.run()
```

## CLI Interface

You can invoke a processing pipeline using the CLI:

```bash
llmdata run config.yaml
```

Full options:
```bash
llmdata -h
usage: llmdata [-h] [--config CONFIG] [--print_config[=flags]] {export_schemas,list,run,validate} ...

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by
                        comma. The supported flags are: comments, skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    export_schemas      Writes all available processor config schemas to a JSON file.
    list                Print all available processors categories and names.
    run                 Run a pipeline using a config file.
    validate            Validate a config file.
```

## Built-in Processors

```bash
$ llmdata list
=== Available Components ===

aggregation:
  - sum
  - count
  - quantile
  - mean
  - min
  - max
  - std
  - absmax
  - unique

reader:
  - parquet
  - jsonl
  - csv
  - text

writer:
  - parquet
  - jsonl
  - csv

extractor:
  - html
  - tei
  - plain

filter:
  - language
  - gopher_quality
  - gopher_repetition
  - num_tokens
  - value
  - exists

formatter:
  - deduplication
  - ftfy
  - ocr_error
  - pii

tagger:
  - language
  - gopher_quality
  - gopher_repetition
  - token_count
  - length
  - value
```

## Development

```bash
# Setup
uv sync --dev
uv run pre-commit install

# Testing
uv run pytest

# Code quality
make check
```

