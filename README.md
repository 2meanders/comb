# comb

comb is a command-line indexing and search tool for local file collections. It walks a folder tree, extracts text from supported file formats, stores that text in a local index, and then lets you search the indexed content quickly.

The project is designed for archive-style search workflows: rather than relying only on filenames, it attempts to recover text from file contents wherever possible and make that content searchable.

## What comb supports

comb can extract and index text from:

- plain text and source files
- PDF documents
- image files via OCR
- Microsoft Word, Excel, and PowerPoint files
- audio files via speech-to-text transcription
- ZIP archives and nested archive members

The extraction pipeline is implemented in [extractors.py](extractors.py), which dispatches by file extension and uses format-specific handlers when available.

## How it works

The workflow is roughly:

1. Walk the target folder and discover files.
2. Extract text from each supported file using the best available extractor.
3. Store the extracted text and metadata in a local SQLite-backed index.
4. Search the index using full-text search or regex-based matching.

The index is created and maintained by [index.py](index.py), while [search_engine.py](search_engine.py) handles query execution and output formatting.

## Installation

This project requires Python 3 and the packages listed in [requirements.txt](requirements.txt).

Install them with:

```bash
pip install -r requirements.txt
```

Depending on your environment, some extractors may require additional system dependencies. For example, OCR and audio transcription features may depend on external tools or runtime libraries being available.

## Usage

From the repository root, the CLI is invoked as:

```bash
python comb.py <command> [options]
```

### Build an index

```bash
python comb.py build --folder "path/to/folder"
```

This scans the selected folder and builds or updates the local index.

### Search an indexed folder

```bash
python comb.py search "invoice" --folder "path/to/folder"
```

This searches the indexed content for the provided term. The default search mode is automatic, but you can also choose full-text search or regex mode explicitly.

### Rebuild from scratch

```bash
python comb.py rebuild --folder "path/to/folder"
```

### Clear the index

```bash
python comb.py clear --folder "path/to/folder"
```

## CLI notes

The tool supports a few useful flags:

- `--folder`, `-f`: target directory to index or search
- `--verbose`: print more detailed progress output
- `--include-hidden`, `-a`: include hidden and ignored paths
- `--context`, `-c`: number of characters to show around a match
- `--no-update`: skip updating the index before a search
- `--clear`: remove the index after the search completes

## Example

```bash
python comb.py build --folder "Invoices"
python comb.py search "Acme" --folder "Invoices"
```

## Contributing

Contributions are welcome. If you would like to improve the project, please open an issue or submit a pull request.

## License

See the [LICENSE](LICENSE) file in this repository.
