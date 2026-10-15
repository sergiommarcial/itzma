# Examples

Small scripts, each runnable on its own against an in-memory store. From `py-itzma/`, after `make py-dev`:

```sh
uv run python examples/quickstart.py
```

- `quickstart.py`: create a namespace, write, query with a filter, open it again.
- `filters.py`: every kind of condition, the null rule, and paging with `filter()`.
- `bulk_load.py`: loading from numpy, pyarrow and Polars, and a keyed write that is safe to retry.
- `sparse.py`: sparse (term id, weight) vectors.
- `maintenance.py`: `compact`, `index`, `gc` and `maintain` by hand, then the background maintainer.
- `asyncio_queries.py`: `AsyncNamespace` with many queries in flight.

To try one against S3 or a directory, change the `itzma.Store("memory://")` line. A `file://` store cannot compact or index, since it has no conditional update.

`tests/test_examples.py` runs every script here, so `make py` fails if one breaks.
