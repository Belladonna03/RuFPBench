# Progress bars for cascade mining

- Enabled tqdm rendering in `utils.map_parallel(..., desc=...)`.
- Added progress descriptions for generation, fast filter, scout target calls, full safety judge, and final target calls in `cascade.py`.
- Progress can be disabled with `RUFP_DISABLE_TQDM=1`.
- Tqdm update interval can be tuned with `RUFP_TQDM_MININTERVAL`.

Tested with `pytest -q`: 37 passed.
