# Contributing to q.it

q.it is a Raspberry Pi-first edge-model benchmark runner.
Changes should preserve comparable measurements, durable failure evidence, inert benchmark packs, and a useful command-line workflow over SSH.

## Before changing anything

Read [AGENTS.md](AGENTS.md) for repository-wide instructions.
Read [CONTEXT.md](CONTEXT.md) for project vocabulary.
Read [ADR-0005](docs/adr/0005-benchmark-packs-and-provider-adapters.md) for the active product shape.
Read [the benchmark landscape](docs/research/pi5-edge-benchmark-landscape.md) before changing provider or metric semantics.
Read [the built-in benchmark guide](docs/built-in-benchmarks.md) before changing the bundled cases.
Read [the reproducibility guide](docs/reproducible-runs.md) before changing stored measurements or run settings.
Read [the benchmark pack guide](docs/benchmark-packs.md) before changing the pack format.

## Product seams

The Cargo-installed command is the execution interface.
The loopback HTTP dashboard is a read-only analysis interface.
Provider adapters own protocol behavior and exact timing semantics.
Benchmark packs contain only declarative definitions, cases, and media.

Do not put executable scorers, scripts, or model code in a benchmark pack.
Do not make model acquisition a hidden side effect of a measured run.
Do not count transport chunks as tokens.
Do not merge failed runs into the successful run dataset.

## Test strategy

Tests should exercise the product through the command or dashboard interfaces.
Use temporary state and local fake provider endpoints instead of real model weights or network services.
Provider fixtures must include the same terminal marker and usage metadata that the real protocol requires.

Run the Rust suite:

```bash
cargo test
```

Build the dashboard:

```bash
cd qit-web
npm run build
```

Verify the packaged product:

```bash
scripts/test-packaged-product.sh
```

The network is not part of the normal test suite.
Real Raspberry Pi qualification should record the board, RAM, cooling, power, OS, provider version, model identity, and run configuration.

## Benchmark-pack changes

Keep cases deterministic and use stable identifiers.
Record publisher, source, version, and license metadata.
Keep media inside the pack and avoid symbolic links.
An incorrect model answer should receive a lower quality score rather than become a harness failure.

Validate an authored pack with:

```bash
qit benchmark verify ./path-to-pack
```

## Pull requests

Check formatting, tests, generated dashboard assets, and the final diff before handing work off.
Do not edit generated dashboard files manually.
Do not manually edit `CHANGELOG.md`.
