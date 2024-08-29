# Taoverse Package

This is a package containing various python modules for use in bittensor subnets.

Currently it is primarily supporting the Macrocosmos SN 9 Pretraining and SN 37 Finetuning subnets but much of the code is generally useful and all are welcome to leverage it for their own work as well.

Find the latest published package here: https://pypi.org/project/taoverse/

Following is an overview of each module and some of the bigger pieces of code they include.

## Metagraph

The metagraph module contains code relating to metagraph operations.

- The `MetagraphSyncer` allows for easily refreshing a specified metagraph at a specified cadence. It will operate in asynchronously on the asyncio event loop and will notify any registered listeners after each sync.

- The `MinerIterator` provides a thread safe infinite iterator that safely handles adding new uids.

## Model

The model module contains code relating to large language models.

- The `ModelTracker` will track model metadata and evaluation history on a per hotkey basis. It is designed to keep state across restarts and has methods to save and load state.

- The `ModelUpdater` reads metadata from the chain, validates it, and checks if it is an update to a previous tracked hotkey.

- The model.competition module supports grouping models into distinct competitions.
  - The `CompetitionTracker` helps track weights at a per competition and per subnet level. Like the `ModelTracker` it is designed to keep state across restarts.
  - The `EpislonFunc` allows for computing epsilon with a custom function on a per competition basis. Both a `FixedEpsilon` and a `LinearDecay` implementation are included.

- The model.storage module supports storing and retrieving models and metadata.
  - For models there are included implementations for Hugging Face and the local disk.
  - For metadata there is an included implementation using the Bittensor chain.

## Utilities

The utilities module contains code found to be generically useful in the context of subnets.

- The `PerfMonitor` is a context manager that tracks the performance of a block of code by taking several samples
- In `utils.py` there are some helpers to `run_in_thread` or `run_in_subprocess`.
  - Running metagraph operations in a separate thread with a ttl can help avoid getting stuck on those actions.
  - Running model evaluation within a sub process can help ensure that the model is completely removed from the gpu afterwards.