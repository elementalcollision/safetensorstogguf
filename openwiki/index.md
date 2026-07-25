---
type: Documentation Index
title: "OpenWiki"
description: "Files and subdirectories in OpenWiki."
---

# Files

- [Architecture — Adapter Pattern and llama.cpp Compatibility](architecture.md) - How the toolkit probes and adapts to multiple generations of llama.cpp's converter API, the UpstreamConverter class, module dependency graph, and the version compatibility matrix.
- [Operations — Setup, Troubleshooting, and Known Limitations](operations.md) - How to set up llama.cpp for the toolkit, how the discovery pattern works, common errors and their fixes, known limitations, and how to regenerate this wiki.
- [SafeTensors to GGUF — Quickstart](quickstart.md) - Entry point for the safetensorstogguf toolkit. Covers what the tools do, how to install them, quick-start commands, and links to architecture, workflows, operations, and testing documentation.
- [Testing — Regression Suite and Compatibility Tests](testing.md) - How to run the test suite, what each test file covers, the regression history behind each test, and guidance for adding new tests when changing the adapter or quantization logic.
- [Workflows — Conversion, Quantization, and MoE Pipeline](workflows.md) - The three core CLI workflows (SafeTensors→GGUF conversion, GGUF quantization with MoE support, and the two-step convert+quantize pipeline), plus model analysis. Includes CLI reference for all flags.
