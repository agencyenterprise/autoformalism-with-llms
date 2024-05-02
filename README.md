# autoformalism-with-llms

[![PyPI - Version](https://img.shields.io/pypi/v/autoformalism-with-llms.svg)](https://pypi.org/project/autoformalism-with-llms)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autoformalism-with-llms.svg)](https://pypi.org/project/autoformalism-with-llms)

-----

**Table of Contents**

- [About](#about)
- [Installation](#installation)
- [License](#license)


## About
This repository reproduces the in-context Autoformalism experiments from the paper 
[Autoformalism with Large Language Models](https://arxiv.org/abs/2205.12615).  There 
are minor difference in the approach, for example, the strongest results in the paper
were obtained with the Codex model which is now retired, so we use GPT-4 and Llama3
in its place.  Our models are also chat variants so the exact format of the few-shot
prompt is slightly different.  Given these small difference, we create the exact same
prompts on the exact same dataset to attempt to recreate the results of the paper.

The results are in the `experiments` folder, viewable as notebooks.

## Installation

```console
pip install autoformalism-with-llms
```

## License

`autoformalism-with-llms` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
