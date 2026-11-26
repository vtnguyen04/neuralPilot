# Third-Party Licenses & Attribution Notice

`NeuroPilot` is an independent, modular self-driving deep learning framework. Certain neural network architectures, pretrained checkpoint conversions, camera geometry utilities, and public demonstration media included or supported by `NeuroPilot` are derived from or compatible with open-source works released under permissive licenses.

All original copyright notices, license texts, and attribution requirements are preserved below.

---

## 1. comma.ai (`openpilot`, `openpilot_driving_models`, `commaVQ`, `worldmodel-4B`, `comma2k19`)

- **Upstream Repositories:**
  - `https://github.com/commaai/openpilot`
  - `https://huggingface.co/commaai/openpilot_driving_models`
  - `https://huggingface.co/commaai/commavq`
  - `https://huggingface.co/commaai/worldmodel-4B`
  - `https://github.com/commaai/comma2k19` / `https://huggingface.co/datasets/commaai/comma2k19`
- **Affected NeuroPilot Modules:**
  - `neuro_pilot/cfg/models/zoo/{pilot_s,pilot_l,pilot_nav,driver,vq_enc,vq_dec,ae,dit,gpt}.yaml`
  - `neuro_pilot/engine/predictors/{planner,world}.py`
  - `neuro_pilot/utils/camera.py` (`MEDMODEL`/`SBIGMODEL` virtual camera constants)
  - Public README demonstration clips sourced from `comma2k19` (`assets/demos/`)
- **License:** MIT License

```text
MIT License

Copyright (c) 2018-2026 comma.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. The Autoware Foundation (`VisionPilot` and `LatentDrive`)

- **Upstream Repositories:**
  - `https://github.com/autowarefoundation/autoware.privately-owned-vehicles` (VisionPilot)
  - `https://github.com/autowarefoundation/autoware_latent_drive`
  - `https://huggingface.co/AutowareFoundation/latent_drive`
- **Affected NeuroPilot Modules & Modifications:**
  - `neuro_pilot/cfg/models/zoo/{cipo,lane,drive,depth,jepa}.yaml` (reconstructed into layer-by-layer modular YAML graphs from upstream ONNX exports)
  - `neuro_pilot/engine/predictors/{perception,world}.py`
- **License:** Apache License, Version 2.0

```text
Copyright The Autoware Foundation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 3. DDRNet (`DDRNet23-Slim`)

- **Upstream Repositories:**
  - `https://github.com/chenjun2hao/DDRNet.pytorch`
  - `https://huggingface.co/qualcomm/DDRNet23-Slim`
- **Affected NeuroPilot Modules:**
  - `neuro_pilot/cfg/models/zoo/seg.yaml`
  - `neuro_pilot/nn/zoo/blocks/seg.py`
  - `neuro_pilot/nn/zoo/converters/seg.py`
- **License:** MIT License

```text
MIT License

Copyright (c) 2021 Yuanduo Hong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 4. EoMT (`cityscapes_semantic_eomt_large_1024`)

- **Upstream Repositories:**
  - `https://github.com/tue-mps/eomt`
  - `https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024`
- **Affected NeuroPilot Modules:**
  - `neuro_pilot/cfg/models/zoo/seg_l.yaml`
  - `neuro_pilot/nn/zoo/blocks/eomt.py`
  - `neuro_pilot/nn/zoo/converters/eomt.py`
- **License:** MIT License

```text
MIT License

Copyright (c) 2025 Tommie Kerssies, Niccolo Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, Daan de Geus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 5. Dataset Redistribution Policy (Important)

- **Permitted for Public Demos & Redistribution:** Only media from permissively licensed datasets (such as `comma2k19` under the **MIT License**) or self-recorded first-party vehicle logs may be committed into `assets/demos/` or displayed in public READMEs.
- **Excluded Datasets:** Datasets governed by restricted/confidential agreements (such as `comma5k` under `comma-av-dataset`, `CoVLA-Dataset`, or `BDD100K` raw media) are **strictly excluded** from public redistribution, commits, and promotional benchmarks.
