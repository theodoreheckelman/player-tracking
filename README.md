# player-mapping


DIY player tracking pipeline (detection → tracking → OCR → homography → UV).




## Quick start

I have included a sample game clip. If you want to use a different game clip, just put a new game clip in data following the same naming convetion and remove the old game_clip.

```bash
# install uv (follow official docs) then:
uv init player-mapping
# replace pyproject.toml with this project's pyproject.toml
uv sync
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
cd train/YOLOX
pip install -e .
python -m pip install -r train/YOLOX/requirements.txt
```
If you want to only make a homography
```bash
uv run python player_mapping/src/calibration.py --video data/game_clip.mp4 --make-homography
```
To run the entire pipeline 
```bash
uv run python player_mapping/src/main.py --video data/game_clip.mp4 --make-homography 
``` 

### Training
Run 
```bash 
ffmpeg -i ../data/game_clip.mp4 frames/frame_%05d.png
uv run python train/train.py
``` 
Annotate your images
```bash
cd YOLOX
pip install -U pip
pip install -v -e .  # installs YOLOX in editable mode
pip install -r requirements.txt
cp ../../etc/my_player_dataset.py yolox/exp/ 
python3 tools/train.py -f yolox/exp/my_player_dataset.py -d 1 -b 8 -os
``` 
I set up the training to be used only for CPU. This was custom by me and if there is a YOLOX source code in here, it can be assumed to be a forked version of YOLOX. If you want the default GPU option than just reclone YOLOX on their main branch. There might need to be file paths changed in your custom dataset. I have added a copy of my data set in /etc. This goes in 
```bash
./train/YOLOX/yolox/exp/
```

FILE STRUCTURE
.
├── README.md
├── data
│   └── game_clip.mp4
├── frames
├── homography.npz
├── outputs
│   └── tracking.csv
├── player_mapping
│   └── src
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── bytetrack.cpython-312.pyc
│       │   ├── calibration.cpython-312.pyc
│       │   ├── detection.cpython-312.pyc
│       │   ├── ocr.cpython-312.pyc
│       │   ├── tracker.cpython-312.pyc
│       │   ├── tracking.cpython-312.pyc
│       │   ├── utils.cpython-312.pyc
│       │   └── visualizer.cpython-312.pyc
│       ├── bytetrack.py
│       ├── calibration.py
│       ├── detection.py
│       ├── main.py
│       ├── ocr.py
│       ├── tracker.py
│       ├── tracking.py
│       ├── utils.py
│       └── visualizer.py
├── pyproject.toml
├── train
│   ├── YOLOX
│   │   ├── LICENSE
│   │   ├── MANIFEST.in
│   │   ├── README.md
│   │   ├── SECURITY.md
│   │   ├── YOLOX_outputs
│   │   │   └── my_player_dataset
│   │   │       └── train_log.txt
│   │   ├── assets
│   │   │   ├── assignment.png
│   │   │   ├── demo.png
│   │   │   ├── dog.jpg
│   │   │   ├── git_fig.png
│   │   │   ├── logo.png
│   │   │   └── sunjian.png
│   │   ├── build
│   │   │   ├── lib.macosx-10.9-x86_64-3.8
│   │   │   │   └── yolox
│   │   │   │       └── layers
│   │   │   │           └── fast_cocoeval.cpython-38-darwin.so
│   │   │   └── temp.macosx-10.9-x86_64-3.8
│   │   │       ├── build.ninja
│   │   │       └── yolox
│   │   │           └── layers
│   │   │               └── cocoeval
│   │   │                   └── cocoeval.o
│   │   ├── datasets
│   │   │   └── README.md
│   │   ├── demo
│   │   │   ├── MegEngine
│   │   │   │   ├── cpp
│   │   │   │   │   ├── README.md
│   │   │   │   │   ├── build.sh
│   │   │   │   │   └── yolox.cpp
│   │   │   │   └── python
│   │   │   │       ├── README.md
│   │   │   │       ├── build.py
│   │   │   │       ├── convert_weights.py
│   │   │   │       ├── demo.py
│   │   │   │       ├── dump.py
│   │   │   │       └── models
│   │   │   │           ├── __init__.py
│   │   │   │           ├── darknet.py
│   │   │   │           ├── network_blocks.py
│   │   │   │           ├── yolo_fpn.py
│   │   │   │           ├── yolo_head.py
│   │   │   │           ├── yolo_pafpn.py
│   │   │   │           └── yolox.py
│   │   │   ├── ONNXRuntime
│   │   │   │   ├── README.md
│   │   │   │   └── onnx_inference.py
│   │   │   ├── OpenVINO
│   │   │   │   ├── README.md
│   │   │   │   ├── cpp
│   │   │   │   │   ├── CMakeLists.txt
│   │   │   │   │   ├── README.md
│   │   │   │   │   └── yolox_openvino.cpp
│   │   │   │   └── python
│   │   │   │       ├── README.md
│   │   │   │       └── openvino_inference.py
│   │   │   ├── TensorRT
│   │   │   │   ├── cpp
│   │   │   │   │   ├── CMakeLists.txt
│   │   │   │   │   ├── README.md
│   │   │   │   │   ├── logging.h
│   │   │   │   │   └── yolox.cpp
│   │   │   │   └── python
│   │   │   │       └── README.md
│   │   │   ├── ncnn
│   │   │   │   ├── README.md
│   │   │   │   ├── android
│   │   │   │   │   ├── README.md
│   │   │   │   │   ├── app
│   │   │   │   │   │   ├── build.gradle
│   │   │   │   │   │   └── src
│   │   │   │   │   │       └── main
│   │   │   │   │   │           ├── AndroidManifest.xml
│   │   │   │   │   │           ├── assets
│   │   │   │   │   │           │   └── yolox.param
│   │   │   │   │   │           ├── java
│   │   │   │   │   │           │   └── com
│   │   │   │   │   │           │       └── megvii
│   │   │   │   │   │           │           └── yoloXncnn
│   │   │   │   │   │           │               ├── MainActivity.java
│   │   │   │   │   │           │               └── yoloXncnn.java
│   │   │   │   │   │           ├── jni
│   │   │   │   │   │           │   ├── CMakeLists.txt
│   │   │   │   │   │           │   └── yoloXncnn_jni.cpp
│   │   │   │   │   │           └── res
│   │   │   │   │   │               ├── layout
│   │   │   │   │   │               │   └── main.xml
│   │   │   │   │   │               └── values
│   │   │   │   │   │                   └── strings.xml
│   │   │   │   │   ├── build.gradle
│   │   │   │   │   ├── gradle
│   │   │   │   │   │   └── wrapper
│   │   │   │   │   │       ├── gradle-wrapper.jar
│   │   │   │   │   │       └── gradle-wrapper.properties
│   │   │   │   │   ├── gradlew
│   │   │   │   │   ├── gradlew.bat
│   │   │   │   │   └── settings.gradle
│   │   │   │   └── cpp
│   │   │   │       ├── README.md
│   │   │   │       └── yolox.cpp
│   │   │   └── nebullvm
│   │   │       ├── README.md
│   │   │       └── nebullvm_optimization.py
│   │   ├── docs
│   │   │   ├── Makefile
│   │   │   ├── _static
│   │   │   │   └── css
│   │   │   │       └── custom.css
│   │   │   ├── assignment_visualization.md
│   │   │   ├── cache.md
│   │   │   ├── conf.py
│   │   │   ├── demo
│   │   │   │   ├── megengine_cpp_readme.md -> ../../demo/MegEngine/cpp/README.md
│   │   │   │   ├── megengine_py_readme.md -> ../../demo/MegEngine/python/README.md
│   │   │   │   ├── ncnn_android_readme.md -> ../../demo/ncnn/android/README.md
│   │   │   │   ├── ncnn_cpp_readme.md -> ../../demo/ncnn/cpp/README.md
│   │   │   │   ├── onnx_readme.md -> ../../demo/ONNXRuntime/README.md
│   │   │   │   ├── openvino_cpp_readme.md -> ../../demo/OpenVINO/cpp/README.md
│   │   │   │   ├── openvino_py_readme.md -> ../../demo/OpenVINO/python/README.md
│   │   │   │   ├── trt_cpp_readme.md -> ../../demo/TensorRT/cpp/README.md
│   │   │   │   └── trt_py_readme.md -> ../../demo/TensorRT/python/README.md
│   │   │   ├── freeze_module.md
│   │   │   ├── index.rst
│   │   │   ├── manipulate_training_image_size.md
│   │   │   ├── mlflow_integration.md
│   │   │   ├── model_zoo.md
│   │   │   ├── quick_run.md
│   │   │   ├── requirements-doc.txt
│   │   │   ├── train_custom_data.md
│   │   │   └── updates_note.md
│   │   ├── exps
│   │   │   ├── __init__.py
│   │   │   ├── default
│   │   │   │   ├── __init__.py
│   │   │   │   ├── yolov3.py
│   │   │   │   ├── yolox_l.py
│   │   │   │   ├── yolox_m.py
│   │   │   │   ├── yolox_nano.py
│   │   │   │   ├── yolox_s.py
│   │   │   │   ├── yolox_tiny.py
│   │   │   │   └── yolox_x.py
│   │   │   └── example
│   │   │       ├── custom
│   │   │       │   ├── nano.py
│   │   │       │   └── yolox_s.py
│   │   │       └── yolox_voc
│   │   │           └── yolox_voc_s.py
│   │   ├── hubconf.py
│   │   ├── requirements.txt
│   │   ├── setup.cfg
│   │   ├── setup.py
│   │   ├── tests
│   │   │   ├── __init__.py
│   │   │   └── utils
│   │   │       └── test_model_utils.py
│   │   ├── tools
│   │   │   ├── __init__.py
│   │   │   ├── demo.py
│   │   │   ├── eval.py
│   │   │   ├── export_onnx.py
│   │   │   ├── export_torchscript.py
│   │   │   ├── train.py
│   │   │   ├── trt.py
│   │   │   └── visualize_assign.py
│   │   ├── yolox
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   └── __init__.cpython-38.pyc
│   │   │   ├── core
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── launch.cpython-38.pyc
│   │   │   │   │   └── trainer.cpython-38.pyc
│   │   │   │   ├── launch.py
│   │   │   │   └── trainer.py
│   │   │   ├── data
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── data_augment.cpython-38.pyc
│   │   │   │   │   ├── data_prefetcher.cpython-38.pyc
│   │   │   │   │   ├── dataloading.cpython-38.pyc
│   │   │   │   │   └── samplers.cpython-38.pyc
│   │   │   │   ├── data_augment.py
│   │   │   │   ├── data_prefetcher.py
│   │   │   │   ├── dataloading.py
│   │   │   │   ├── datasets
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── __pycache__
│   │   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   │   ├── coco.cpython-38.pyc
│   │   │   │   │   │   ├── coco_classes.cpython-38.pyc
│   │   │   │   │   │   ├── datasets_wrapper.cpython-38.pyc
│   │   │   │   │   │   ├── mosaicdetection.cpython-38.pyc
│   │   │   │   │   │   ├── voc.cpython-38.pyc
│   │   │   │   │   │   └── voc_classes.cpython-38.pyc
│   │   │   │   │   ├── coco.py
│   │   │   │   │   ├── coco_classes.py
│   │   │   │   │   ├── datasets_wrapper.py
│   │   │   │   │   ├── mosaicdetection.py
│   │   │   │   │   ├── voc.py
│   │   │   │   │   └── voc_classes.py
│   │   │   │   └── samplers.py
│   │   │   ├── evaluators
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── coco_evaluator.cpython-38.pyc
│   │   │   │   │   ├── voc_eval.cpython-38.pyc
│   │   │   │   │   └── voc_evaluator.cpython-38.pyc
│   │   │   │   ├── coco_evaluator.py
│   │   │   │   ├── voc_eval.py
│   │   │   │   └── voc_evaluator.py
│   │   │   ├── exp
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── base_exp.cpython-38.pyc
│   │   │   │   │   ├── build.cpython-38.pyc
│   │   │   │   │   ├── my_player_dataset.cpython-38.pyc
│   │   │   │   │   └── yolox_base.cpython-38.pyc
│   │   │   │   ├── base_exp.py
│   │   │   │   ├── build.py
│   │   │   │   ├── default
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── my_player_dataset.py
│   │   │   │   └── yolox_base.py
│   │   │   ├── layers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cocoeval
│   │   │   │   │   ├── cocoeval.cpp
│   │   │   │   │   └── cocoeval.h
│   │   │   │   ├── fast_coco_eval_api.py
│   │   │   │   ├── fast_cocoeval.cpython-38-darwin.so
│   │   │   │   └── jit_ops.py
│   │   │   ├── models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── build.cpython-38.pyc
│   │   │   │   │   ├── darknet.cpython-38.pyc
│   │   │   │   │   ├── losses.cpython-38.pyc
│   │   │   │   │   ├── network_blocks.cpython-38.pyc
│   │   │   │   │   ├── yolo_fpn.cpython-38.pyc
│   │   │   │   │   ├── yolo_head.cpython-38.pyc
│   │   │   │   │   ├── yolo_pafpn.cpython-38.pyc
│   │   │   │   │   └── yolox.cpython-38.pyc
│   │   │   │   ├── build.py
│   │   │   │   ├── darknet.py
│   │   │   │   ├── losses.py
│   │   │   │   ├── network_blocks.py
│   │   │   │   ├── yolo_fpn.py
│   │   │   │   ├── yolo_head.py
│   │   │   │   ├── yolo_pafpn.py
│   │   │   │   └── yolox.py
│   │   │   ├── tools
│   │   │   │   └── __init__.py
│   │   │   └── utils
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-38.pyc
│   │   │       │   ├── allreduce_norm.cpython-38.pyc
│   │   │       │   ├── boxes.cpython-38.pyc
│   │   │       │   ├── checkpoint.cpython-38.pyc
│   │   │       │   ├── compat.cpython-38.pyc
│   │   │       │   ├── demo_utils.cpython-38.pyc
│   │   │       │   ├── dist.cpython-38.pyc
│   │   │       │   ├── ema.cpython-38.pyc
│   │   │       │   ├── logger.cpython-38.pyc
│   │   │       │   ├── lr_scheduler.cpython-38.pyc
│   │   │       │   ├── metric.cpython-38.pyc
│   │   │       │   ├── mlflow_logger.cpython-38.pyc
│   │   │       │   ├── model_utils.cpython-38.pyc
│   │   │       │   ├── setup_env.cpython-38.pyc
│   │   │       │   └── visualize.cpython-38.pyc
│   │   │       ├── allreduce_norm.py
│   │   │       ├── boxes.py
│   │   │       ├── checkpoint.py
│   │   │       ├── compat.py
│   │   │       ├── demo_utils.py
│   │   │       ├── dist.py
│   │   │       ├── ema.py
│   │   │       ├── logger.py
│   │   │       ├── lr_scheduler.py
│   │   │       ├── metric.py
│   │   │       ├── mlflow_logger.py
│   │   │       ├── model_utils.py
│   │   │       ├── setup_env.py
│   │   │       └── visualize.py
│   │   └── yolox.egg-info
│   │       ├── PKG-INFO
│   │       ├── SOURCES.txt
│   │       ├── dependency_links.txt
│   │       ├── requires.txt
│   │       └── top_level.txt
│   ├── YOLOX_dataset
│   │   ├── annotations
│   │   │   └── instances_train.json
│   │   └── images
│   │       ├── train
│   │       └── val
│   ├── frames
│   │   ├── example frames I used for training
│   └── train.py
├── uv.lock
└── yolov8m.pt