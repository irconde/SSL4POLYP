import contextlib
import csv
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "distutils" not in sys.modules:
    sys.modules["distutils"] = types.ModuleType("distutils")
if "distutils.version" not in sys.modules:
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")
if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.ModuleType("yaml")
class _NoGradStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __call__(self, fn):
        return fn


if "numpy" not in sys.modules:
    class _NumpyFunctionStub:
        def __call__(self, *args, **kwargs):
            return None

    class _NumpyStub(types.ModuleType):
        integer = int
        floating = float
        ndarray = object

        def __getattr__(self, name):
            stub = _NumpyFunctionStub()
            setattr(self, name, stub)
            return stub

    numpy_stub = _NumpyStub("numpy")
    numpy_stub.random = types.SimpleNamespace(
        seed=lambda *args, **kwargs: None,
        rand=lambda *args, **kwargs: None,
        choice=lambda *args, **kwargs: None,
    )
    sys.modules["numpy"] = numpy_stub
if "PIL" not in sys.modules:
    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_image_draw_stub = types.ModuleType("PIL.ImageDraw")
    pil_image_enhance_stub = types.ModuleType("PIL.ImageEnhance")
    pil_image_filter_stub = types.ModuleType("PIL.ImageFilter")
    pil_stub.Image = pil_image_stub
    pil_stub.ImageDraw = pil_image_draw_stub
    pil_stub.ImageEnhance = pil_image_enhance_stub
    pil_stub.ImageFilter = pil_image_filter_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = pil_image_stub
    sys.modules["PIL.ImageDraw"] = pil_image_draw_stub
    sys.modules["PIL.ImageEnhance"] = pil_image_enhance_stub
    sys.modules["PIL.ImageFilter"] = pil_image_filter_stub
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.device = lambda *args, **kwargs: (args, kwargs)
    torch_stub.manual_seed = lambda *args, **kwargs: None
    torch_stub.as_tensor = lambda *args, **kwargs: None
    torch_stub.tensor = lambda *args, **kwargs: None
    torch_stub.full = lambda *args, **kwargs: None
    torch_stub.cat = lambda *args, **kwargs: None
    torch_stub.stack = lambda *args, **kwargs: None
    torch_stub.arange = lambda *args, **kwargs: None
    torch_stub.zeros = lambda *args, **kwargs: None
    torch_stub.ones = lambda *args, **kwargs: None
    torch_stub.sigmoid = lambda *args, **kwargs: None
    torch_stub.softmax = lambda *args, **kwargs: None
    torch_stub.topk = lambda *args, **kwargs: None
    torch_stub.save = lambda *args, **kwargs: None
    torch_stub.load = lambda *args, **kwargs: None
    torch_stub.no_grad = lambda *args, **kwargs: _NoGradStub()
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *args, **kwargs: None,
    )
    torch_stub.cuda.amp = types.SimpleNamespace(
        GradScaler=type("GradScaler", (), {}),
        autocast=lambda **kwargs: contextlib.nullcontext(),
    )
    torch_stub.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False, deterministic=False)
    )
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_stub.Module = type("Module", (), {})
    torch_nn_stub.Parameter = type("Parameter", (), {})
    torch_nn_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_parallel_stub = types.ModuleType("torch.nn.parallel")
    torch_nn_parallel_stub.DistributedDataParallel = type(
        "DistributedDataParallel", (), {}
    )
    torch_distributed_stub = types.ModuleType("torch.distributed")
    torch_multiprocessing_stub = types.ModuleType("torch.multiprocessing")
    torch_optim_stub = types.ModuleType("torch.optim")
    torch_optim_stub.Optimizer = type("Optimizer", (), {})
    torch_lr_stub = types.ModuleType("torch.optim.lr_scheduler")
    torch_lr_stub.LambdaLR = type("LambdaLR", (), {})
    torch_lr_stub.ReduceLROnPlateau = type("ReduceLROnPlateau", (), {})
    torch_optim_stub.lr_scheduler = torch_lr_stub
    torch_stub.nn = torch_nn_stub
    torch_stub.distributed = torch_distributed_stub
    torch_stub.multiprocessing = torch_multiprocessing_stub
    torch_stub.optim = torch_optim_stub
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_data_stub = types.ModuleType("torch.utils.data")
    torch_utils_data_stub.DataLoader = type("DataLoader", (), {})
    torch_utils_data_stub.Dataset = type("Dataset", (), {})
    torch_utils_data_stub.DistributedSampler = type("DistributedSampler", (), {})
    torch_utils_stub.data = torch_utils_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_nn_functional_stub
    sys.modules["torch.nn.parallel"] = torch_nn_parallel_stub
    sys.modules["torch.distributed"] = torch_distributed_stub
    sys.modules["torch.multiprocessing"] = torch_multiprocessing_stub
    sys.modules["torch.optim"] = torch_optim_stub
    sys.modules["torch.optim.lr_scheduler"] = torch_lr_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_utils_data_stub
if "torchvision" not in sys.modules:
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_transforms_stub = types.ModuleType("torchvision.transforms")
    torchvision_stub.transforms = torchvision_transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = torchvision_transforms_stub
if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    sklearn_metrics_stub = types.ModuleType("sklearn.metrics")
    for _name in (
        "balanced_accuracy_score",
        "average_precision_score",
        "roc_auc_score",
        "roc_curve",
        "precision_recall_curve",
        "f1_score",
        "precision_score",
        "recall_score",
        "matthews_corrcoef",
    ):
        setattr(sklearn_metrics_stub, _name, lambda *args, **kwargs: None)
    sklearn_stub.metrics = sklearn_metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = sklearn_metrics_stub

from ssl4polyp.classification.train_classification import _export_frame_outputs


@pytest.mark.parametrize(
    "dataset_name",
    [
        "polypgen_fewshot",
        "polypgen_fewshot_s50",
        "PolypGen_FewShot_S200",
        "polypgen_clean_test",
        "polypgen_clean_test_extended",
        "PolypGen_Clean_Test",
    ],
)
def test_export_frame_outputs_polypgen_adjusts_columns(
    tmp_path: Path, dataset_name: str
) -> None:
    path = tmp_path / "polypgen.csv"
    metadata_rows = [
        {
            "dataset": "PolypGen",
            "case_id": "c2",
            "origin": "polypgen_clean",
            "store_id": "polypgen_clean",
            "frame_id": "POLYPGEN.case0001",
        },
        {
            "dataset": "PolypGen",
            "case_id": "None",
            "origin": "polypgen_clean",
            "store_id": "polypgen_clean",
            "frame_id": "NEGSEQ.seq15_neg.00017",
        },
    ]
    _export_frame_outputs(
        path,
        metadata_rows=metadata_rows,
        probabilities=[0.9, 0.1],
        targets=[1, 0],
        preds=[1, 0],
        dataset_name=dataset_name,
    )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "frame_id",
            "prob",
            "label",
            "pred",
            "origin",
            "center_id",
            "sequence_id",
        ]
        rows = list(reader)

    assert rows[0]["center_id"] == "C2"
    assert rows[0]["sequence_id"] == ""
    assert rows[1]["center_id"] == "None"
    assert rows[1]["sequence_id"] == "15"
    assert rows[1]["origin"] == "polypgen_clean"


def test_export_frame_outputs_preserves_columns_for_other_datasets(tmp_path: Path) -> None:
    path = tmp_path / "sun.csv"
    metadata_rows = [
        {
            "dataset": "SUN",
            "case_id": "caseA",
            "sequence_id": "seq1",
            "origin": "sun",
            "frame_id": "frame0",
            "morphology": "flat",
        }
    ]
    _export_frame_outputs(
        path,
        metadata_rows=metadata_rows,
        probabilities=[0.75],
        targets=[1],
        preds=[1],
        dataset_name="sun_test",
    )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "frame_id",
            "prob",
            "label",
            "pred",
            "case_id",
            "origin",
            "center_id",
            "sequence_id",
            "morphology",
        ]
        rows = list(reader)

    assert rows[0]["case_id"] == "caseA"
    assert rows[0]["morphology"] == "flat"
