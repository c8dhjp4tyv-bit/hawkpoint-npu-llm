import importlib.util
import hashlib
from pathlib import Path
import tempfile

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "tools/convert_smollm2.py"
SPEC = importlib.util.spec_from_file_location("convert_smollm2", MODULE)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def test_per_channel_quantization():
    rng = np.random.default_rng(7)
    weight = rng.normal(0, 0.2, (32, 64)).astype(np.float32)
    q, scale = CONVERTER.quantize_per_output_channel(weight)
    restored = q.astype(np.float32) * scale[:, None]
    error = np.abs(restored - weight)
    assert q.dtype == np.int8
    assert scale.dtype == np.float32
    assert np.all(error.max(axis=1) <= scale / 2 + 1e-6)


def test_architecture_validation():
    valid = {
        **CONVERTER.EXPECTED_ARCHITECTURE,
        "model_type": "llama",
    }
    assert CONVERTER._validate_architecture(valid) == "llama"
    invalid = {**valid, "hidden_size": 960}
    try:
        CONVERTER._validate_architecture(invalid)
        raise AssertionError("incompatible architecture unexpectedly accepted")
    except ValueError as exc:
        assert "hidden_size=960" in str(exc)
    qwen = {
        **CONVERTER.QWEN_ARCHITECTURE,
        "model_type": "qwen2",
    }
    assert CONVERTER._validate_architecture(qwen) == "qwen2"


def test_atomic_publish_and_failed_conversion_preserves_previous():
    original = CONVERTER._convert_into
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        output.mkdir()
        (output / "old.bin").write_bytes(b"old")

        def successful(_source, staging, **_metadata):
            data = b"complete-model"
            (staging / "weights.bin").write_bytes(data)
            return {
                "files": {
                    "weights.bin": {
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                }
            }

        CONVERTER._convert_into = successful
        CONVERTER.convert(source, output)
        assert not (output / "old.bin").exists()
        assert (output / "weights.bin").read_bytes() == b"complete-model"

        def failing(_source, staging, **_metadata):
            (staging / "partial.bin").write_bytes(b"partial")
            raise OSError("simulated disk-full failure")

        CONVERTER._convert_into = failing
        try:
            CONVERTER.convert(source, output)
            raise AssertionError("failed conversion unexpectedly succeeded")
        except OSError:
            pass
        assert (output / "weights.bin").read_bytes() == b"complete-model"
        assert not (output / "partial.bin").exists()
    CONVERTER._convert_into = original


def test_publish_faults_never_lose_a_good_package():
    """Injected failures around the rename must always leave a usable output."""
    original_convert_into = CONVERTER._convert_into
    original_fsync_directory = CONVERTER._fsync_directory
    original_replace = CONVERTER.os.replace

    def publisher(payload):
        def _convert_into(_source, staging, **_metadata):
            (staging / "weights.bin").write_bytes(payload)
            return {
                "files": {
                    "weights.bin": {
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                }
            }

        return _convert_into

    def expect_failure(source, output, message):
        try:
            CONVERTER.convert(source, output)
        except OSError as exc:
            assert message in str(exc), str(exc)
            return exc
        raise AssertionError(f"{message}: conversion unexpectedly succeeded")

    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            CONVERTER._convert_into = publisher(b"first-good-package")
            CONVERTER.convert(source, output)
            assert (output / "weights.bin").read_bytes() == b"first-good-package"

            # 1. Failure before publish: the previous package is untouched and
            # no staging or backup directory is left behind.
            def failing_before(_source, staging, **_metadata):
                (staging / "partial.bin").write_bytes(b"partial")
                raise OSError("simulated failure before publish")

            CONVERTER._convert_into = failing_before
            expect_failure(source, output, "simulated failure before publish")
            assert (output / "weights.bin").read_bytes() == b"first-good-package"
            assert not list(root.glob(".output.*"))

            # 2. Failure during publish: the staging rename itself fails, so
            # the backup must be renamed back into place.
            CONVERTER._convert_into = publisher(b"second-package")

            def failing_replace(src, dst):
                if Path(src).name.startswith(".output.staging-"):
                    raise OSError("simulated failure during publish")
                return original_replace(src, dst)

            CONVERTER.os.replace = failing_replace
            try:
                expect_failure(source, output, "simulated failure during publish")
            finally:
                CONVERTER.os.replace = original_replace
            assert (output / "weights.bin").read_bytes() == b"first-good-package"
            assert not list(root.glob(".output.backup-*"))

            # 3. Failure immediately after publish: the parent fsync raises
            # once the rename has already landed. The previous package must be
            # restored exactly, and the new one kept aside rather than deleted.
            def failing_fsync(path):
                if Path(path) == output.parent:
                    raise OSError("simulated parent fsync failure")
                return original_fsync_directory(path)

            CONVERTER._fsync_directory = failing_fsync
            try:
                exc = expect_failure(
                    source, output, "simulated parent fsync failure"
                )
            finally:
                CONVERTER._fsync_directory = original_fsync_directory
            assert (output / "weights.bin").read_bytes() == b"first-good-package"
            rejected = list(root.glob(".output.failed-*"))
            assert len(rejected) == 1, rejected
            assert (rejected[0] / "weights.bin").read_bytes() == b"second-package"
            assert any("was kept at" in note for note in getattr(exc, "__notes__", []))
            assert not list(root.glob(".output.backup-*"))
            assert not list(root.glob(".output.staging-*"))

            # 4. Recovery: a later successful conversion still publishes.
            CONVERTER._convert_into = publisher(b"third-package")
            CONVERTER.convert(source, output)
            assert (output / "weights.bin").read_bytes() == b"third-package"
    finally:
        CONVERTER._convert_into = original_convert_into
        CONVERTER._fsync_directory = original_fsync_directory
        CONVERTER.os.replace = original_replace


if __name__ == "__main__":
    test_per_channel_quantization()
    test_architecture_validation()
    test_atomic_publish_and_failed_conversion_preserves_previous()
    test_publish_faults_never_lose_a_good_package()
    print("PASS!")
