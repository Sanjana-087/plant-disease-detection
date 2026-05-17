"""
Load plant leaf images from directory-based datasets.

Supports PlantVillage (nested color/segmented folders), PlantDoc (train splits
and spaced class names), and custom Arecanut layouts (class_name/image.jpg).
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from training.config import (
    ARECANUT_PATH,
    CLASS_NAMES_PATH,
    IMAGE_EXTENSIONS,
    LABEL_MAPPING_PATH,
    PLANTDOC_PATH,
    PLANTVILLAGE_PATH,
    RANDOM_SEED,
    update_class_info,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Scan image datasets organized by class subfolders and encode labels.

    Each disease class is a directory under the scan root (or nested one level
    for PlantVillage color/segmented layouts). Returns file paths and integer
    labels suitable for stratified splitting and tf.data pipelines.
    """

    # Known intermediate folders in PlantVillage archives (not class names)
    PLANTVILLAGE_INTERMEDIATE_DIRS = {
        "color",
        "segmented",
        "grayscale",
        "plantvillage",
        "plantvillage dataset",
        "plant leaves",
    }

    # PlantDoc often ships with train/validation/test splits
    PLANTDOC_SPLIT_DIRS = {"train", "validation", "val", "test", "testing"}

    def __init__(
        self,
        root_dir: str | Path,
        label_mapping_path: str | None = None,
        dataset_name: str = "auto",
        validate_images: bool = True,
        min_images_per_class: int = 1,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            root_dir: Path to dataset root (e.g. dataset/plantvillage).
            label_mapping_path: JSON path for label mapping export.
                Defaults to config LABEL_MAPPING_PATH.
            dataset_name: 'auto', 'plantvillage', 'plantdoc', or 'arecanut'.
                Controls how the scan root is resolved.
            validate_images: If True, skip corrupt/unreadable files.
            min_images_per_class: Minimum images required to keep a class.

        Raises:
            FileNotFoundError: If root_dir does not exist.
            ValueError: If no images are found after scanning.
        """
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.label_mapping_path = label_mapping_path or LABEL_MAPPING_PATH
        self.dataset_name = dataset_name.lower().strip()
        self.validate_images = validate_images
        self.min_images_per_class = min_images_per_class

        self.label_encoder = LabelEncoder()
        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.class_names: list[str] = []
        self.label_mapping: dict = {}

    def _is_image_file(self, path: Path) -> bool:
        """Return True if path has a supported image extension."""
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def _validate_image_file(self, path: Path) -> bool:
        """
        Verify that an image file can be opened and decoded.

        Args:
            path: Path to the image file.

        Returns:
            True if the image is valid, False otherwise.
        """
        try:
            with Image.open(path) as img:
                img.verify()
            # verify() closes the file; reopen for a quick load check
            with Image.open(path) as img:
                img.load()
            return True
        except Exception as exc:
            logger.warning("Skipping corrupt or unreadable image %s: %s", path, exc)
            return False

    def _resolve_scan_roots(self) -> list[Path]:
        """
        Determine which directories to scan based on dataset layout.

        Returns:
            List of root paths to search for class folders and images.

        Raises:
            ValueError: If dataset_name is unknown.
        """
        name = self.dataset_name

        if name == "auto":
            name = self._infer_dataset_type()

        if name == "plantvillage":
            return self._plantvillage_scan_roots()
        if name == "plantdoc":
            return self._plantdoc_scan_roots()
        if name == "arecanut":
            return [self.root_dir]
        if name == "generic":
            return [self.root_dir]

        raise ValueError(
            f"Unknown dataset_name '{self.dataset_name}'. "
            "Use 'auto', 'plantvillage', 'plantdoc', 'arecanut', or 'generic'."
        )

    def _infer_dataset_type(self) -> str:
        """
        Guess dataset type from root path name or folder structure.

        Returns:
            One of 'plantvillage', 'plantdoc', 'arecanut', 'generic'.
        """
        root_str = str(self.root_dir).lower()
        if "plantdoc" in root_str:
            return "plantdoc"
        if "arecanut" in root_str:
            return "arecanut"
        if "plantvillage" in root_str or "plant_village" in root_str:
            return "plantvillage"

        # Structure-based detection
        children = {p.name.lower() for p in self.root_dir.iterdir() if p.is_dir()}
        if children & self.PLANTVILLAGE_INTERMEDIATE_DIRS:
            return "plantvillage"
        if children & self.PLANTDOC_SPLIT_DIRS:
            return "plantdoc"
        return "generic"

    def _plantvillage_scan_roots(self) -> list[Path]:
        """
        PlantVillage archives often use root/color/ClassName/ or root/ClassName/.

        Returns:
            List of directories containing class subfolders.
        """
        roots: list[Path] = []
        for sub_name in ("color", "segmented", "grayscale"):
            candidate = self.root_dir / sub_name
            if candidate.is_dir():
                roots.append(candidate)

        if roots:
            logger.info(
                "PlantVillage layout: scanning %s",
                ", ".join(str(r) for r in roots),
            )
            return roots

        # Single-level class folders directly under root
        logger.info("PlantVillage layout: scanning class folders under %s", self.root_dir)
        return [self.root_dir]

    def _plantdoc_scan_roots(self) -> list[Path]:
        """
        PlantDoc may use train/, validation/, test/ each with class subfolders.

        Returns:
            List of split directories to scan, or [root_dir] if flat layout.
        """
        split_roots = [
            self.root_dir / split
            for split in self.PLANTDOC_SPLIT_DIRS
            if (self.root_dir / split).is_dir()
        ]
        if split_roots:
            logger.info(
                "PlantDoc layout: scanning splits %s",
                ", ".join(p.name for p in split_roots),
            )
            return split_roots

        logger.info("PlantDoc layout: scanning class folders under %s", self.root_dir)
        return [self.root_dir]

    def _collect_from_class_folder(
        self,
        class_name: str,
        class_dir: Path,
        paths: list[str],
        raw_labels: list[str],
    ) -> None:
        """
        Collect all images under a class directory (recursive).

        Args:
            class_name: Human-readable disease / plant class name.
            class_dir: Directory containing images for that class.
            paths: Accumulator for image path strings.
            raw_labels: Parallel list of string class names.

        Returns:
            None
        """
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file() or not self._is_image_file(file_path):
                continue
            if self.validate_images and not self._validate_image_file(file_path):
                continue
            paths.append(str(file_path))
            raw_labels.append(class_name)

    def _scan_class_directories(self, scan_root: Path) -> tuple[list[str], list[str]]:
        """
        Scan immediate subdirectories of scan_root as class folders.

        Args:
            scan_root: Directory whose children are class names.

        Returns:
            Tuple of (image_paths, raw_label_strings).
        """
        paths: list[str] = []
        raw_labels: list[str] = []

        if not scan_root.is_dir():
            logger.warning("Scan root does not exist: %s", scan_root)
            return paths, raw_labels

        class_dirs = sorted(
            [p for p in scan_root.iterdir() if p.is_dir()],
            key=lambda p: p.name.lower(),
        )

        for class_dir in class_dirs:
            if class_dir.name.lower() in self.PLANTVILLAGE_INTERMEDIATE_DIRS:
                continue
            if class_dir.name.lower() in self.PLANTDOC_SPLIT_DIRS:
                # Nested split inside a split (unusual); recurse one level
                nested_paths, nested_labels = self._scan_class_directories(class_dir)
                paths.extend(nested_paths)
                raw_labels.extend(nested_labels)
                continue

            self._collect_from_class_folder(class_dir.name, class_dir, paths, raw_labels)

        return paths, raw_labels

    def _scan_flat_images(self, scan_root: Path) -> tuple[list[str], list[str]]:
        """
        Fallback: assign class from parent folder name for all images under root.

        Args:
            scan_root: Root directory to walk recursively.

        Returns:
            Tuple of (image_paths, raw_label_strings).
        """
        paths: list[str] = []
        raw_labels: list[str] = []

        for file_path in scan_root.rglob("*"):
            if not file_path.is_file() or not self._is_image_file(file_path):
                continue
            if self.validate_images and not self._validate_image_file(file_path):
                continue
            class_name = file_path.parent.name
            if class_name.lower() in self.PLANTVILLAGE_INTERMEDIATE_DIRS:
                continue
            paths.append(str(file_path))
            raw_labels.append(class_name)

        return paths, raw_labels

    def scan(self) -> tuple[list[str], list[str]]:
        """
        Scan the dataset and return paths with string class labels.

        Returns:
            Tuple (image_paths, raw_labels) before integer encoding.

        Raises:
            ValueError: If no images are found.
        """
        all_paths: list[str] = []
        all_raw_labels: list[str] = []

        scan_roots = self._resolve_scan_roots()
        for scan_root in scan_roots:
            paths, labels = self._scan_class_directories(scan_root)
            all_paths.extend(paths)
            all_raw_labels.extend(labels)

        # Fallback for unusual layouts (only images, no class subdirs found)
        if not all_paths:
            logger.warning(
                "No class-folder images under %s; trying recursive parent-label scan.",
                self.root_dir,
            )
            all_paths, all_raw_labels = self._scan_flat_images(self.root_dir)

        if not all_paths:
            raise ValueError(
                f"No valid images found under {self.root_dir}. "
                "Ensure each class is a subfolder containing .jpg/.png images."
            )

        # Filter classes with too few samples
        counts = Counter(all_raw_labels)
        keep_classes = {
            cls
            for cls, count in counts.items()
            if count >= self.min_images_per_class
        }
        if len(keep_classes) < len(counts):
            dropped = set(counts) - keep_classes
            logger.warning(
                "Dropping classes with fewer than %d images: %s",
                self.min_images_per_class,
                sorted(dropped),
            )
            filtered_paths = []
            filtered_labels = []
            for path, label in zip(all_paths, all_raw_labels):
                if label in keep_classes:
                    filtered_paths.append(path)
                    filtered_labels.append(label)
            all_paths, all_raw_labels = filtered_paths, filtered_labels

        logger.info(
            "Scan complete: %d images, %d classes under %s",
            len(all_paths),
            len(keep_classes),
            self.root_dir,
        )
        return all_paths, all_raw_labels

    def encode_labels(self, raw_labels: list[str]) -> list[int]:
        """
        Fit LabelEncoder and convert string labels to integers.

        Args:
            raw_labels: List of class name strings.

        Returns:
            List of integer-encoded labels.
        """
        encoded = self.label_encoder.fit_transform(raw_labels)
        self.class_names = list(self.label_encoder.classes_)
        update_class_info(self.class_names)
        return encoded.tolist()

    def build_label_mapping(self) -> dict:
        """
        Build a JSON-serializable label mapping dict.

        Returns:
            Dict with class_names, class_to_idx, and idx_to_class keys.
        """
        class_to_idx = {
            name: int(idx) for idx, name in enumerate(self.class_names)
        }
        idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}
        self.label_mapping = {
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "dataset_root": str(self.root_dir),
            "random_seed": RANDOM_SEED,
        }
        return self.label_mapping

    def save_label_mapping(self, path: str | None = None) -> str:
        """
        Save label mapping to JSON and mirror class names to config path.

        Args:
            path: Output JSON path. Defaults to self.label_mapping_path.

        Returns:
            Path where the mapping was written.

        Raises:
            ValueError: If label mapping has not been built yet.
        """
        if not self.label_mapping:
            self.build_label_mapping()

        out_path = path or self.label_mapping_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.label_mapping, f, indent=2)

        with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump({"class_names": self.class_names}, f, indent=2)

        logger.info("Label mapping saved to %s", out_path)
        return out_path

    @classmethod
    def load_label_mapping(cls, path: str | None = None) -> dict:
        """
        Load a previously saved label mapping from JSON.

        Args:
            path: JSON file path. Defaults to LABEL_MAPPING_PATH.

        Returns:
            Label mapping dictionary.

        Raises:
            FileNotFoundError: If the mapping file does not exist.
        """
        map_path = path or LABEL_MAPPING_PATH
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Label mapping not found: {map_path}")

        with open(map_path, encoding="utf-8") as f:
            mapping = json.load(f)

        if "class_names" in mapping:
            update_class_info(mapping["class_names"])

        return mapping

    def get_class_distribution(self) -> dict[str, int]:
        """
        Count images per class using encoded labels.

        Returns:
            Dict mapping class name to image count.

        Raises:
            ValueError: If load() has not been called yet.
        """
        if not self.image_paths or not self.labels:
            raise ValueError("No data loaded. Call load() first.")

        counts: dict[str, int] = {}
        for label_idx in self.labels:
            name = self.class_names[int(label_idx)]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def print_class_distribution(self) -> None:
        """
        Log a formatted table of class names and image counts.

        Returns:
            None
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())

        logger.info("=" * 60)
        logger.info("CLASS DISTRIBUTION (%d images, %d classes)", total, len(distribution))
        logger.info("=" * 60)
        logger.info("%-40s %10s", "Class", "Count")
        logger.info("-" * 60)

        for class_name in sorted(distribution.keys()):
            count = distribution[class_name]
            pct = 100.0 * count / total if total else 0.0
            logger.info("%-40s %10d (%5.1f%%)", class_name, count, pct)

        logger.info("-" * 60)
        logger.info("%-40s %10d", "TOTAL", total)
        logger.info("=" * 60)

    def load(self) -> tuple[list[str], list[int]]:
        """
        Full pipeline: scan, encode labels, save mapping, print distribution.

        Returns:
            Tuple (image_paths, integer_labels).

        Raises:
            ValueError: If scanning finds no images.
        """
        raw_paths, raw_labels = self.scan()
        self.image_paths = raw_paths
        self.labels = self.encode_labels(raw_labels)
        self.build_label_mapping()
        self.save_label_mapping()
        self.print_class_distribution()
        return self.image_paths, self.labels


def load_dataset(
    dataset_name: str = "plantvillage",
    root_dir: str | None = None,
    validate_images: bool = True,
) -> tuple[list[str], list[int]]:
    """
    Convenience function to load a named dataset by config path.

    Args:
        dataset_name: 'plantvillage', 'plantdoc', or 'arecanut'.
        root_dir: Optional override path. Uses config default if None.
        validate_images: Whether to skip corrupt files.

    Returns:
        Tuple (image_paths, integer_labels).

    Raises:
        FileNotFoundError: If the dataset directory is missing.
        ValueError: If no images are found.
    """
    path_map = {
        "plantvillage": PLANTVILLAGE_PATH,
        "plantdoc": PLANTDOC_PATH,
        "arecanut": ARECANUT_PATH,
    }
    key = dataset_name.lower().strip()
    if key not in path_map:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    resolved_root = root_dir or path_map[key]
    loader = DataLoader(
        root_dir=resolved_root,
        dataset_name=key,
        validate_images=validate_images,
    )
    return loader.load()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan a plant disease image dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="plantvillage",
        choices=["plantvillage", "plantdoc", "arecanut"],
        help="Dataset name (uses config paths).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override dataset root directory.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip PIL image validation (faster scan).",
    )
    args = parser.parse_args()

    paths, labels = load_dataset(
        dataset_name=args.dataset,
        root_dir=args.root,
        validate_images=not args.no_validate,
    )
    logger.info("Loaded %d images with %d unique labels.", len(paths), len(set(labels)))
