"""Obsidian vault export."""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger("booksum.obsidian")

BOOKS_SUBDIR = "books"


class ObsidianError(Exception):
    """Raised when a vault export cannot be completed."""


def is_vault(path: str) -> bool:
    """True when ``path`` looks like an Obsidian vault."""
    return bool(path) and os.path.isdir(path) and os.path.isdir(os.path.join(path, ".obsidian"))


def export_to_vault(
    output_path: str,
    vault_path: str,
    images_dir: str | None = None,
) -> str:
    """Copy a rendered document (and its images) into ``<vault>/books``.

    Returns the destination path. Raises ``ObsidianError`` on failure.
    """
    if not is_vault(vault_path):
        raise ObsidianError(f"not a valid Obsidian vault: {vault_path}")
    if not os.path.exists(output_path):
        raise ObsidianError(f"nothing to export: {output_path}")

    books_dir = os.path.join(vault_path, BOOKS_SUBDIR)
    os.makedirs(books_dir, exist_ok=True)

    file_name = os.path.basename(output_path)
    target = os.path.join(books_dir, file_name)
    shutil.copy2(output_path, target)

    if images_dir and os.path.isdir(images_dir):
        base = os.path.basename(os.path.normpath(images_dir))
        target_images = os.path.join(books_dir, base)
        if os.path.exists(target_images):
            shutil.rmtree(target_images)
        shutil.copytree(images_dir, target_images)
        logger.info("copied images to %s", target_images)

    logger.info("exported %s to %s", output_path, target)
    return target
