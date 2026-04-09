#!/usr/bin/env python3
"""
Procedural Book Generator using Mitsuba

This script generates 3D book models with varying heights, widths, and depths,
and renders images of them using the Mitsuba physically-based renderer.
"""

import argparse
import os
import random
import sys

# Set LLVM library path for Mitsuba before importing it
# Check if current path is valid, if not find a valid one
current_llvm_path = os.environ.get("DRJIT_LIBLLVM_PATH")
if current_llvm_path is None or not os.path.exists(current_llvm_path):
    # Try common LLVM library locations
    llvm_paths = [
        "/usr/lib/llvm-17/lib/libLLVM.so",
        "/usr/lib/llvm-18/lib/libLLVM.so",
        "/usr/lib/llvm-14/lib/libLLVM.so",
        "/usr/lib/x86_64-linux-gnu/libLLVM-17.so",
        "/usr/lib/x86_64-linux-gnu/libLLVM-18.so",
    ]
    for llvm_path in llvm_paths:
        if os.path.exists(llvm_path):
            os.environ["DRJIT_LIBLLVM_PATH"] = llvm_path
            break

import mitsuba as mi
import numpy as np

# Set the scalar RGB variant for CPU rendering
mi.set_variant("scalar_rgb")


def create_book_scene(
    height: float = 25.0,
    width: float = 18.0,
    depth: float = 4.0,
    cover_color: tuple = (0.6, 0.1, 0.1),
    spine_color: tuple = None,
    page_color: tuple = (0.95, 0.92, 0.85),
    seed: int = None,
    image_width: int = 512,
    image_height: int = 512,
    sample_count: int = 128,
    rotation: float = 0.0,
    zoom: float = 1.0,
) -> dict:
    """
    Create a Mitsuba scene dictionary for a book with given dimensions.

    Args:
        height: Height of the book in cm (top to bottom when standing)
        width: Width of the book in cm (spine to page edge)
        depth: Thickness/depth of the book in cm (front to back cover)
        cover_color: RGB tuple for the main cover color
        spine_color: RGB tuple for the spine (defaults to darker cover color)
        page_color: RGB tuple for the page edges
        seed: Random seed for reproducibility
        image_width: Output image width in pixels
        image_height: Output image height in pixels
        sample_count: Number of samples per pixel for rendering
        rotation: Rotation of the book around the Y axis in degrees.
                  0 (default) shows the spine facing the camera.
                  Positive values rotate the book counter-clockwise.
        zoom: Zoom factor for the camera. Values < 1 zoom in (closer view),
              values > 1 zoom out (wider view). Default is 1.0.

    Returns:
        A Mitsuba scene dictionary
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if spine_color is None:
        # Make spine slightly darker than cover
        spine_color = tuple(max(0, c * 0.7) for c in cover_color)

    # Scale factors to convert cm to scene units (we use cm as our unit)
    # The book will be centered at origin, standing upright

    # Book body dimensions (scaled to scene units, 1 unit = 1 cm)
    book_scale_x = width / 2.0  # Half width for centering
    book_scale_y = height / 2.0  # Half height for centering
    book_scale_z = depth / 2.0  # Half depth for centering

    # Page edge strip dimensions (thin strip on the right side)
    page_strip_width = 0.3  # cm
    page_strip_depth = depth * 0.95  # Slightly inset

    # Camera positioning:
    # Default: spine faces the camera (spine is on negative X side of the book)
    # The base camera position is from the left side (negative X) looking at the book
    # Zoom scales the distance from the book
    base_camera_distance_x = width * 1.2
    base_camera_distance_z = depth * 2.0
    base_camera_height = height * 0.8

    # Calculate camera position with rotation (orbit around the book)
    # Convert rotation to radians for trig calculations
    rotation_rad = np.radians(rotation)
    cos_rot = np.cos(rotation_rad)
    sin_rot = np.sin(rotation_rad)

    # Default camera position (at rotation=0): facing the spine from negative X
    base_x = -base_camera_distance_x
    base_z = base_camera_distance_z

    # Rotate camera position around Y axis
    camera_x = (base_x * cos_rot - base_z * sin_rot) * zoom
    camera_y = base_camera_height
    camera_z = (base_x * sin_rot + base_z * cos_rot) * zoom

    # Camera target - look at the center of the book
    target_y = height * 0.3

    scene_dict = {
        "type": "scene",
        # Path tracer integrator
        "integrator": {
            "type": "path",
            "max_depth": 8,
        },
        # Camera setup - positioned to show the spine by default
        "sensor": {
            "type": "perspective",
            "fov": 40,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[camera_x, camera_y, camera_z],
                target=[0, target_y, 0],
                up=[0, 1, 0],
            ),
            "sampler": {
                "type": "independent",
                "sample_count": sample_count,
            },
            "film": {
                "type": "hdrfilm",
                "width": image_width,
                "height": image_height,
                "rfilter": {"type": "gaussian"},
            },
        },
        # Environment lighting - soft ambient light
        "emitter_env": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.3, 0.3, 0.35]},
        },
        # Main directional light (simulating window/studio light)
        "emitter_sun": {
            "type": "directional",
            "direction": [0.5, -1.0, -0.3],
            "irradiance": {"type": "rgb", "value": [4.0, 3.8, 3.5]},
        },
        # Fill light from opposite side
        "emitter_fill": {
            "type": "directional",
            "direction": [-0.8, -0.5, 0.4],
            "irradiance": {"type": "rgb", "value": [1.0, 1.0, 1.2]},
        },
        # Main book body (the covers and pages as one block)
        "book_body": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate([0, book_scale_y, 0])
            @ mi.ScalarTransform4f.scale([book_scale_x, book_scale_y, book_scale_z]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": list(cover_color),
                },
            },
        },
        # Spine detail - a slightly different colored strip on the left
        "spine": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [-book_scale_x + 0.05, book_scale_y, 0]
            )
            @ mi.ScalarTransform4f.scale(
                [0.1, book_scale_y * 0.98, book_scale_z * 0.98]
            ),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": list(spine_color),
                },
            },
        },
        # Page edges on the right side
        "page_edges": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [book_scale_x - page_strip_width / 2, book_scale_y, 0]
            )
            @ mi.ScalarTransform4f.scale(
                [page_strip_width / 2, book_scale_y * 0.96, page_strip_depth / 2]
            ),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": list(page_color),
                },
            },
        },
        # Top and bottom page edge highlights
        "page_edges_top": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [0, height - 0.1, 0]
            )
            @ mi.ScalarTransform4f.scale([book_scale_x * 0.95, 0.1, book_scale_z * 0.9]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": list(page_color),
                },
            },
        },
        "page_edges_bottom": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate([0, 0.1, 0])
            @ mi.ScalarTransform4f.scale([book_scale_x * 0.95, 0.1, book_scale_z * 0.9]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": list(page_color),
                },
            },
        },
        # Optional: Add a subtle title area on the cover (darker rectangle)
        "title_area": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [width * 0.05, height * 0.55, depth / 2 + 0.05]
            )
            @ mi.ScalarTransform4f.scale([width * 0.35, height * 0.12, 0.1]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [c * 0.5 for c in cover_color],
                },
            },
        },
        # Ground plane to catch shadows
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0])
            @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            @ mi.ScalarTransform4f.scale([50, 50, 1]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.15, 0.15, 0.15],
                },
            },
        },
    }

    return scene_dict


def create_bookshelf_scene(
    books: list,
    shelf_width: float = 100.0,
    shelf_depth: float = 25.0,
    shelf_height: float = 100.0,
    num_shelves: int = 4,
    image_width: int = 1024,
    image_height: int = 512,
    sample_count: int = 128,
    zoom: float = 1.0,
    rotation: float = 0.0,
    fill_direction: str = "random",
) -> dict:
    """
    Create a Mitsuba scene dictionary for a full enclosed bookshelf with multiple shelves and books.

    Args:
        books: List of book dictionaries for ALL shelves combined, each containing:
            - height: Book height in cm
            - width: Book width in cm (spine to page edge)
            - depth: Book thickness in cm
            - cover_color: RGB tuple for cover color
            - spine_color: RGB tuple for spine (optional)
            - page_color: RGB tuple for page edges (optional)
            - lean: Lean angle in degrees for tilting (optional, default 0)
            - shelf_index: Which shelf to place book on (optional)
        shelf_width: Width of the bookshelf in cm
        shelf_depth: Depth of the bookshelf in cm
        shelf_height: Total height of the bookshelf in cm
        num_shelves: Number of shelves (including bottom)
        image_width: Output image width in pixels
        image_height: Output image height in pixels
        sample_count: Number of samples per pixel for rendering
        zoom: Zoom factor for the camera
        rotation: Camera rotation around the scene in degrees
        fill_direction: Direction books fill from - "left", "right", "both", or "random"

    Returns:
        A Mitsuba scene dictionary
    """
    # Wood colors for the bookshelf
    wood_color = [0.4, 0.25, 0.15]  # Main wood color
    wood_dark = [0.35, 0.22, 0.13]  # Darker wood for back/sides
    panel_thickness = 2.0  # Thickness of wood panels
    
    # Calculate shelf spacing
    shelf_spacing = shelf_height / num_shelves
    
    # Determine fill direction
    if fill_direction == "random":
        fill_direction = random.choice(["left", "right", "both"])

    # Calculate camera position with rotation (orbit around the shelf center)
    # Default camera position: facing the shelf from positive Z
    base_y = shelf_height * 0.5
    base_z = shelf_depth * 4.0 * zoom

    # Rotate camera position around the shelf center
    rotation_rad = np.radians(rotation)
    cos_rot = np.cos(rotation_rad)
    sin_rot = np.sin(rotation_rad)

    # Camera position relative to shelf center
    rel_x = 0  # Centered horizontally
    rel_z = shelf_depth * 4.0 * zoom

    # Apply rotation
    camera_x = shelf_width * 0.5 + (rel_x * cos_rot - rel_z * sin_rot)
    camera_y = base_y
    camera_z = (rel_x * sin_rot + rel_z * cos_rot)

    scene_objects = {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": 8,
        },
        # Camera setup - positioned in FRONT of the shelf looking at the spines
        "sensor": {
            "type": "perspective",
            "fov": 50,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[camera_x, camera_y, camera_z],
                target=[shelf_width * 0.5, shelf_height * 0.4, 0],
                up=[0, 1, 0],
            ),
            "sampler": {
                "type": "independent",
                "sample_count": sample_count,
            },
            "film": {
                "type": "hdrfilm",
                "width": image_width,
                "height": image_height,
                "rfilter": {"type": "gaussian"},
            },
        },
        # Environment lighting - soft ambient light
        "emitter_env": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.4, 0.4, 0.45]},
        },
        # Main directional light from front (toward the spines)
        "emitter_main": {
            "type": "directional",
            "direction": [0, -0.5, -1.0],
            "irradiance": {"type": "rgb", "value": [3.0, 2.9, 2.7]},
        },
        # Fill light from above
        "emitter_fill": {
            "type": "directional",
            "direction": [0.3, -1.0, -0.3],
            "irradiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
        # === FULL BOOKSHELF STRUCTURE ===
        # Back panel (full height)
        "shelf_back": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [shelf_width / 2, shelf_height / 2, -shelf_depth / 2 - panel_thickness / 2]
            )
            @ mi.ScalarTransform4f.scale([shelf_width / 2, shelf_height / 2, panel_thickness / 2]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": wood_dark},
            },
        },
        # Left side panel
        "shelf_left": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [-panel_thickness / 2, shelf_height / 2, 0]
            )
            @ mi.ScalarTransform4f.scale([panel_thickness / 2, shelf_height / 2, shelf_depth / 2]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": wood_color},
            },
        },
        # Right side panel
        "shelf_right": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [shelf_width + panel_thickness / 2, shelf_height / 2, 0]
            )
            @ mi.ScalarTransform4f.scale([panel_thickness / 2, shelf_height / 2, shelf_depth / 2]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": wood_color},
            },
        },
        # Top panel
        "shelf_top": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [shelf_width / 2, shelf_height + panel_thickness / 2, 0]
            )
            @ mi.ScalarTransform4f.scale([shelf_width / 2 + panel_thickness, panel_thickness / 2, shelf_depth / 2]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": wood_color},
            },
        },
        # Bottom panel (base)
        "shelf_bottom": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [shelf_width / 2, -panel_thickness / 2, 0]
            )
            @ mi.ScalarTransform4f.scale([shelf_width / 2 + panel_thickness, panel_thickness / 2, shelf_depth / 2]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": wood_color},
            },
        },
    }

    # Add individual shelf boards
    for shelf_idx in range(num_shelves):
        shelf_y = shelf_idx * shelf_spacing
        scene_objects[f"shelf_board_{shelf_idx}"] = {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(
                [shelf_width / 2, shelf_y, 0]
            )
            @ mi.ScalarTransform4f.scale([shelf_width / 2, 1, shelf_depth / 2]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": wood_color},
            },
        }

    # Group books by shelf
    books_by_shelf = {}
    for book in books:
        shelf_idx = book.get("shelf_index", 0)
        if shelf_idx not in books_by_shelf:
            books_by_shelf[shelf_idx] = []
        books_by_shelf[shelf_idx].append(book)

    # Render books on each shelf
    book_counter = 0
    for shelf_idx in range(num_shelves):
        shelf_y = shelf_idx * shelf_spacing
        shelf_books = books_by_shelf.get(shelf_idx, [])
        
        if not shelf_books:
            continue

        # Calculate starting positions based on fill direction
        current_x = 2.0  # Default starting position
        if fill_direction == "left":
            # Fill from left side
            current_x = 2.0
        elif fill_direction == "right":
            # Fill from right side - calculate total width needed including lean space
            # This needs to match the gap calculation logic in the placement loop
            total_depth = 0.0
            for idx, b in enumerate(shelf_books):
                depth = b.get("depth", 4.0)
                lean = b.get("lean", 0.0)
                height = b.get("height", 25.0)

                # Base gap and depth
                total_depth += depth + 0.2

                # If this book leans right, only add gap if next book doesn't also lean right
                if lean < 0:
                    if idx < len(shelf_books) - 1:
                        next_book = shelf_books[idx + 1]
                        next_lean = next_book.get("lean", 0.0)
                        if next_lean >= 0:  # Next book doesn't lean right
                            lean_ext = height * np.sin(np.radians(abs(lean)))
                            total_depth += lean_ext

                # If next book leans left, only add gap if current book doesn't also lean left
                if idx < len(shelf_books) - 1:
                    next_book = shelf_books[idx + 1]
                    next_lean = next_book.get("lean", 0.0)
                    if next_lean > 0 and lean <= 0:  # Next leans left, current doesn't
                        next_height = next_book.get("height", 25.0)
                        next_lean_ext = next_height * np.sin(np.radians(next_lean))
                        total_depth += next_lean_ext

            current_x = shelf_width - total_depth - 2.0
        else:  # "both"
            # Split books between left and right sides
            # Just mark which side they belong to, positions calculated later
            mid_point = len(shelf_books) // 2
            for i, book in enumerate(shelf_books):
                book["_fill_side"] = "left" if i < mid_point else "right"

        # First pass: normalize lean angles for consecutive same-direction leans
        # Books leaning in the same direction should have the same angle
        i = 0
        while i < len(shelf_books):
            lean = shelf_books[i].get("lean", 0.0)
            if lean == 0:
                i += 1
                continue

            # Find consecutive books with same lean direction
            j = i + 1
            while j < len(shelf_books):
                next_lean = shelf_books[j].get("lean", 0.0)
                # Same direction: both positive (lean left) or both negative (lean right)
                if (lean > 0 and next_lean > 0) or (lean < 0 and next_lean < 0):
                    j += 1
                else:
                    break

            # Normalize all books in this group to the same angle
            if j > i + 1:
                # Use the angle of the first book in the group
                normalized_lean = lean
                for k in range(i, j):
                    shelf_books[k]["lean"] = normalized_lean

            i = j

        # Second pass: determine final lean values based on available support
        # A book needs support in the direction it's leaning
        for idx, book in enumerate(shelf_books):
            lean = book.get("lean", 0.0)
            if lean == 0:
                continue

            # Determine if this book has support based on position and fill direction
            # For left fill: first book is against left wall
            # For right fill: last book is against right wall
            # For both fill: need to check _fill_side

            is_first = (idx == 0)
            is_last = (idx == len(shelf_books) - 1)

            # Check wall support based on fill direction
            against_left_wall = False
            against_right_wall = False

            if fill_direction == "left":
                against_left_wall = is_first
            elif fill_direction == "right":
                against_right_wall = is_last
            else:  # "both"
                fill_side = book.get("_fill_side", "left")
                if fill_side == "left":
                    against_left_wall = is_first
                else:
                    against_right_wall = is_last

            # A book can lean left if there's a book to its left OR it's against the left wall
            # But it must actually REACH the support (touch it)
            can_lean_left = (idx > 0) or against_left_wall
            # A book can lean right if there's a book to its right OR it's against the right wall
            can_lean_right = (idx < len(shelf_books) - 1) or against_right_wall

            # Adjust lean based on available support
            if lean > 0 and not can_lean_left:  # Top leans left, needs left support
                if can_lean_right:
                    book["lean"] = -lean  # Flip to lean right
                else:
                    book["lean"] = 0.0  # No support available
            elif lean < 0 and not can_lean_right:  # Top leans right, needs right support
                if can_lean_left:
                    book["lean"] = -lean  # Flip to lean left
                else:
                    book["lean"] = 0.0  # No support available

        # Third pass: check shelf boundary clipping and adjust positions/leans
        panel_thickness = 2.0
        for idx, book in enumerate(shelf_books):
            lean = book.get("lean", 0.0)
            height = book.get("height", 25.0)
            depth = book.get("depth", 4.0)

            if lean == 0:
                continue

            lean_ext = height * np.sin(np.radians(abs(lean)))

            # Check if this book would clip into shelf sides
            # We'll check this during position calculation, but for now
            # disable leans that would definitely clip
            is_first = (idx == 0)
            is_last = (idx == len(shelf_books) - 1)

            if fill_direction == "left":
                against_left_wall = is_first
                against_right_wall = False
            elif fill_direction == "right":
                against_left_wall = False
                against_right_wall = is_last
            else:  # "both"
                fill_side = book.get("_fill_side", "left")
                against_left_wall = is_first and fill_side == "left"
                against_right_wall = is_last and fill_side == "right"

            # If leaning left and against left wall, check if lean extension clips into wall
            if lean > 0 and against_left_wall:
                # Book's top would extend left by lean_ext
                # The wall is at x=0 (with panel at x < panel_thickness)
                # This is actually fine - the book can lean against the wall
                pass

            # If leaning right and against right wall, check if lean extension clips into wall
            if lean < 0 and against_right_wall:
                # Book's top would extend right by lean_ext
                # This is fine - the book can lean against the right wall
                pass

        # Fourth pass: calculate positions based on final leans
        # Books leaning in the same direction should touch and support each other
        # Books leaning toward each other need space for the lean
        book_positions = []

        if fill_direction == "both":
            # Handle left side books (placed left to right)
            left_books = [b for b in shelf_books if b.get("_fill_side") == "left"]
            right_books = [b for b in shelf_books if b.get("_fill_side") == "right"]

            left_x = 2.0  # Start with small margin from left wall
            for idx, book in enumerate(left_books):
                height = book.get("height", 25.0)
                depth = book.get("depth", 4.0)
                lean = book.get("lean", 0.0)

                x_start = left_x
                x_end = x_start + depth
                book_positions.append((book, x_start, x_end))

                # Calculate gap to next book
                gap = 0.2  # Small base gap

                # If current book leans right, its top extends toward next book
                # Need extra space unless next book also leans right
                if lean < 0:
                    if idx < len(left_books) - 1:
                        next_book = left_books[idx + 1]
                        next_lean = next_book.get("lean", 0.0)
                        if next_lean >= 0:  # Next book doesn't lean right
                            lean_ext = height * np.sin(np.radians(abs(lean)))
                            gap += lean_ext

                # If next book leans left, its top will extend toward current book
                # Need extra space unless current book also leans left
                if idx < len(left_books) - 1:
                    next_book = left_books[idx + 1]
                    next_lean = next_book.get("lean", 0.0)
                    if next_lean > 0 and lean <= 0:  # Next leans left, current doesn't
                        next_height = next_book.get("height", 25.0)
                        next_lean_ext = next_height * np.sin(np.radians(next_lean))
                        gap += next_lean_ext

                left_x = x_end + gap

            # Handle right side books (placed right to left)
            right_x = shelf_width - 2.0  # Start with small margin from right wall
            for idx, book in enumerate(reversed(right_books)):
                height = book.get("height", 25.0)
                depth = book.get("depth", 4.0)
                lean = book.get("lean", 0.0)

                x_end = right_x
                x_start = x_end - depth
                book_positions.append((book, x_start, x_end))

                # Calculate gap to previous book (in original order, next in reversed)
                gap = 0.2

                # If current book leans left, its top extends toward the left
                # Need extra space unless next book (to the left) also leans left
                if lean > 0:
                    if idx < len(right_books) - 1:
                        prev_book = right_books[len(right_books) - 1 - idx - 1]
                        prev_lean = prev_book.get("lean", 0.0)
                        if prev_lean <= 0:  # Next book doesn't lean left
                            lean_ext = height * np.sin(np.radians(lean))
                            gap += lean_ext

                # If next book (to the left) leans right, its top extends toward current book
                # Need extra space unless current book also leans right
                if idx < len(right_books) - 1:
                    prev_book = right_books[len(right_books) - 1 - idx - 1]
                    prev_lean = prev_book.get("lean", 0.0)
                    if prev_lean < 0 and lean >= 0:  # Next leans right, current doesn't
                        prev_height = prev_book.get("height", 25.0)
                        prev_lean_ext = prev_height * np.sin(np.radians(abs(prev_lean)))
                        gap += prev_lean_ext

                right_x = x_start - gap
        else:
            # Left or right fill - books placed sequentially
            temp_x = 2.0 if fill_direction == "left" else current_x

            for idx, book in enumerate(shelf_books):
                height = book.get("height", 25.0)
                depth = book.get("depth", 4.0)
                lean = book.get("lean", 0.0)

                x_start = temp_x
                x_end = x_start + depth
                book_positions.append((book, x_start, x_end))

                # Calculate gap to next book
                gap = 0.2

                # If current book leans right, its top extends toward next book
                # Need extra space unless next book also leans right
                if lean < 0:
                    if idx < len(shelf_books) - 1:
                        next_book = shelf_books[idx + 1]
                        next_lean = next_book.get("lean", 0.0)
                        if next_lean >= 0:  # Next book doesn't lean right
                            lean_ext = height * np.sin(np.radians(abs(lean)))
                            gap += lean_ext

                # If next book leans left, its top will extend toward current book
                # Need extra space unless current book also leans left
                if idx < len(shelf_books) - 1:
                    next_book = shelf_books[idx + 1]
                    next_lean = next_book.get("lean", 0.0)
                    if next_lean > 0 and lean <= 0:  # Next leans left, current doesn't
                        next_height = next_book.get("height", 25.0)
                        next_lean_ext = next_height * np.sin(np.radians(next_lean))
                        gap += next_lean_ext

                temp_x = x_end + gap

        # Fifth pass: render books
        for book, x_start, x_end in book_positions:
            height = book.get("height", 25.0)
            width = book.get("width", 18.0)
            depth = book.get("depth", 4.0)
            cover_color = book.get("cover_color", (0.6, 0.1, 0.1))
            spine_color = book.get("spine_color")
            page_color = book.get("page_color", (0.95, 0.92, 0.85))
            lean = book.get("lean", 0.0)

            if spine_color is None:
                spine_color = tuple(max(0, c * 0.7) for c in cover_color)

            # Book dimensions
            book_scale_x = depth / 2.0
            book_scale_y = height / 2.0
            book_scale_z = width / 2.0

            # Use pre-calculated position
            book_x = x_start + book_scale_x

            # Ensure book fits on shelf (height constraint)
            max_book_height = shelf_spacing - 3.0  # Leave some clearance
            if height > max_book_height:
                height = max_book_height
                book_scale_y = height / 2.0

            # Base transform for book position on this shelf
            base_transform = mi.ScalarTransform4f.translate([book_x, shelf_y + book_scale_y + 1.0, 0])

            # Apply lean/tilt if specified
            if lean != 0:
                # Lean around the bottom edge of the book
                # Pivot point is at the bottom of the book
                lean_rad = np.radians(lean)
                # First translate to pivot, rotate, then translate back
                base_transform = (
                    mi.ScalarTransform4f.translate([book_x, shelf_y + 1.0, 0])
                    @ mi.ScalarTransform4f.rotate([0, 0, 1], lean)  # Rotate around Z axis
                    @ mi.ScalarTransform4f.translate([0, book_scale_y, 0])
                )

            # Add book objects to scene with unique names
            prefix = f"book_{book_counter:03d}_"
            book_counter += 1

            # Main book body (the cover)
            scene_objects[f"{prefix}body"] = {
                "type": "cube",
                "to_world": base_transform
                @ mi.ScalarTransform4f.scale([book_scale_x, book_scale_y, book_scale_z]),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": list(cover_color),
                    },
                },
            }

            # Spine - facing +Z toward the camera
            scene_objects[f"{prefix}spine"] = {
                "type": "cube",
                "to_world": base_transform
                @ mi.ScalarTransform4f.translate([0, 0, book_scale_z + 0.02])
                @ mi.ScalarTransform4f.scale([book_scale_x * 0.96, book_scale_y * 0.98, 0.1]),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": list(spine_color),
                    },
                },
            }

            # Page edges on the back side (-Z, against the back of shelf)
            scene_objects[f"{prefix}page_edges"] = {
                "type": "cube",
                "to_world": base_transform
                @ mi.ScalarTransform4f.translate([0, 0, -book_scale_z + 0.2])
                @ mi.ScalarTransform4f.scale([book_scale_x * 0.9, book_scale_y * 0.96, 0.15]),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": list(page_color),
                    },
                },
            }

            # Top page edge
            scene_objects[f"{prefix}page_edges_top"] = {
                "type": "cube",
                "to_world": base_transform
                @ mi.ScalarTransform4f.translate([0, book_scale_y - 0.1, 0])
                @ mi.ScalarTransform4f.scale([book_scale_x * 0.9, 0.1, book_scale_z * 0.9]),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": list(page_color),
                    },
                },
            }

            # Decorative title band on the spine
            scene_objects[f"{prefix}title_band"] = {
                "type": "cube",
                "to_world": base_transform
                @ mi.ScalarTransform4f.translate([0, height * 0.1, book_scale_z + 0.06])
                @ mi.ScalarTransform4f.scale([book_scale_x * 0.7, height * 0.2, 0.08]),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [min(1.0, c * 1.2) for c in spine_color],
                    },
                },
            }

    return scene_objects


def render_book(
    height: float,
    width: float,
    depth: float,
    output_path: str,
    cover_color: tuple = None,
    seed: int = None,
    image_width: int = 512,
    image_height: int = 512,
    rotation: float = 0.0,
    zoom: float = 1.0,
) -> None:
    """
    Generate and render a book with the given dimensions, saving to output_path.

    Args:
        height: Height of the book in cm
        width: Width of the book in cm
        depth: Thickness of the book in cm
        output_path: Path to save the rendered image (PNG format)
        cover_color: Optional RGB tuple for cover color (random if None)
        seed: Random seed for reproducibility
        image_width: Output image width
        image_height: Output image height
        rotation: Rotation of the book around the Y axis in degrees.
                  0 (default) shows the spine facing the camera.
        zoom: Zoom factor for the camera. Values < 1 zoom in, > 1 zoom out.
    """
    if cover_color is None:
        # Generate a pleasant random book cover color
        # Favor reds, blues, greens, and earth tones
        hue_type = random.choice(["red", "blue", "green", "brown", "purple"])
        if hue_type == "red":
            cover_color = (
                random.uniform(0.5, 0.8),
                random.uniform(0.05, 0.15),
                random.uniform(0.05, 0.15),
            )
        elif hue_type == "blue":
            cover_color = (
                random.uniform(0.05, 0.2),
                random.uniform(0.2, 0.5),
                random.uniform(0.5, 0.8),
            )
        elif hue_type == "green":
            cover_color = (
                random.uniform(0.05, 0.2),
                random.uniform(0.4, 0.7),
                random.uniform(0.1, 0.3),
            )
        elif hue_type == "brown":
            cover_color = (
                random.uniform(0.4, 0.6),
                random.uniform(0.25, 0.4),
                random.uniform(0.1, 0.2),
            )
        else:  # purple
            cover_color = (
                random.uniform(0.4, 0.7),
                random.uniform(0.1, 0.25),
                random.uniform(0.5, 0.8),
            )

    print(f"Generating book: {width:.1f}cm x {height:.1f}cm x {depth:.1f}cm")
    print(f"Cover color: ({cover_color[0]:.2f}, {cover_color[1]:.2f}, {cover_color[2]:.2f})")
    print(f"Rotation: {rotation:.1f}°, Zoom: {zoom:.2f}x")

    scene_dict = create_book_scene(
        height=height,
        width=width,
        depth=depth,
        cover_color=cover_color,
        seed=seed,
        image_width=image_width,
        image_height=image_height,
        rotation=rotation,
        zoom=zoom,
    )

    # Load and render the scene
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene)

    # Convert to 8-bit and save as PNG
    bitmap = mi.Bitmap(image)
    bitmap = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8)
    bitmap.write(output_path)

    print(f"Saved rendered image to: {output_path}")


def render_bookshelf(
    books: list,
    output_path: str,
    shelf_width: float = 100.0,
    shelf_depth: float = 25.0,
    shelf_height: float = 100.0,
    num_shelves: int = 4,
    image_width: int = 1024,
    image_height: int = 512,
    zoom: float = 1.0,
    rotation: float = 0.0,
    fill_direction: str = "random",
) -> None:
    """
    Render a full enclosed bookshelf with multiple shelves and books.

    Args:
        books: List of book dictionaries with book properties
        output_path: Path to save the rendered image
        shelf_width: Width of the bookshelf in cm
        shelf_depth: Depth of the bookshelf in cm
        shelf_height: Total height of the bookshelf in cm
        num_shelves: Number of shelves (including bottom)
        image_width: Output image width
        image_height: Output image height
        zoom: Zoom factor for the camera
        rotation: Camera rotation around the scene in degrees
        fill_direction: Direction books fill from - "left", "right", "both", or "random"
    """
    print(f"Generating bookshelf with {len(books)} books on {num_shelves} shelves")
    print(f"Bookshelf dimensions: {shelf_width:.1f}cm x {shelf_depth:.1f}cm x {shelf_height:.1f}cm")

    scene_dict = create_bookshelf_scene(
        books=books,
        shelf_width=shelf_width,
        shelf_depth=shelf_depth,
        shelf_height=shelf_height,
        num_shelves=num_shelves,
        image_width=image_width,
        image_height=image_height,
        zoom=zoom,
        rotation=rotation,
        fill_direction=fill_direction,
    )

    # Load and render the scene
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene)

    # Convert to 8-bit and save as PNG
    bitmap = mi.Bitmap(image)
    bitmap = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8)
    bitmap.write(output_path)

    print(f"Saved rendered image to: {output_path}")


def generate_random_bookshelf(
    num_books: int = 12,
    output_path: str = "bookshelf.png",
    shelf_width: float = 100.0,
    shelf_depth: float = 25.0,
    shelf_height: float = 100.0,
    num_shelves: int = 4,
    seed: int = None,
    rotation: float = 0.0,
    fill_percentage: float = 70.0,
    fill_direction: str = "random",
    width_variation: float = 0.0,
    height_variation: float = 0.0,
    lean_probability: float = 0.15,
    lean_angle_range: tuple = (5.0, 15.0),
) -> None:
    """
    Generate a full enclosed bookshelf with randomly generated books.

    Args:
        num_books: Approximate number of books to place on the bookshelf
        output_path: Path to save the rendered image
        shelf_width: Base width of the bookshelf in cm
        shelf_depth: Base depth of the bookshelf in cm
        shelf_height: Base height of the bookshelf in cm
        num_shelves: Number of shelves (including bottom)
        seed: Random seed for reproducibility
        rotation: Camera rotation around the scene in degrees
        fill_percentage: Minimum percentage of shelf to fill (0-100)
        fill_direction: Direction books fill from - "left", "right", "both", or "random"
        width_variation: Random variation in shelf width (+/- cm)
        height_variation: Random variation in shelf height (+/- cm)
        lean_probability: Probability of a book leaning (0.0 to 1.0)
        lean_angle_range: Range of lean angles in degrees (min, max)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        # Generate a random seed for reproducibility
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)

    print(f"Seed: {seed}")

    # Apply size variation to bookshelf
    actual_width = shelf_width + random.uniform(-width_variation, width_variation)
    actual_height = shelf_height + random.uniform(-height_variation, height_variation)

    # Calculate shelf spacing
    shelf_spacing = actual_height / num_shelves

    # Determine fill direction
    actual_fill_direction = fill_direction
    if fill_direction == "random":
        actual_fill_direction = random.choice(["left", "right", "both"])

    # Calculate how many books we need based on fill percentage
    # Average book depth is about 3-4 cm
    avg_book_depth = 3.5
    usable_width = actual_width - 4.0  # Account for margins
    target_fill_width = usable_width * (fill_percentage / 100.0)
    books_per_shelf_estimate = int(target_fill_width / avg_book_depth)

    # Distribute books across shelves
    total_books_needed = books_per_shelf_estimate * num_shelves
    # Adjust num_books if it's significantly different from what fill_percentage implies
    if num_books < total_books_needed * 0.5:
        num_books = int(total_books_needed * 0.7)  # Use at least 70% of calculated

    books = []
    books_per_shelf = num_books // num_shelves
    extra_books = num_books % num_shelves

    # Color palette for books
    color_types = ["red", "blue", "green", "brown", "purple", "orange", "teal", "navy", "maroon", "forest"]

    def generate_cover_color():
        hue_type = random.choice(color_types)
        if hue_type == "red":
            return (random.uniform(0.5, 0.8), random.uniform(0.05, 0.15), random.uniform(0.05, 0.15))
        elif hue_type == "blue":
            return (random.uniform(0.05, 0.2), random.uniform(0.2, 0.5), random.uniform(0.5, 0.8))
        elif hue_type == "green":
            return (random.uniform(0.05, 0.2), random.uniform(0.4, 0.7), random.uniform(0.1, 0.3))
        elif hue_type == "brown":
            return (random.uniform(0.4, 0.6), random.uniform(0.25, 0.4), random.uniform(0.1, 0.2))
        elif hue_type == "purple":
            return (random.uniform(0.4, 0.7), random.uniform(0.1, 0.25), random.uniform(0.5, 0.8))
        elif hue_type == "orange":
            return (random.uniform(0.7, 0.9), random.uniform(0.4, 0.6), random.uniform(0.05, 0.15))
        elif hue_type == "teal":
            return (random.uniform(0.05, 0.2), random.uniform(0.5, 0.7), random.uniform(0.5, 0.7))
        elif hue_type == "navy":
            return (random.uniform(0.05, 0.15), random.uniform(0.1, 0.25), random.uniform(0.4, 0.6))
        elif hue_type == "maroon":
            return (random.uniform(0.5, 0.7), random.uniform(0.05, 0.15), random.uniform(0.1, 0.25))
        else:  # forest
            return (random.uniform(0.05, 0.15), random.uniform(0.3, 0.5), random.uniform(0.1, 0.25))

    # Generate books for each shelf
    for shelf_idx in range(num_shelves):
        shelf_book_count = books_per_shelf + (1 if shelf_idx < extra_books else 0)
        max_book_height = shelf_spacing - 4.0  # Leave clearance

        for i in range(shelf_book_count):
            # Generate realistic book dimensions
            # Height varies but must fit on shelf
            height = random.uniform(min(15.0, max_book_height * 0.6), min(max_book_height, 30.0))
            width = random.uniform(12.0, 22.0)
            depth = random.uniform(1.5, 6.0)

            cover_color = generate_cover_color()

            # Determine if this book should lean
            lean = 0.0
            if i > 0 and random.random() < lean_probability:
                lean = random.uniform(lean_angle_range[0], lean_angle_range[1]) * random.choice([-1, 1])

            book = {
                "height": height,
                "width": width,
                "depth": depth,
                "cover_color": cover_color,
                "lean": lean,
                "shelf_index": shelf_idx,
            }
            books.append(book)

    print(f"Fill direction: {actual_fill_direction}")
    print(f"Fill percentage target: {fill_percentage}%")

    render_bookshelf(
        books=books,
        output_path=output_path,
        shelf_width=actual_width,
        shelf_depth=shelf_depth,
        shelf_height=actual_height,
        num_shelves=num_shelves,
        rotation=rotation,
        fill_direction=actual_fill_direction,
    )


def generate_random_books(
    num_books: int = 5,
    output_dir: str = "book_renders",
    rotation: float = None,
    zoom: float = 1.0,
    rotation_range: tuple = (-45.0, 45.0),
    seed: int = None,
) -> None:
    """
    Generate multiple books with random dimensions and colors.

    Args:
        num_books: Number of books to generate
        output_dir: Directory to save rendered images
        rotation: Fixed rotation for all books in degrees. If None, random rotation
                  within rotation_range is used for each book.
        zoom: Zoom factor for the camera.
        rotation_range: Tuple of (min, max) degrees for random rotation when
                        rotation is not specified.
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        # Generate a random seed for reproducibility
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)

    print(f"Seed: {seed}")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_books):
        # Generate realistic book dimensions
        # Typical ranges:
        # - Height: 15-35 cm (pocket books to large coffee table books)
        # - Width: 10-25 cm
        # - Depth: 1-8 cm (varies greatly with page count and paper type)

        height = random.uniform(16.0, 32.0)
        width = random.uniform(12.0, 24.0)
        depth = random.uniform(1.5, 7.0)

        # Determine rotation for this book
        if rotation is not None:
            book_rotation = rotation
        else:
            book_rotation = random.uniform(rotation_range[0], rotation_range[1])

        output_path = os.path.join(output_dir, f"book_{i:03d}.png")

        render_book(
            height=height,
            width=width,
            depth=depth,
            output_path=output_path,
            seed=i,
            rotation=book_rotation,
            zoom=zoom,
        )
        print()  # Blank line between books


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural book models and render images using Mitsuba"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Book height in cm (default: random 16-32)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Book width in cm (default: random 12-24)",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Book depth/thickness in cm (default: random 1.5-7)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="book_render.png",
        help="Output image path (default: book_render.png)",
    )
    parser.add_argument(
        "--num-books",
        "-n",
        type=int,
        default=1,
        help="Number of books to generate (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="book_renders",
        help="Output directory when generating multiple books (default: book_renders)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=512,
        help="Output image width in pixels (default: 512)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=512,
        help="Output image height in pixels (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cover-color",
        type=str,
        default=None,
        help="Cover color as comma-separated RGB values (0-1), e.g., '0.6,0.1,0.1'",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="Rotation of the book around the Y axis in degrees. "
             "0 (default) shows the spine facing the camera. "
             "Positive values rotate counter-clockwise.",
    )
    parser.add_argument(
        "--rotation-range",
        type=str,
        default=None,
        help="For multiple books: random rotation range as 'min,max' in degrees "
             "(e.g., '-30,30'). Ignored if --rotation is specified.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor for the camera. Values < 1 zoom in (closer view), "
             "values > 1 zoom out (wider view). Default is 1.0.",
    )
    parser.add_argument(
        "--bookshelf",
        action="store_true",
        help="Generate a full enclosed bookshelf with multiple shelves instead of individual books",
    )
    parser.add_argument(
        "--shelf-width",
        type=float,
        default=100.0,
        help="Width of the bookshelf in cm (default: 100)",
    )
    parser.add_argument(
        "--shelf-depth",
        type=float,
        default=25.0,
        help="Depth of the bookshelf in cm (default: 25)",
    )
    parser.add_argument(
        "--shelf-height",
        type=float,
        default=100.0,
        help="Height of the bookshelf in cm (default: 100)",
    )
    parser.add_argument(
        "--num-shelves",
        type=int,
        default=4,
        help="Number of shelves in the bookshelf (default: 4)",
    )
    parser.add_argument(
        "--fill-percentage",
        type=float,
        default=70.0,
        help="Minimum percentage of each shelf to fill with books (0-100, default: 70)",
    )
    parser.add_argument(
        "--fill-direction",
        type=str,
        choices=["left", "right", "both", "random"],
        default="random",
        help="Direction to fill shelves: left, right, both (split), or random (default: random)",
    )
    parser.add_argument(
        "--width-variation",
        type=float,
        default=0.0,
        help="Random variation in bookshelf width (+/- cm, default: 0)",
    )
    parser.add_argument(
        "--height-variation",
        type=float,
        default=0.0,
        help="Random variation in bookshelf height (+/- cm, default: 0)",
    )
    parser.add_argument(
        "--lean-probability",
        type=float,
        default=0.15,
        help="Probability of a book leaning/tilting (0.0-1.0, default: 0.15)",
    )
    parser.add_argument(
        "--lean-angle-range",
        type=str,
        default="5,15",
        help="Range of lean angles as 'min,max' in degrees (default: 5,15)",
    )

    args = parser.parse_args()

    # Parse cover color if provided
    cover_color = None
    if args.cover_color:
        try:
            cover_color = tuple(float(x) for x in args.cover_color.split(","))
            if len(cover_color) != 3:
                raise ValueError
        except (ValueError, AttributeError):
            print("Error: --cover-color must be three comma-separated values (e.g., '0.6,0.1,0.1')")
            sys.exit(1)

    # Parse rotation range if provided
    rotation_range = (-45.0, 45.0)  # Default range for random rotation
    if args.rotation_range:
        try:
            parts = args.rotation_range.split(",")
            if len(parts) != 2:
                raise ValueError
            rotation_range = (float(parts[0]), float(parts[1]))
        except (ValueError, AttributeError):
            print("Error: --rotation-range must be two comma-separated values (e.g., '-30,30')")
            sys.exit(1)

    # Parse lean angle range if provided
    lean_angle_range = (5.0, 15.0)
    if args.lean_angle_range:
        try:
            parts = args.lean_angle_range.split(",")
            if len(parts) != 2:
                raise ValueError
            lean_angle_range = (float(parts[0]), float(parts[1]))
        except (ValueError, AttributeError):
            print("Error: --lean-angle-range must be two comma-separated values (e.g., '5,15')")
            sys.exit(1)

    if args.bookshelf:
        # Generate full enclosed bookshelf with multiple shelves
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        generate_random_bookshelf(
            num_books=args.num_books,
            output_path=args.output,
            shelf_width=args.shelf_width,
            shelf_depth=args.shelf_depth,
            shelf_height=args.shelf_height,
            num_shelves=args.num_shelves,
            seed=args.seed,
            rotation=args.rotation,
            fill_percentage=args.fill_percentage,
            fill_direction=args.fill_direction,
            width_variation=args.width_variation,
            height_variation=args.height_variation,
            lean_probability=args.lean_probability,
            lean_angle_range=lean_angle_range,
        )
    elif args.num_books == 1 and args.height is not None and args.width is not None and args.depth is not None:
        # Single book with specified dimensions
        render_book(
            height=args.height,
            width=args.width,
            depth=args.depth,
            output_path=args.output,
            cover_color=cover_color,
            seed=args.seed,
            image_width=args.image_width,
            image_height=args.image_height,
            rotation=args.rotation,
            zoom=args.zoom,
        )
    elif args.num_books == 1:
        # Single book with random dimensions
        height = args.height if args.height else random.uniform(16.0, 32.0)
        width = args.width if args.width else random.uniform(12.0, 24.0)
        depth = args.depth if args.depth else random.uniform(1.5, 7.0)

        render_book(
            height=height,
            width=width,
            depth=depth,
            output_path=args.output,
            cover_color=cover_color,
            seed=args.seed,
            image_width=args.image_width,
            image_height=args.image_height,
            rotation=args.rotation,
            zoom=args.zoom,
        )
    else:
        # Multiple books with random dimensions (individual renders)
        generate_random_books(
            num_books=args.num_books,
            output_dir=args.output_dir,
            rotation=args.rotation if args.rotation != 0.0 else None,
            zoom=args.zoom,
            rotation_range=rotation_range,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

