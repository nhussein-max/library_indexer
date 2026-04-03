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

    # Build the book's rotation transform around the Y axis
    # This is applied to all book parts so they rotate together
    book_rotation = mi.ScalarTransform4f.rotate([0, 1, 0], rotation)

    # Camera positioning:
    # Default: spine faces the camera (spine is on negative X side of the book)
    # The base camera position is from the left side (negative X) looking at the book
    # Zoom scales the distance from the book
    base_camera_distance_x = width * 1.2
    base_camera_distance_z = depth * 2.0
    base_camera_height = height * 0.8

    camera_x = -base_camera_distance_x * zoom  # Negative X = facing the spine
    camera_y = base_camera_height
    camera_z = base_camera_distance_z * zoom

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
        # Rotation is applied so the book can be turned to show different sides
        "book_body": {
            "type": "cube",
            "to_world": book_rotation
            @ mi.ScalarTransform4f.translate([0, book_scale_y, 0])
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
            "to_world": book_rotation
            @ mi.ScalarTransform4f.translate(
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
            "to_world": book_rotation
            @ mi.ScalarTransform4f.translate(
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
            "to_world": book_rotation
            @ mi.ScalarTransform4f.translate(
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
            "to_world": book_rotation
            @ mi.ScalarTransform4f.translate([0, 0.1, 0])
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
            "to_world": book_rotation
            @ mi.ScalarTransform4f.translate(
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


def generate_random_books(
    num_books: int = 5,
    output_dir: str = "book_renders",
    rotation: float = None,
    zoom: float = 1.0,
    rotation_range: tuple = (-45.0, 45.0),
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
    """
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

    if args.num_books == 1 and args.height is not None and args.width is not None and args.depth is not None:
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
        # Multiple books with random dimensions
        generate_random_books(
            num_books=args.num_books,
            output_dir=args.output_dir,
            rotation=args.rotation if args.rotation != 0.0 else None,
            zoom=args.zoom,
            rotation_range=rotation_range,
        )


if __name__ == "__main__":
    main()
