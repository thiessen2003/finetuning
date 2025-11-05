#!/usr/bin/env python3
"""
Convert all PNG images to RGB/RGBA format for OpenAI fine-tuning.
Preserves original images and writes converted versions to a new folder.
"""

from PIL import Image
from pathlib import Path
import sys
import shutil


def convert_image_to_rgb(source_path: Path, dest_path: Path) -> tuple[bool, str]:
    """
    Convert an image to RGB or RGBA format and save to destination.
    
    Args:
        source_path: Path to the source image file
        dest_path: Path where the converted image should be saved
        
    Returns:
        Tuple of (was_converted: bool, mode: str)
    """
    try:
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Image.open(source_path) as img:
            original_mode = img.mode
            
            # Check if conversion is needed
            if img.mode in ('RGB', 'RGBA'):
                # Already in correct format, just copy the file
                shutil.copy2(source_path, dest_path)
                return False, original_mode
            
            # Convert to RGB or RGBA based on transparency
            if img.mode in ('P', 'L', 'LA'):
                # Convert palette/grayscale to RGB
                if img.mode == 'P':
                    # Check if image has transparency
                    if 'transparency' in img.info:
                        new_img = img.convert('RGBA')
                    else:
                        new_img = img.convert('RGB')
                elif img.mode == 'LA':
                    new_img = img.convert('RGBA')
                else:  # L (grayscale)
                    new_img = img.convert('RGB')
                
                # Save the converted image
                new_img.save(dest_path, 'PNG', optimize=True)
                return True, f"{original_mode} -> {new_img.mode}"
            else:
                # For other modes, convert to RGB
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                else:
                    rgb_img.paste(img)
                rgb_img.save(dest_path, 'PNG', optimize=True)
                return True, f"{original_mode} -> RGB"
    except Exception as e:
        print(f"Error converting {source_path}: {e}")
        return False, f"ERROR: {e}"


def convert_all_images(source_dir: Path, output_dir: Path):
    """
    Convert all PNG images in the source directory and save to output directory.
    
    Args:
        source_dir: Directory containing original images
        output_dir: Directory where converted images will be saved
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    converted_count = 0
    copied_count = 0
    error_count = 0
    
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scanning images...\n")
    
    for birads_folder in sorted(source_dir.glob('birads*')):
        folder_name = birads_folder.name
        output_folder = output_dir / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {folder_name}...")
        folder_converted = 0
        folder_copied = 0
        
        for image_path in sorted(birads_folder.glob('*.png')):
            dest_path = output_folder / image_path.name
            was_converted, mode_info = convert_image_to_rgb(image_path, dest_path)
            
            if "ERROR" in mode_info:
                error_count += 1
                print(f"  ✗ Error with {image_path.name}: {mode_info}")
            elif was_converted:
                converted_count += 1
                folder_converted += 1
                print(f"  ✓ Converted: {image_path.name} ({mode_info})")
            else:
                copied_count += 1
                folder_copied += 1
        
        print(f"  Summary: {folder_converted} converted, {folder_copied} copied (already RGB/RGBA)\n")
    
    print(f"{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  Converted: {converted_count} images")
    print(f"  Copied (already RGB/RGBA): {copied_count} images")
    if error_count > 0:
        print(f"  Errors: {error_count} images")
    print(f"\nConverted images saved to: {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).parent
    source_dir = project_root / "data" / "İnbreast"
    output_dir = project_root / "data_rgb" / "İnbreast"
    
    if not source_dir.exists():
        print(f"Error: Source directory not found at {source_dir}")
        sys.exit(1)
    
    # Ask for confirmation
    print(f"This will convert images from:")
    print(f"  {source_dir}")
    print(f"To:")
    print(f"  {output_dir}")
    print(f"\nOriginal images will be preserved.\n")
    
    convert_all_images(source_dir, output_dir)
