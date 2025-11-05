#!/usr/bin/env python3
"""
Prepare training data for OpenAI visual fine-tuning.
Converts images organized by BIRADS categories into JSONL format.
"""

import json
import base64
from pathlib import Path
from typing import Dict


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_training_example(
    image_path: Path,
    birads_category: str,
    system_message: str = "You are a medical imaging assistant that classifies breast mammograms using BIRADS categories.",
    user_text: str = "What is the BIRADS classification for this mammogram?",
) -> Dict:
    """
    Create a single training example in OpenAI fine-tuning format.
    
    Args:
        image_path: Path to the image file
        birads_category: The BIRADS category (e.g., "1", "2", "3", "4", "5")
        system_message: System message for the conversation
        user_text: User's text prompt
        
    Returns:
        Dictionary representing the training example
    """
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Create the message structure matching OpenAI's format
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_text
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": f"BIRADS {birads_category}"
        }
    ]
    
    return {"messages": messages}


def prepare_finetuning_data(
    data_dir: Path,
    output_file: Path,
    system_message: str = None,
    user_text: str = None,
    max_images_per_category: int = 30,
) -> None:
    """
    Prepare all images for fine-tuning and write to JSONL file.
    
    Args:
        data_dir: Directory containing BIRADS category folders
        output_file: Path to output JSONL file
        system_message: Optional custom system message
        user_text: Optional custom user text prompt
        max_images_per_category: Maximum number of images to use per category (default: 30)
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    
    # Default messages
    default_system = "You are a medical imaging assistant that classifies breast mammograms using BIRADS categories."
    default_user = "What is the BIRADS classification for this mammogram?"
    
    system_msg = system_message or default_system
    user_txt = user_text or default_user
    
    # Find all BIRADS category folders
    birads_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("birads")])
    
    if not birads_folders:
        raise ValueError(f"No BIRADS category folders found in {data_dir}")
    
    training_examples = []
    category_counts = {}
    
    print(f"Processing images from {len(birads_folders)} BIRADS categories...")
    
    for birads_folder in birads_folders:
        # Extract category number (e.g., "birads1" -> "1")
        category = birads_folder.name.replace("birads", "")
        
        # Find all PNG images in this folder
        image_files = sorted(birads_folder.glob("*.png"))
        
        # Limit to max_images_per_category (or fewer if not available)
        image_files = image_files[:max_images_per_category]
        num_images = len(image_files)
        category_counts[f"BIRADS {category}"] = num_images
        
        print(f"  Processing {num_images} images from {birads_folder.name} (BIRADS {category})...")
        
        for image_path in image_files:
            example = create_training_example(
                image_path=image_path,
                birads_category=category,
                system_message=system_msg,
                user_text=user_txt,
            )
            training_examples.append(example)
    
    # Write to JSONL file (one JSON object per line)
    print(f"\nWriting {len(training_examples)} training examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✓ Successfully created {output_file}")
    print(f"  Total examples: {len(training_examples)}")
    
    # Print summary by category
    print("\nSummary by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} images")


if __name__ == "__main__":
    # Set up paths (relative to project root)
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "İnbreast"
    output_file = project_root / "training_data.jsonl"
    
    # You can customize these messages if needed
    custom_system_message = None  # Use default if None
    custom_user_text = None  # Use default if None
    
    # Example custom messages (uncomment to use):
    # custom_system_message = "You are an expert radiologist assistant specialized in breast imaging."
    # custom_user_text = "Analyze this mammogram and provide the BIRADS classification."
    
    prepare_finetuning_data(
        data_dir=data_dir,
        output_file=output_file,
        system_message=custom_system_message,
        user_text=custom_user_text,
        max_images_per_category=30,
    )
