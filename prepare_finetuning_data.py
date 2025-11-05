#!/usr/bin/env python3
"""
Prepare training data for OpenAI visual fine-tuning.
Converts images organized by BIRADS categories into JSONL format.
"""

import json
from pathlib import Path
from typing import Dict
from urllib.parse import quote


def get_github_raw_url(image_path: Path, project_root: Path, github_user: str = "thiessen2003", github_repo: str = "finetuning", branch: str = "main") -> str:
    """
    Generate GitHub raw URL for an image file.
    
    Args:
        image_path: Absolute path to the image file
        project_root: Root directory of the project
        github_user: GitHub username
        github_repo: Repository name
        branch: Branch name (default: main)
        
    Returns:
        GitHub raw URL string
    """
    # Get relative path from project root
    try:
        relative_path = image_path.relative_to(project_root)
    except ValueError:
        # If path is not relative to project root, use it as-is
        relative_path = image_path
    
    # Convert to string and URL encode each part
    path_parts = str(relative_path).replace("\\", "/").split("/")
    
    # Replace data_rgb with data in the URL (since we'll upload data_rgb as data to GitHub)
    # Or keep data_rgb if you want to maintain separate folders
    # For now, we'll replace data_rgb with data for GitHub compatibility
    path_parts = [part.replace("data_rgb", "data") if "data_rgb" in part else part for part in path_parts]
    
    encoded_parts = [quote(part, safe="") for part in path_parts]
    encoded_path = "/".join(encoded_parts)
    
    # Construct GitHub raw URL
    return f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{branch}/{encoded_path}"


def create_training_example(
    image_url: str,
    birads_category: str,
    system_message: str = "You are a medical imaging assistant that classifies breast mammograms using BIRADS categories.",
    user_text: str = "What is the BIRADS classification for this mammogram?",
) -> Dict:
    """
    Create a single training example in OpenAI fine-tuning format.
    
    Args:
        image_url: GitHub raw URL to the image
        birads_category: The BIRADS category (e.g., "1", "2", "3", "4", "5")
        system_message: System message for the conversation
        user_text: User's text prompt
        
    Returns:
        Dictionary representing the training example
    """
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
                        "url": image_url
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
    project_root: Path,
    system_message: str = None,
    user_text: str = None,
    max_images_per_category: int = 30,
    github_user: str = "thiessen2003",
    github_repo: str = "finetuning",
    branch: str = "main",
) -> None:
    """
    Prepare all images for fine-tuning and write to JSONL file.
    
    Args:
        data_dir: Directory containing BIRADS category folders
        output_file: Path to output JSONL file
        project_root: Root directory of the project (for generating GitHub URLs)
        system_message: Optional custom system message
        user_text: Optional custom user text prompt
        max_images_per_category: Maximum number of images to use per category (default: 30)
        github_user: GitHub username (default: thiessen2003)
        github_repo: Repository name (default: finetuning)
        branch: Branch name (default: main)
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    project_root = Path(project_root)
    
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
            # Generate GitHub raw URL
            image_url = get_github_raw_url(
                image_path=image_path,
                project_root=project_root,
                github_user=github_user,
                github_repo=github_repo,
                branch=branch,
            )
            
            example = create_training_example(
                image_url=image_url,
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
    # Use data_rgb folder which contains RGB/RGBA converted images
    data_dir = project_root / "data_rgb" / "İnbreast"
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
        project_root=project_root,
        system_message=custom_system_message,
        user_text=custom_user_text,
        max_images_per_category=30,
        github_user="thiessen2003",
        github_repo="finetuning",
        branch="main",
    )
