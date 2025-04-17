import json
from datasets import load_dataset
import pandas as pd
import os


def export_husst_dataset():
    try:
        # Load dataset
        dataset = load_dataset("NYTK/HuSST")

        # Create output directory
        os.makedirs("husst_export", exist_ok=True)

        # Export each split
        for split in ['train', 'validation', 'test']:
            # Convert to list of dictionaries
            data = []
            for example in dataset[split]:
                data.append({
                    "text": example["sentence"],
                    "label": example["label"],
                    # Add other fields if available
                    # "id": example.get("id", ""),
                })

            # Save to JSON
            json_path = f"husst_export/{split}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Save to CSV
            csv_path = f"husst_export/{split}.csv"
            pd.DataFrame(data).to_csv(csv_path, index=False, encoding='utf-8')

            print(f"Exported {split} set to:")
            print(f"- {json_path}")
            print(f"- {csv_path}")

        print("\nExport completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    export_husst_dataset()

