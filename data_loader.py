from datasets import load_dataset

def format_prompt(example):
    """
    Formats a single example from the dataset into a standardized prompt format.
    Handles cases where the 'input' field is empty or contains '<noinput>'.
    """
    if example.get("input") and example['input'].lower() != '<noinput>':
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

def load_and_preprocess_data(dataset_name="lavita/AlpaCare-MedInstruct-52k"):
    """
    Loads the dataset, preprocesses it, and splits it into train, validation, and test sets.
    
    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, split="train")
    
    # Shuffle the dataset for random splitting
    shuffled_dataset = dataset.shuffle(seed=42)
    
    # Split 90% for training
    train_test_split = shuffled_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    
    # Split the remaining 10% into 5% validation and 5% test
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5)
    validation_dataset = test_val_split['train']
    test_dataset = test_val_split['test']
    
    print(f"Data loading and splitting complete.")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_dataset, validation_dataset, test_dataset

if __name__ == '__main__':
    # This block allows you to run the script directly to test it
    train_ds, val_ds, test_ds = load_and_preprocess_data()
    print("\n--- Sample formatted prompt ---")
    # You need to apply the formatting function when you use the data,
    # for example, in the SFTTrainer.
    print(format_prompt(train_ds[0]))