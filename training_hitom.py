import json
import random

# Set random seed to ensure reproducibility
random.seed(42)

def convert_data_processed_to_hiex_format(data_processed_sample):

    
    # Build complete story text (including story and choices)
    story_text = data_processed_sample['story']
    question_text = data_processed_sample['question']
    choices_text = data_processed_sample['choices']
    
    # Build complete question, including choice options
    full_question = f"{question_text}\n{choices_text}"
    

    new_sample = {
        "prompt": [
            {
                "content": f"<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\nRead the following story and answer the question. \nStory: {story_text}\nQuestion: {full_question}\n<|im_end|>\n<|im_start|>assistant\n<think>",
                "role": "user"
            }
        ],
        "answer": data_processed_sample['answer'],
        "data_source": "hi_tom",
        "question": full_question,
        "story": story_text,
        "ability": "theory_of_mind",
        "reward_model": {
            "ground_truth": data_processed_sample['answer'],
            "style": "rule"
        },
        "extra_info": {
            "key": "dummy",
        }
    }
    
    return new_sample

print("Reading hitom_first.json...")
with open('hitom_first.json', 'r', encoding='utf-8') as f:
    data_processed = json.load(f)

# Convert all samples
print(f"Converting {len(data_processed['data'])} samples...")
converted_samples = []
for sample in data_processed['data']:
    converted_sample = convert_data_processed_to_hiex_format(sample)
    converted_samples.append(converted_sample)

print(f"Conversion completed, total {len(converted_samples)} samples")

# Read existing ToM_train_HiEx_hint_first_person.json (if exists)
try:
    with open('ToM_train_HiEx_hint_first_person.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    print(f"Read existing file, contains {len(existing_data)} samples")
except FileNotFoundError:
    existing_data = []
    print("Existing file not found, creating new file")

# Merge data
combined_data = existing_data + converted_samples
print(f"After merging, total {len(combined_data)} samples")

# Shuffle the order
random.shuffle(combined_data)

with open('ToM_train_HiEx_hint_first_person.json', 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"Saved to ToM_train_HiEx_hint_first_person.json, total {len(combined_data)} samples")

# Statistics of data sources
data_sources = {}
for sample in combined_data:
    source = sample.get('data_source', 'unknown')
    data_sources[source] = data_sources.get(source, 0) + 1

print("\nData source statistics:")
for source, count in data_sources.items():
    print(f"  {source}: {count} samples")
