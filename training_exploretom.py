from datasets import load_dataset
import re
import json
import ast
import random


random.seed(42)

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("facebook/ExploreToM")['train']

# Separately collect 1st and 2nd order samples
order_1_samples = []
order_2_samples = []

for i in ds:
    nth_order = i['qprop=nth_order']
    
    # Only process 1st and 2nd order samples
    if nth_order not in [1, 2]:
        continue
    
    # Get the first person name from qprop=params
    params = i['qprop=params']
    if isinstance(params, str):
        try:

            parsed_params = ast.literal_eval(params)
            if isinstance(parsed_params, tuple) and len(parsed_params) > 0 and isinstance(parsed_params[0], list) and len(parsed_params[0]) > 0:
                first_name = parsed_params[0][0]  
            else:
                continue
        except:
            continue
        
        # Replace person name with "I" in story and question
        story = i['infilled_story'].replace(first_name, 'I')
        question = i['question'].replace(first_name, 'I')
        

        new_sample = {
            "prompt": [
                {
                    "content": f"<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> <\/think> and <answer> <\/answer> tags, respectively, i.e., <think> reasoning process here <\/think><answer> answer here <\/answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> <\/answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\nRead the following story and answer the question. \nStory: {story}\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n<think>",
                    "role": "user"
                }
            ],
            "answer": i['expected_answer'],
            "data_source": "explore_tom",
            "question": question,
            "story": story,
            "ability": "theory_of_mind",
            "reward_model": {
                "ground_truth": i['expected_answer'],
                "style": "rule"
            },
            "extra_info": {
                "key": "dummy"
            }
        }
        
        # Classify by order
        if nth_order == 1:
            order_1_samples.append(new_sample)
        elif nth_order == 2:
            order_2_samples.append(new_sample)

# Randomly select 600 samples from each order
print(f"Total 1st order samples: {len(order_1_samples)}")
print(f"Total 2nd order samples: {len(order_2_samples)}")

selected_order_1 = random.sample(order_1_samples, min(600, len(order_1_samples)))
selected_order_2 = random.sample(order_2_samples, min(600, len(order_2_samples)))

print(f"Selected 1st order samples: {len(selected_order_1)}")
print(f"Selected 2nd order samples: {len(selected_order_2)}")

# Merge all selected samples
final_data = selected_order_1 + selected_order_2

# Shuffle the order
random.shuffle(final_data)

print(f"Final total samples: {len(final_data)}")

# Save as JSON file
with open('ToM_train_HiEx_hint_first_person.json', 'w') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)