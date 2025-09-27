from datasets import load_dataset
import re
import json
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("facebook/ExploreToM")['train']
new_data = []
for i in ds:
    if i['qprop=nth_order'] < 0:
        continue
    
    params = i['qprop=params']
    if isinstance(params, str):
        import ast
        try:
            parsed_params = ast.literal_eval(params)
            if isinstance(parsed_params, tuple) and len(parsed_params) > 0 and isinstance(parsed_params[0], list) and len(parsed_params[0]) > 0:
                first_name = parsed_params[0][0] 
            else:
                continue
        except:
            continue
        
        story = i['infilled_story'].replace(first_name, 'I')
        question = i['question'].replace(first_name, 'I')
        
        new_sample = {}
        new_sample['instruction'] = 'Answer the question based on the story.'
        new_sample['input'] = 'Story: ' + story + '\n\nQuestion: ' + question
        new_sample['output'] = i['expected_answer']
        new_data.append(new_sample)

with open('exploretom_first.jsonl', 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)